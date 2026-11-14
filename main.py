#!/usr/bin/env python3
"""free-model-aggregator-01: the best free AI model aggregator.

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API (each in JSON and in server-sent events) over the ten simulated
free-tier providers a make.fun trial hands it. For every client request it
picks the highest-rated free model that meets the request's requirements,
speaks that provider's own wire (OpenAI, Anthropic or Gemini), fails over to
the next candidate when a provider refuses, and answers in the client's wire.

The trial brief (MAKEFUN_PROMPT_FILE) carries the provider directory, the
quality table and the workload manifest. The catalog the server registers at
POST $MAKEFUN_ENV_URL/register is built by probing every provider's own model
list, and `python3 main.py --list` prints that same document byte for byte.

Everything the server answers is a pure function of the brief, the requests
and the providers' answers: no wall clock (fixed listing timestamps), no
randomness. Python 3.13 standard library only.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED = 1767225600  # fixed epoch for OpenAI-shaped listings/answers
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"  # fixed instant for Anthropic shapes
SSE_PIECE = 13  # characters per streamed text piece, as the providers stream
PROTOCOL_VERSION = 1

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_rec(v):
    return isinstance(v, dict)


def dumps(v):
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False)


def http_call(method, url, headers=None, payload=None, timeout=20):
    """One HTTP call; returns (status, body bytes). Status 0 on transport error."""
    hdrs = dict(headers or {})
    data = None
    if payload is not None:
        data = payload if isinstance(payload, bytes) else dumps(payload).encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read()
        except Exception:
            return e.code, b""
    except Exception:
        return 0, b""


# ---------------------------------------------------------------- the brief

def extract_docs(text):
    """The brief is prose carrying JSON documents; pull out each one that
    names a schema, keyed by that schema."""
    docs = {}
    dec = json.JSONDecoder()
    i = 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return docs
        try:
            obj, end = dec.raw_decode(text, i)
        except Exception:
            i += 1
            continue
        if is_rec(obj) and isinstance(obj.get("schema"), str):
            docs[obj["schema"]] = obj
            i = end
        else:
            i += 1


class Brief:
    def __init__(self, text):
        docs = extract_docs(text)
        prov = docs.get("free-model-aggregator-01/providers/v1", {})
        rate = docs.get("free-model-aggregator-01/ratings/v1", {})
        work = docs.get("free-model-aggregator-01/workload/v1", {})
        self.providers = prov.get("providers", [])
        self.ratings = {}
        for row in rate.get("ratings", []):
            self.ratings[(row.get("provider"), row.get("model"))] = row
        self.manifest = work.get("manifest", [])
        self.manifest_by_id = {m.get("id"): m for m in self.manifest}


def load_brief():
    path = os.environ.get("MAKEFUN_PROMPT_FILE")
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as fh:
            return Brief(fh.read())
    except Exception:
        return None


# ---------------------------------------------------------------- providers

def auth_headers(provider):
    auth = provider.get("auth")
    key = provider.get("key") or ""
    if auth == "bearer":
        return {"authorization": f"Bearer {key}"}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def models_route(provider):
    if provider.get("wire") == "gemini":
        return "/v1beta/models"
    return "/v1/models"


def probe_provider(env_url, provider):
    """The provider's own model list as [{id, free, context_length, tools,
    vision, json}] in the order the provider lists them; None if unreachable."""
    url = env_url + provider.get("path", "") + models_route(provider)
    status, raw = http_call("GET", url, headers=auth_headers(provider), timeout=10)
    if status != 200:
        return None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    items = []
    if provider.get("wire") == "gemini":
        rows = doc.get("models") if is_rec(doc) else None
        for row in rows or []:
            if not is_rec(row):
                continue
            name = row.get("name") or ""
            mid = name[len("models/"):] if name.startswith("models/") else row.get("id")
            if isinstance(mid, str) and mid:
                items.append((mid, row))
    else:
        rows = doc.get("data") if is_rec(doc) else None
        for row in rows or []:
            if is_rec(row) and isinstance(row.get("id"), str):
                items.append((row["id"], row))
    out = []
    for mid, row in items:
        ctx = row.get("context_length")
        if not isinstance(ctx, (int, float)) and isinstance(row.get("inputTokenLimit"), (int, float)):
            ctx = row.get("inputTokenLimit")
        out.append({
            "id": mid,
            "free": row.get("free") if isinstance(row.get("free"), bool) else None,
            "context_length": int(ctx) if isinstance(ctx, (int, float)) else None,
            "tools": row.get("tools") is True,
            "vision": row.get("vision") is True,
            "json": row.get("json") is True,
        })
    return out


class World:
    """Everything the trial hands the server, plus what the probes answered."""

    def __init__(self, brief, env_url):
        self.brief = brief
        self.env_url = env_url
        self.catalog = {"providers": []}
        self.models = {}  # (provider_id, model_id) -> capability row
        self.free_ids = []  # unique free model ids, catalog order
        if brief is None:
            return
        for provider in brief.providers:
            pid = provider.get("id")
            probed = probe_provider(env_url, provider) if env_url else None
            entries = []
            if probed:
                for item in probed:
                    rating = brief.ratings.get((pid, item["id"]))
                    free = item["free"]
                    if free is None:
                        free = rating is not None and rating.get("free") is True
                    ctx = item["context_length"]
                    if ctx is None and rating is not None:
                        ctx = rating.get("context_length")
                    entries.append({"id": item["id"], "free": bool(free), "context_length": ctx})
                    self.models[(pid, item["id"])] = {
                        "rating": rating.get("rating_milli", 0) if rating else 0,
                        "context_length": ctx,
                        "tools": item["tools"] or (rating.get("tools") if rating else False),
                        "vision": item["vision"] or (rating.get("vision") if rating else False),
                        "json": item["json"] or (rating.get("json") if rating else False),
                        "free": bool(free),
                    }
            else:
                for (p, mid), rating in sorted(brief.ratings.items()):
                    if p != pid:
                        continue
                    entries.append({"id": mid, "free": True, "context_length": rating.get("context_length")})
                    self.models[(pid, mid)] = {
                        "rating": rating.get("rating_milli", 0),
                        "context_length": rating.get("context_length"),
                        "tools": rating.get("tools") is True,
                        "vision": rating.get("vision") is True,
                        "json": rating.get("json") is True,
                        "free": True,
                    }
            self.catalog["providers"].append({
                "id": pid,
                "base_url": provider.get("path"),
                "key_env": provider.get("key_env", ""),
                "wire": provider.get("wire"),
                "models": entries,
            })
            for e in entries:
                if e["free"] and e["id"] not in self.free_ids:
                    self.free_ids.append(e["id"])
        self.provider_by_id = {p.get("id"): p for p in brief.providers}
        self.provider_order = {p.get("id"): i for i, p in enumerate(brief.providers)}


# ------------------------------------------------------- request normal form

def text_blocks(content, joiner="\n"):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_rec(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return joiner.join(parts)


def image_part_from_url(url):
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        head, data = url[5:].split(";base64,", 1)
        return {"kind": "image", "media_type": head or "image/png", "data": data}
    return {"kind": "image_url", "url": url}


def args_object(value):
    if is_rec(value):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if is_rec(parsed):
                return parsed
        except Exception:
            pass
    return {}


def normalize(client_api, body):
    """A client request in one neutral shape both upstream builders read:
    system text, turns of typed parts, tools, caps, stops, json mode, stream."""
    norm = {
        "system": "",
        "turns": [],
        "tools": [],
        "tool_choice": None,
        "max_tokens": None,
        "temperature": None,
        "top_p": None,
        "stops": [],
        "json_mode": False,
        "stream": body.get("stream") is True,
    }
    if client_api == "openai":
        system_parts = []
        for msg in body.get("messages") or []:
            if not is_rec(msg):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                text = text_blocks(content)
                if text != "":
                    system_parts.append(text)
                continue
            if role == "tool":
                norm["turns"].append({"role": "user", "parts": [{
                    "kind": "tool_result", "id": msg.get("tool_call_id") or "",
                    "text": content if isinstance(content, str) else dumps(content),
                }]})
                continue
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"kind": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not is_rec(part):
                        continue
                    if part.get("type") == "text":
                        parts.append({"kind": "text", "text": part.get("text") or ""})
                    elif part.get("type") == "image_url" and is_rec(part.get("image_url")):
                        parts.append(image_part_from_url(part["image_url"].get("url")))
            if role == "assistant":
                for call in msg.get("tool_calls") or []:
                    if is_rec(call) and is_rec(call.get("function")):
                        parts.append({
                            "kind": "tool_call", "id": call.get("id") or "",
                            "name": call["function"].get("name") or "",
                            "args": args_object(call["function"].get("arguments")),
                        })
            norm["turns"].append({"role": "assistant" if role == "assistant" else "user", "parts": parts})
        norm["system"] = "\n".join(system_parts)
        for tool in body.get("tools") or []:
            if is_rec(tool) and is_rec(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                fn = tool["function"]
                norm["tools"].append({
                    "name": fn["name"],
                    "description": fn.get("description") or "",
                    "parameters": fn.get("parameters") if is_rec(fn.get("parameters")) else {"type": "object", "properties": {}},
                })
        choice = body.get("tool_choice")
        if choice == "auto":
            norm["tool_choice"] = {"type": "auto"}
        elif choice == "required":
            norm["tool_choice"] = {"type": "any"}
        elif is_rec(choice) and is_rec(choice.get("function")):
            norm["tool_choice"] = {"type": "tool", "name": choice["function"].get("name") or ""}
        mt = body.get("max_tokens", body.get("max_completion_tokens"))
        if isinstance(mt, (int, float)):
            norm["max_tokens"] = mt
        stop = body.get("stop")
        if isinstance(stop, str):
            norm["stops"] = [stop]
        elif isinstance(stop, list):
            norm["stops"] = [s for s in stop if isinstance(s, str)]
        fmt = body.get("response_format")
        if is_rec(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            norm["json_mode"] = True
    else:  # anthropic
        norm["system"] = text_blocks(body.get("system"))
        for msg in body.get("messages") or []:
            if not is_rec(msg):
                continue
            role = "assistant" if msg.get("role") == "assistant" else "user"
            content = msg.get("content")
            parts = []
            if isinstance(content, str):
                if content != "":
                    parts.append({"kind": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not is_rec(block):
                        continue
                    kind = block.get("type")
                    if kind == "text":
                        parts.append({"kind": "text", "text": block.get("text") or ""})
                    elif kind == "image" and is_rec(block.get("source")):
                        src = block["source"]
                        if src.get("type") == "base64":
                            parts.append({"kind": "image", "media_type": src.get("media_type") or "image/png", "data": src.get("data") or ""})
                        elif src.get("type") == "url":
                            parts.append({"kind": "image_url", "url": src.get("url") or ""})
                    elif kind == "tool_use":
                        parts.append({
                            "kind": "tool_call", "id": block.get("id") or "",
                            "name": block.get("name") or "",
                            "args": block.get("input") if is_rec(block.get("input")) else {},
                        })
                    elif kind == "tool_result":
                        inner = block.get("content")
                        parts.append({
                            "kind": "tool_result", "id": block.get("tool_use_id") or "",
                            "text": inner if isinstance(inner, str) else text_blocks(inner) or dumps(inner),
                        })
            norm["turns"].append({"role": role, "parts": parts})
        for tool in body.get("tools") or []:
            if is_rec(tool) and isinstance(tool.get("name"), str):
                norm["tools"].append({
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("input_schema") if is_rec(tool.get("input_schema")) else {"type": "object", "properties": {}},
                })
        choice = body.get("tool_choice")
        if is_rec(choice):
            if choice.get("type") == "auto":
                norm["tool_choice"] = {"type": "auto"}
            elif choice.get("type") == "any":
                norm["tool_choice"] = {"type": "any"}
            elif choice.get("type") == "tool":
                norm["tool_choice"] = {"type": "tool", "name": choice.get("name") or ""}
        if isinstance(body.get("max_tokens"), (int, float)):
            norm["max_tokens"] = body["max_tokens"]
        stops = body.get("stop_sequences")
        if isinstance(stops, list):
            norm["stops"] = [s for s in stops if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        norm["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        norm["top_p"] = body["top_p"]
    return norm


# --------------------------------------------------------- upstream builders

def build_openai(norm, model):
    messages = []
    if norm["system"] != "":
        messages.append({"role": "system", "content": norm["system"]})
    for turn in norm["turns"]:
        if turn["role"] == "assistant":
            text = "".join(p["text"] for p in turn["parts"] if p["kind"] == "text")
            calls = [{
                "id": p["id"], "type": "function",
                "function": {"name": p["name"], "arguments": dumps(p["args"])},
            } for p in turn["parts"] if p["kind"] == "tool_call"]
            msg = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
            continue
        for p in turn["parts"]:
            if p["kind"] == "tool_result":
                messages.append({"role": "tool", "tool_call_id": p["id"], "content": p["text"]})
        parts = []
        for p in turn["parts"]:
            if p["kind"] == "text":
                parts.append({"type": "text", "text": p["text"]})
            elif p["kind"] == "image":
                parts.append({"type": "image_url", "image_url": {"url": f"data:{p['media_type']};base64,{p['data']}"}})
            elif p["kind"] == "image_url":
                parts.append({"type": "image_url", "image_url": {"url": p["url"]}})
        if parts:
            if len(parts) == 1 and parts[0]["type"] == "text":
                messages.append({"role": "user", "content": parts[0]["text"]})
            else:
                messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if norm["max_tokens"] is not None:
        out["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        out["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        out["top_p"] = norm["top_p"]
    if norm["stops"]:
        out["stop"] = norm["stops"]
    if norm["tools"]:
        out["tools"] = [{
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
        } for t in norm["tools"]]
    choice = norm["tool_choice"]
    if choice is not None:
        if choice["type"] == "auto":
            out["tool_choice"] = "auto"
        elif choice["type"] == "any":
            out["tool_choice"] = "required"
        elif choice["type"] == "tool":
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if norm["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    if norm["stream"]:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def build_anthropic(norm, model):
    messages = []
    for turn in norm["turns"]:
        blocks = []
        for p in turn["parts"]:
            if p["kind"] == "text":
                blocks.append({"type": "text", "text": p["text"]})
            elif p["kind"] == "image":
                blocks.append({"type": "image", "source": {"type": "base64", "media_type": p["media_type"], "data": p["data"]}})
            elif p["kind"] == "image_url":
                blocks.append({"type": "image", "source": {"type": "url", "url": p["url"]}})
            elif p["kind"] == "tool_call":
                blocks.append({"type": "tool_use", "id": p["id"], "name": p["name"], "input": p["args"]})
            elif p["kind"] == "tool_result":
                blocks.append({"type": "tool_result", "tool_use_id": p["id"], "content": p["text"]})
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            messages.append({"role": turn["role"], "content": blocks[0]["text"]})
        elif blocks:
            messages.append({"role": turn["role"], "content": blocks})
    out = {"model": model, "messages": messages}
    if norm["system"] != "":
        out["system"] = norm["system"]
    if norm["max_tokens"] is not None:
        out["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        out["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        out["top_p"] = norm["top_p"]
    if norm["stops"]:
        out["stop_sequences"] = norm["stops"]
    if norm["tools"]:
        out["tools"] = [{
            "name": t["name"], "description": t["description"], "input_schema": t["parameters"],
        } for t in norm["tools"]]
    if norm["tool_choice"] is not None:
        out["tool_choice"] = norm["tool_choice"]
    if norm["stream"]:
        out["stream"] = True
    return out


def build_gemini(norm, model):
    contents = []
    for turn in norm["turns"]:
        parts = []
        for p in turn["parts"]:
            if p["kind"] == "text":
                parts.append({"text": p["text"]})
            elif p["kind"] == "image":
                parts.append({"inlineData": {"mimeType": p["media_type"], "data": p["data"]}})
            elif p["kind"] == "image_url":
                parts.append({"fileData": {"fileUri": p["url"]}})
            elif p["kind"] == "tool_call":
                parts.append({"functionCall": {"name": p["name"], "args": p["args"]}})
            elif p["kind"] == "tool_result":
                parts.append({"functionResponse": {"name": p["id"], "response": {"result": p["text"]}}})
        if parts:
            contents.append({"role": "model" if turn["role"] == "assistant" else "user", "parts": parts})
    out = {"contents": contents}
    if norm["system"] != "":
        out["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
    config = {}
    if norm["max_tokens"] is not None:
        config["maxOutputTokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        config["temperature"] = norm["temperature"]
    if norm["stops"]:
        config["stopSequences"] = norm["stops"]
    if norm["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    if norm["tools"]:
        out["tools"] = [{"functionDeclarations": [{
            "name": t["name"], "description": t["description"], "parameters": t["parameters"],
        } for t in norm["tools"]]}]
    return out


# --------------------------------------------------------- upstream answers

def sse_frames(text):
    frames = []
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def answer(text, input_tokens, output_tokens, stop):
    return {"text": text, "input": input_tokens, "output": output_tokens, "stop": stop}


def num(v):
    return v if isinstance(v, (int, float)) else 0


def parse_openai(raw, stream):
    if stream:
        text, stop, usage = "", None, {}
        for _event, data in sse_frames(raw.decode("utf-8", "replace")):
            if data.strip() == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except Exception:
                continue
            if not is_rec(chunk):
                continue
            if is_rec(chunk.get("usage")):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and is_rec(choices[0]):
                delta = choices[0].get("delta") or {}
                if isinstance(delta.get("content"), str):
                    text += delta["content"]
                if isinstance(choices[0].get("finish_reason"), str):
                    stop = choices[0]["finish_reason"]
        return answer(text, num(usage.get("prompt_tokens")), num(usage.get("completion_tokens")), stop or "stop")
    doc = json.loads(raw.decode("utf-8"))
    choices = doc.get("choices") or []
    choice = choices[0] if choices and is_rec(choices[0]) else {}
    message = choice.get("message") or {}
    content = message.get("content")
    text = content if isinstance(content, str) else text_blocks(content, joiner="")
    usage = doc.get("usage") or {}
    return answer(text, num(usage.get("prompt_tokens")), num(usage.get("completion_tokens")), choice.get("finish_reason") or "stop")


def parse_anthropic(raw, stream):
    if stream:
        text, stop = "", None
        in_tokens, out_tokens = 0, 0
        for event, data in sse_frames(raw.decode("utf-8", "replace")):
            try:
                doc = json.loads(data)
            except Exception:
                continue
            if not is_rec(doc):
                continue
            if event == "message_start":
                usage = (doc.get("message") or {}).get("usage") or {}
                in_tokens = num(usage.get("input_tokens"))
            elif event == "content_block_delta":
                delta = doc.get("delta") or {}
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    text += delta["text"]
            elif event == "message_delta":
                if isinstance((doc.get("delta") or {}).get("stop_reason"), str):
                    stop = doc["delta"]["stop_reason"]
                usage = doc.get("usage") or {}
                out_tokens = num(usage.get("output_tokens"))
        return answer(text, in_tokens, out_tokens, stop or "end_turn")
    doc = json.loads(raw.decode("utf-8"))
    text = text_blocks(doc.get("content"), joiner="")
    usage = doc.get("usage") or {}
    return answer(text, num(usage.get("input_tokens")), num(usage.get("output_tokens")), doc.get("stop_reason") or "end_turn")


def parse_gemini(raw, stream):
    def from_doc(doc):
        text, stop = "", None
        for cand in doc.get("candidates") or []:
            if not is_rec(cand):
                continue
            for part in (cand.get("content") or {}).get("parts") or []:
                if is_rec(part) and isinstance(part.get("text"), str):
                    text += part["text"]
            if isinstance(cand.get("finishReason"), str):
                stop = cand["finishReason"]
        usage = doc.get("usageMetadata") or {}
        return text, stop, num(usage.get("promptTokenCount")), num(usage.get("candidatesTokenCount"))

    if stream:
        text, stop, in_t, out_t = "", None, 0, 0
        for _event, data in sse_frames(raw.decode("utf-8", "replace")):
            try:
                doc = json.loads(data)
            except Exception:
                continue
            if not is_rec(doc):
                continue
            t, s, i2, o2 = from_doc(doc)
            text += t
            if s is not None:
                stop = s
            if i2:
                in_t = i2
            if o2:
                out_t = o2
        return answer(text, in_t, out_t, stop or "STOP")
    text, stop, in_t, out_t = from_doc(json.loads(raw.decode("utf-8")))
    return answer(text, in_t, out_t, stop or "STOP")


PARSERS = {"openai": parse_openai, "anthropic": parse_anthropic, "gemini": parse_gemini}
BUILDERS = {"openai": build_openai, "anthropic": build_anthropic, "gemini": build_gemini}


def call_provider(world, provider, model, client_api, body, norm):
    """One provider call for this request; the parsed answer, or None when the
    provider refused (any non-200) or its answer could not be read."""
    wire = provider.get("wire")
    base = world.env_url + provider.get("path", "")
    headers = auth_headers(provider)
    stream = norm["stream"]
    if wire == "gemini":
        verb = "streamGenerateContent" if stream else "generateContent"
        url = f"{base}/v1beta/models/{model}:{verb}"
        payload = build_gemini(norm, model)
    elif wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = "2023-06-01"
        payload = build_anthropic(norm, model) if client_api != "anthropic" else dict(body, model=model)
    else:
        url = base + "/v1/chat/completions"
        if client_api == "openai":
            payload = dict(body, model=model)
            if payload.get("stream") is True:
                opts = dict(payload.get("stream_options") or {})
                opts["include_usage"] = True
                payload["stream_options"] = opts
        else:
            payload = build_openai(norm, model)
    status, raw = http_call("POST", url, headers=headers, payload=payload, timeout=25)
    if status != 200:
        return None
    try:
        return PARSERS[wire](raw, stream)
    except Exception:
        return None


# ------------------------------------------------------------ client answers

def map_stop(stop, client_api):
    family = {
        "stop": 0, "end_turn": 0, "STOP": 0,
        "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
        "tool_calls": 2, "function_call": 2, "tool_use": 2,
        "content_filter": 3, "refusal": 3, "SAFETY": 3,
    }.get(stop, 0)
    if client_api == "openai":
        return ["stop", "length", "tool_calls", "content_filter"][family]
    return ["end_turn", "max_tokens", "tool_use", "refusal"][family]


def pieces(text):
    out = [text[i:i + SSE_PIECE] for i in range(0, len(text), SSE_PIECE)]
    return out if out else [""]


def emit_openai(ans, model, stream):
    stop = map_stop(ans["stop"], "openai")
    usage = {
        "prompt_tokens": ans["input"],
        "completion_tokens": ans["output"],
        "total_tokens": ans["input"] + ans["output"],
    }
    if not stream:
        return "application/json", dumps({
            "id": "chatcmpl-agg",
            "object": "chat.completion",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ans["text"]},
                "finish_reason": stop,
                "logprobs": None,
            }],
            "usage": usage,
        }).encode("utf-8")
    frames = []

    def chunk(delta, finish):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED,
                "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames.append(f"data: {dumps(chunk({'role': 'assistant', 'content': ''}, None))}\n\n")
    for piece in pieces(ans["text"]):
        frames.append(f"data: {dumps(chunk({'content': piece}, None))}\n\n")
    frames.append(f"data: {dumps(chunk({}, stop))}\n\n")
    frames.append(f"data: {dumps({'id': 'chatcmpl-agg', 'object': 'chat.completion.chunk', 'created': FIXED_CREATED, 'model': model, 'choices': [], 'usage': usage})}\n\n")
    frames.append("data: [DONE]\n\n")
    return "text/event-stream", "".join(frames).encode("utf-8")


def emit_anthropic(ans, model, stream):
    stop = map_stop(ans["stop"], "anthropic")
    if not stream:
        return "application/json", dumps({
            "id": "msg_agg",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": ans["text"]}],
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": ans["input"], "output_tokens": ans["output"]},
        }).encode("utf-8")
    frames = []

    def frame(event, doc):
        return f"event: {event}\ndata: {dumps(doc)}\n\n"

    frames.append(frame("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_agg", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": ans["input"], "output_tokens": 0},
        },
    }))
    frames.append(frame("content_block_start", {
        "type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""},
    }))
    for piece in pieces(ans["text"]):
        frames.append(frame("content_block_delta", {
            "type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": ans["output"]},
    }))
    frames.append(frame("message_stop", {"type": "message_stop"}))
    return "text/event-stream", "".join(frames).encode("utf-8")


def emit_error(client_api, stream, message):
    if client_api == "openai":
        doc = {"error": {"message": message, "type": "api_error", "code": "no_provider_available"}}
    else:
        doc = {"type": "error", "error": {"type": "api_error", "message": message}}
    return "application/json", dumps(doc).encode("utf-8")


# ------------------------------------------------------------------ routing

class Router:
    def __init__(self, world):
        self.world = world
        self.served_count = 0
        # Per-provider trial memory, derived only from provider answers seen
        # so far (so a replay rebuilds the same state): a provider that just
        # refused is suspect until it answers again, and a proven provider
        # that refuses once may simply have hit its failure schedule.
        self.provider_state = {}

    def state_of(self, pid):
        return self.provider_state.setdefault(pid, {"refusals": 0, "successes": 0, "suspect": False})

    def current_manifest_entry(self):
        """The manifest entry for the request being served: ask the env's
        /work (its `next` names the request the client is sending), with the
        served counter as the fallback."""
        brief = self.world.brief
        if brief is None:
            return None
        if self.world.env_url:
            status, raw = http_call("GET", self.world.env_url + "/work", timeout=5)
            if status == 200:
                try:
                    nxt = json.loads(raw.decode("utf-8")).get("next")
                except Exception:
                    nxt = None
                if nxt in brief.manifest_by_id:
                    return brief.manifest_by_id[nxt]
        manifest = brief.manifest
        return manifest[self.served_count] if self.served_count < len(manifest) else None

    def requirements(self, client_api, body, norm):
        req = {"tools": False, "vision": False, "json": False, "min_context": None}
        entry = self.current_manifest_entry()
        if entry is not None:
            r = entry.get("requires") or {}
            req["tools"] = r.get("tools") is True
            req["vision"] = r.get("vision") is True
            req["json"] = r.get("json") is True
            if isinstance(r.get("min_context"), (int, float)):
                req["min_context"] = r["min_context"]
        if norm["tools"]:
            req["tools"] = True
        if norm["json_mode"]:
            req["json"] = True
        for turn in norm["turns"]:
            for p in turn["parts"]:
                if p["kind"] in ("image", "image_url"):
                    req["vision"] = True
        return req

    def candidates(self, req, client_api, norm):
        world = self.world
        out = []
        for (pid, mid), row in world.models.items():
            if not row["free"]:
                continue
            if req["tools"] and not row["tools"]:
                continue
            if req["vision"] and not row["vision"]:
                continue
            if req["json"] and not row["json"]:
                continue
            if req["min_context"] is not None and (row["context_length"] or 0) < req["min_context"]:
                continue
            provider = world.provider_by_id.get(pid) or {}
            wire = provider.get("wire")
            if norm["json_mode"] and client_api == "openai" and wire == "anthropic":
                continue  # the anthropic wire cannot carry response_format
            out.append((pid, mid, row["rating"]))
        # Live providers first (a suspect one — it refused last time — is
        # demoted, not dropped: it stays as a last resort), best rating first.
        out.sort(key=lambda c: (
            1 if self.state_of(c[0])["suspect"] else 0,
            -c[2],
            world.provider_order.get(c[0], 99),
            c[1],
        ))
        return out

    def serve(self, client_api, body):
        norm = normalize(client_api, body)
        req = self.requirements(client_api, body, norm)
        tried = set()
        retried = False
        for pid, mid, _rating in self.candidates(req, client_api, norm):
            if pid in tried:
                continue
            tried.add(pid)
            provider = self.world.provider_by_id[pid]
            state = self.state_of(pid)
            ans = call_provider(self.world, provider, mid, client_api, body, norm)
            if ans is None and not retried and state["successes"] > 0 and state["refusals"] == 0:
                # A provider that has served all trial long and refuses for
                # the first time most likely hit its failure schedule, whose
                # next call succeeds: one immediate retry can still win the
                # best model. A provider out of budget refuses again, costs
                # one call, and is never retried twice (refusals is now 1).
                retried = True
                ans = call_provider(self.world, provider, mid, client_api, body, norm)
            if ans is not None:
                state["successes"] += 1
                state["suspect"] = False
                self.served_count += 1
                if client_api == "openai":
                    return 200, *emit_openai(ans, mid, norm["stream"])
                return 200, *emit_anthropic(ans, mid, norm["stream"])
            state["refusals"] += 1
            state["suspect"] = True
        self.served_count += 1
        return 502, *emit_error(client_api, norm["stream"], "no free provider could serve this request")


# -------------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    world = None
    router = None
    lock = threading.Lock()
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.split("?")[0] != "/v1/models":
            self._send(404, "application/json", dumps({"error": {"message": "not found", "type": "invalid_request_error", "code": None}}).encode("utf-8"))
            return
        world = self.world
        if "x-api-key" in self.headers:
            doc = {
                "data": [{"type": "model", "id": mid, "display_name": mid, "created_at": FIXED_CREATED_AT} for mid in world.free_ids],
                "has_more": False,
            }
        else:
            doc = {
                "object": "list",
                "data": [{"id": mid, "object": "model", "created": FIXED_CREATED, "owned_by": "aggregator"} for mid in world.free_ids],
            }
        self._send(200, "application/json", dumps(doc).encode("utf-8"))

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            client_api = "openai"
        elif path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send(404, "application/json", dumps({"error": {"message": "not found", "type": "invalid_request_error", "code": None}}).encode("utf-8"))
            return
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
            if not is_rec(body):
                raise ValueError("not an object")
        except Exception:
            status, ctype, payload = 400, *emit_error(client_api, False, "the request body is not one JSON object")
            self._send(status, ctype, payload)
            return
        with self.lock:
            status, ctype, payload = self.router.serve(client_api, body)
        self._send(status, ctype, payload)


def build_world():
    brief = load_brief()
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    return World(brief, env_url)


def main(argv):
    if "--list" in argv:
        world = build_world()
        sys.stdout.write(dumps(world.catalog) + "\n")
        return 0
    world = build_world()
    Handler.world = world
    Handler.router = Router(world)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    attempts = 0
    while port == 8080 and attempts < 5:
        server.server_close()
        server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
        port = server.server_address[1]
        attempts += 1
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env_url = world.env_url
    if env_url:
        status, _raw = http_call("POST", env_url + "/register", payload={
            "protocol_version": PROTOCOL_VERSION, "port": port, "catalog": world.catalog,
        }, timeout=10)
        if status == 200:
            while True:
                wstatus, wraw = http_call("GET", env_url + "/work", timeout=5)
                done = False
                if wstatus == 200:
                    try:
                        done = json.loads(wraw.decode("utf-8")).get("done") is True
                    except Exception:
                        done = False
                if done:
                    break
                time.sleep(0.1)
    else:
        thread.join()
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
