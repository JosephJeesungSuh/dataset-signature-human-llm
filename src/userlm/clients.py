"""Clients for simulated conversations between a trained user model and an assistant.

UserModelClient queries a user model served by vLLM through the raw /v1/completions endpoint with the
exact training layout (build_sft_data.py): the intent as system message, the dialogue so far, then the
opening of a user turn. Generation stops at the end-of-turn token; <|endconversation|> means the user
ends the conversation. An empty reply or a reply identical to the previous user turn is resampled up to
seven times, raising the temperature by 0.05 per retry (at most 1.0). An optional validator can reject
replies (task-simulation guardrails), which are then resampled.

AssistantClient queries any OpenAI-compatible chat endpoint; it never sees the intent.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import httpx
import openai
from openai import OpenAI

from src.userlm.chat_templates import CHAT_TEMPLATES, ENDCONV, TURN_MARKERS

_CONTEXT_ERRORS = ("maximum context length", "max_model_len", "context length", "too long", "reduce the length")


def make_client(base_url, max_connections=64, timeout=1800.0, retries=2):
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
    tmo = httpx.Timeout(timeout=timeout, connect=10.0, pool=600.0)
    return OpenAI(api_key="EMPTY", base_url=base_url, max_retries=retries, timeout=tmo,
                  http_client=httpx.Client(limits=limits, timeout=tmo))


def served_model(client):
    """The single model id served by an endpoint."""
    ids = [m.id for m in client.models.list().data]
    if len(ids) != 1:
        raise RuntimeError(f"expected one served model at {client.base_url}, found {ids}")
    return ids[0]


def strip_thinking(text):
    """Remove a reasoning trace that a server returned inside the content."""
    text = text or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return "" if "<think>" in text else text.strip()


@dataclass
class Turn:
    text: str
    terminal: bool
    reason: str          # ok | endconversation | context_limit | empty | error
    meta: Dict = field(default_factory=dict)


class UserModelClient:
    def __init__(self, client, model, tokenizer, base_model, temperature=0.7, top_p=0.9, max_new_tokens=1024,
                 max_model_len=8192, max_resamples=7, temperature_boost=0.05, max_temperature=1.0,
                 logit_bias=None, max_validator_retries=32, retries=8):
        self.client, self.model, self.tok = client, model, tokenizer
        self.template = CHAT_TEMPLATES[base_model]
        self.header, self.stop = TURN_MARKERS[base_model]["generation_prompt"], TURN_MARKERS[base_model]["end_of_turn"]
        self.endconv_id = tokenizer.convert_tokens_to_ids(ENDCONV)
        self.temperature, self.top_p, self.max_new_tokens, self.max_model_len = temperature, top_p, max_new_tokens, max_model_len
        self.max_resamples, self.boost, self.max_temperature = max_resamples, temperature_boost, max_temperature
        self.logit_bias = {str(k): v for k, v in (logit_bias or {}).items()}
        self.max_validator_retries, self.retries = max_validator_retries, retries

    def prompt(self, intent, conversation):
        messages = [{"role": "system", "content": intent.strip()}]
        messages += [{"role": t["role"], "content": t["content"].strip()} for t in conversation]
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                            chat_template=self.template)
        if self.tok.bos_token and not text.startswith(self.tok.bos_token):   # Qwen2.5 has no BOS token
            text = self.tok.bos_token + text
        return text + self.header

    def _complete(self, prompt, temperature):
        # Keep special tokens so that <|endconversation|> and the stop token are visible in the text.
        extra = {"skip_special_tokens": False, "include_stop_str_in_output": True,
                 "spaces_between_special_tokens": False, "add_special_tokens": False}
        for attempt in range(self.retries):
            try:
                kwargs = dict(model=self.model, prompt=prompt, stop=[self.stop], temperature=temperature,
                              top_p=self.top_p, max_tokens=self.max_new_tokens, extra_body=extra)
                if self.logit_bias:
                    kwargs["logit_bias"] = self.logit_bias
                return self.client.completions.create(**kwargs).choices[0].text or ""
            except openai.BadRequestError:
                raise
            except Exception:
                if attempt + 1 == self.retries:
                    raise
                time.sleep(min(1.3 ** attempt, 30.0))

    def generate(self, intent, conversation, allow_terminal=True,
                 validator: Optional[Callable[[str, List[Dict]], Optional[str]]] = None) -> Turn:
        prompt = self.prompt(intent, conversation)
        meta = {"resamples": 0}
        if len(self.tok.encode(prompt, add_special_tokens=False)) + self.max_new_tokens > self.max_model_len:
            return Turn("", True, "context_limit", meta)
        previous = next((t["content"].strip() for t in reversed(conversation) if t["role"] == "user"), None)
        resample = 0
        while True:
            temperature = min(self.temperature + self.boost * resample, self.max_temperature)
            try:
                reply = self._complete(prompt, temperature).strip().removesuffix(self.stop).strip()
            except openai.BadRequestError as e:
                meta["error"] = str(e)[:300]
                return Turn("", True, "context_limit" if any(m in str(e).lower() for m in _CONTEXT_ERRORS)
                            else "error", meta)
            except Exception as e:
                meta["error"] = str(e)[:300]
                return Turn("", True, "error", meta)
            meta.update(resamples=resample, temperature=temperature)
            if ENDCONV in reply:
                if allow_terminal or resample >= self.max_resamples:
                    return Turn(reply, True, "endconversation", meta)
                resample += 1          # the first turn must start the conversation
                continue
            if not reply or reply == previous:
                if resample < self.max_resamples:
                    resample += 1
                    continue
                if not reply:
                    return Turn("", True, "empty", meta)
            why = validator(reply, conversation) if validator else None
            if why:
                meta.setdefault("guardrail_hits", {}).setdefault(why, 0)
                meta["guardrail_hits"][why] += 1
                if resample < self.max_validator_retries:
                    resample += 1
                    continue
                meta["guardrail_exhausted"] = why
            return Turn(reply, False, "ok", meta)


class AssistantClient:
    """Chat-completions assistant; requests are spread over endpoints by outstanding count."""

    def __init__(self, clients, model, temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048,
                 enable_thinking=False, retries=8):
        self.clients, self.model = list(clients), model
        self.inflight, self.lock = [0] * len(self.clients), threading.Lock()
        self.temperature, self.top_p, self.top_k, self.max_tokens = temperature, top_p, top_k, max_tokens
        self.enable_thinking, self.retries = enable_thinking, retries

    def generate(self, conversation) -> Turn:
        extra = {"top_k": self.top_k} if self.top_k is not None else {}
        if "qwen3" in self.model.lower():     # hybrid-thinking models: thinking on/off switch
            extra["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        messages = [{"role": t["role"], "content": t["content"]} for t in conversation]
        for attempt in range(self.retries):
            with self.lock:
                i = min(range(len(self.clients)), key=self.inflight.__getitem__)
                self.inflight[i] += 1
            try:
                r = self.clients[i].chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature, top_p=self.top_p,
                    max_tokens=self.max_tokens, extra_body=extra)
                text = strip_thinking(r.choices[0].message.content)
                if text:
                    return Turn(text, False, "ok", {"finish_reason": r.choices[0].finish_reason})
            except openai.BadRequestError as e:
                reason = "context_limit" if any(m in str(e).lower() for m in _CONTEXT_ERRORS) else "error"
                return Turn("", True, reason, {"error": str(e)[:300]})
            except Exception as e:
                if attempt + 1 == self.retries:
                    return Turn("", True, "error", {"error": str(e)[:300]})
                time.sleep(min(1.3 ** attempt, 30.0))
            finally:
                with self.lock:
                    self.inflight[i] -= 1
        return Turn("", True, "empty", {})
