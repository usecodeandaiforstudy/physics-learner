# -*- coding: utf-8 -*-
"""模式③共享 LLM 客户端（OpenAI 兼容 chat completions，仅标准库，零新依赖）。

兼容 OpenAI / DeepSeek / Moonshot / Qwen 等 OpenAI 格式端点。
配置放 db1/mode3_config.json（模板 mode3_config.example.json）：
  base_url / api_key_env / model / timeout_s / temperature
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

__all__ = ["load_config", "chat", "extract_json"]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def chat(cfg: dict, messages: list[dict], temperature: float | None = None,
         timeout: float | None = None) -> str:
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    key = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))
    if not key:
        raise RuntimeError(
            "缺少 API key：请设置环境变量 %s（config 的 api_key_env）"
            % cfg.get("api_key_env", "OPENAI_API_KEY"))
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": (temperature if temperature is not None
                        else cfg.get("temperature", 0.3)),
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    try:
        with urllib.request.urlopen(
                req, timeout=timeout or cfg.get("timeout_s", 120)) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError("LLM API HTTP %s: %s" % (e.code, body)) from e
    return data["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """从模型输出提取第一个 {...}（剥 markdown 围栏）。"""
    t = text.strip()
    if t.startswith("```"):
        first_nl = t.find("\n")
        if first_nl >= 0:
            t = t[first_nl + 1:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("模型输出中未找到 JSON 对象")
    return json.loads(t[i:j + 1])
