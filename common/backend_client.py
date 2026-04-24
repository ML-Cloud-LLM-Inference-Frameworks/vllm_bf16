import time
from typing import Dict

from openai import OpenAI

from common.config import (
    DEFAULT_BASE_URL,
    MODEL_ID,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    PROMPT_PATH,
)
from common.parser import parse_label

PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(article: str) -> str:
    return PROMPT_TEMPLATE.format(article=article)


def classify_article(
    article: str,
    base_url: str = DEFAULT_BASE_URL,
    model_id: str = MODEL_ID,
    api_key: str = "dummy",
) -> Dict:
    client = OpenAI(base_url=base_url, api_key=api_key)

    prompt = build_prompt(article)

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
    latency_s = time.perf_counter() - t0

    raw_output = resp.choices[0].message.content or ""
    pred_label = parse_label(raw_output)

    return {
        "prediction": pred_label,
        "raw_output": raw_output,
        "latency_s": latency_s,
    }