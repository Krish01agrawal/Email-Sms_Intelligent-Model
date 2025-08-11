#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-test LifafaV0 message parsing on your AWS-hosted LLM.

Features
- Works with OpenAI-compatible chat endpoints (e.g., vLLM) OR a generic endpoint
- Async concurrency with retries + exponential backoff
- Accepts JSONL (recommended) or JSON array inputs
- Outputs NDJSON: each line has { _source_id, input, parsed, raw }
- Compact universal prompt for lower tokens; optional few-shots
- Validates JSON parse from model (drops non-JSON safely)

Usage
  export API_URL="https://<your-aws-endpoint>/v1/chat/completions"
  export API_KEY="sk-..."           # if needed; else leave empty

  python batch_test.py \
    --input sms_data.jsonl \
    --output sms_outputs.ndjson \
    --model "qwen3:8b" \
    --mode openai \
    --concurrency 8

  python batch_test.py \
    --input email_data.jsonl \
    --output email_outputs.ndjson \
    --model "qwen3:8b" \
    --mode openai \
    --concurrency 8

For a generic endpoint that expects {"prompt": "..."} and returns {"text": "..."}:
  export API_URL="https://<your-aws-endpoint>/generate"
  python batch_test.py --mode generic ...
"""

import os
import re
import json
import time
import argparse
import asyncio
from typing import Any, Dict, Optional

import aiohttp
from tqdm import tqdm

API_URL = os.getenv("API_URL", "")
API_KEY = os.getenv("API_KEY", "")

# ---- Compact universal rules (no few-shots to save tokens) ----
UNIVERSAL_RULES = """You are an expert financial data parser for LifafaV0 — an AI financial OS that ingests Gmail + SMS, classifies messages, extracts structured financial data, stores it in MongoDB, and serves real-time analytics.

TASK
Given a single message JSON (SMS or Email), extract ONE JSON object with transaction details. Output valid JSON only (no markdown, no comments). If a field is not confidently available, omit the field.

INPUT MESSAGE JSON (unified)
{
  "channel": "sms | email",
  "sender": "phone/short-code/name/email",
  "subject": "<email subject or null>",
  "body": "<full message body>",
  "date": "ISO 8601 timestamp",
  "type": "received | sent"
}

OUTPUT JSON (fields optional except currency)
{
  "transaction_type": "credit | debit",
  "amount": number,
  "currency": "INR",
  "transaction_date": "ISO 8601",
  "account": { "bank": "string", "account_number": "masked or full" },
  "counterparty": "payee/payer/merchant/origin",
  "balance": number,
  "category": "salary | food-order | grocery | online-shopping | utilities | mobile-recharge | fuel | rent | subscription | entertainment | movie | travel | hotel | dining | healthcare | pharmacy | education | transfer | refund | atm-withdrawal | fees | loan-payment | loan-disbursal | credit-card-payment | investment | dividend | tax | wallet-topup | wallet-withdrawal | insurance | other",
  "tags": ["2–5 short tags"],
  "summary": "<= 10 words",
  "confidence_score": 0.0-1.0,
  "message_intent": "transaction | payment_request | pending_confirmation | info | promo | otp | delivery | alert | other",
  "metadata": {
    "channel": "sms | email",
    "sender": "string",
    "method": "IMPS | NEFT | UPI | Card | Cash | NetBanking | RTGS | MF | SIP | ATM | Wallet | Other",
    "reference_id": "txn/ref/utr/otp/folio/etc",
    "folio": "for mutual funds",
    "scheme": "for mutual funds",
    "original_text": "verbatim body"
  }
}

RULES
- Transaction type: "credited/received/deposit/refund posted/loan disbursed/dividend" → credit; "debited/spent/purchased/withdrawn/payment successful/ATM cash" → debit.
- For requests/pending (“has requested money”, “awaiting confirmation”) do NOT set transaction_type; set message_intent accordingly.
- Amount: primary monetary figure for the event, remove commas; keep decimals.
- Balance: from “Avl Bal/Available Balance/Bal:”.
- Dates: use input ISO date unless body has an explicit unambiguous date; keep ISO if ambiguous.
- Preserve masked account formats (e.g., XXXXXXXX9855, A/cX9855, *1234).
- Counterparty: merchant/person/org (e.g., “STATION91 TECHNOLOG”, “UBI ATM PBGE0110”).
- Category & tags from context; 2–5 concise tags.
- Confidence: 0.90–1.00 for clear transactional SMS; lower for partial/promo/pending.
- Output ONLY one JSON object and ONLY fields you are confident about.

NOW PARSE THE NEXT INPUT MESSAGE AND RETURN ONLY THE OUTPUT JSON.
"""

FEWSHOTS = []  # keep empty by default to reduce tokens; you can populate if needed

def build_prompt(input_msg: Dict[str, Any]) -> str:
    parts = [UNIVERSAL_RULES, "\nINPUT MESSAGE JSON:\n", json.dumps(input_msg, ensure_ascii=False)]
    if FEWSHOTS:
        parts.append("\n\nEXAMPLES:\n")
        parts.extend(FEWSHOTS)
    return "".join(parts)

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Try to parse content as JSON; if it fails, attempt to extract the first {...} block."""
    if not text:
        return None
    text = text.strip()
    # Fast path
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find first JSON object
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

async def call_openai_style(session: aiohttp.ClientSession, model: str, prompt: str, temperature: float, max_tokens: int, top_p: float):
    payload = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    for attempt in range(6):
        try:
            async with session.post(API_URL, json=payload, headers=headers, timeout=180, ssl=False) as resp:
                if resp.status in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(60, 2 ** attempt))
                    continue
                data = await resp.json()
                return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(min(60, 2 ** attempt))
    return None

async def call_generic(session: aiohttp.ClientSession, prompt: str):
    """Generic endpoint: POST {"prompt": "..."} -> {"text": "..."} or {"output": "..."}"""
    payload = {"prompt": prompt}
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    for attempt in range(6):
        try:
            async with session.post(API_URL, json=payload, headers=headers, timeout=180, ssl=False) as resp:
                if resp.status in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(60, 2 ** attempt))
                    continue
                data = await resp.json()
                return data
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(min(60, 2 ** attempt))
    return None

def parse_response(data: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
    """Extract the assistant text content, then parse JSON."""
    content = None
    if not data:
        return None

    if mode == "openai":
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = None
    else:
        # Try common fields
        content = (
            data.get("text")
            or data.get("output")
            or data.get("generated_text")
            or data.get("content")
        )

    return extract_json_object(content) if content else None

async def worker(name: str, queue: asyncio.Queue, out_fp, model: str, mode: str, temperature: float, max_tokens: int, top_p: float):
    async with aiohttp.ClientSession() as session:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break

            src_id = item.get("id")
            input_msg = item["msg"]

            prompt = build_prompt(input_msg)

            if mode == "openai":
                data = await call_openai_style(session, model, prompt, temperature, max_tokens, top_p)
            else:
                data = await call_generic(session, prompt)

            parsed = parse_response(data, mode)
            # Only write the clean parsed JSON if it exists
            if parsed:
                out_fp.write(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n")
                out_fp.flush()  # Ensure it's written immediately
            queue.task_done()

def load_rows(path: str):
    """Yield rows from JSONL or JSON array. Each row MUST include:
       { id: "unique_id", channel, sender, subject, body, date, type }"""
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                yield row
    else:
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for row in arr:
            yield row

async def run_batch(input_path: str, output_path: str, model: str, mode: str, concurrency: int, temperature: float, max_tokens: int, top_p: float):
    q = asyncio.Queue(maxsize=concurrency * 2)

    with open(output_path, "w", encoding="utf-8") as out_fp:
        # Start workers
        tasks = [
            asyncio.create_task(worker(f"w{i}", q, out_fp, model, mode, temperature, max_tokens, top_p))
            for i in range(concurrency)
        ]

        # Enqueue
        total = 0
        for row in load_rows(input_path):
            # normalize required fields; you can customize here
            msg = {
                "channel": row.get("channel", "sms"),
                "sender": row.get("sender"),
                "subject": row.get("subject"),
                "body": row.get("body", ""),
                "date": row.get("date"),
                "type": row.get("type", "received")
            }
            await q.put({"id": row.get("id"), "msg": msg})
            total += 1

        # Progress
        pbar = tqdm(total=total, desc="Processing", unit="msg")

        # Drain queue while updating progress
        prev_done = 0
        while not q.empty():
            done_now = total - q.qsize()
            if done_now > prev_done:
                pbar.update(done_now - prev_done)
                prev_done = done_now
            await asyncio.sleep(0.2)

        await q.join()
        pbar.update(total - prev_done)
        pbar.close()

        # Stop workers
        for _ in tasks:
            await q.put(None)
        await asyncio.gather(*tasks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSONL (recommended) or JSON array")
    parser.add_argument("--output", required=True, help="NDJSON output path")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--mode", choices=["openai", "generic"], default="openai",
                        help="openai = /v1/chat/completions; generic = {'prompt': ...} -> {'text': ...}")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    if not API_URL:
        raise SystemExit("Set API_URL env var to your AWS endpoint.")

    print(f"Endpoint: {API_URL} | Mode: {args.mode} | Model: {args.model} | Concurrency: {args.concurrency}")
    asyncio.run(run_batch(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        mode=args.mode,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
    ))

if __name__ == "__main__":
    main()
