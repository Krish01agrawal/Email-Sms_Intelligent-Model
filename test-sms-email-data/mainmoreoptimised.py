#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed SMS Processing Script for LifafaV0
========================================

Processes SMS data through LLM and outputs structured JSON array format.
Fixed to handle the actual test_sms.json structure and output proper JSON.
"""

import os
import re
import json
import time
import argparse
import asyncio
from typing import Any, Dict, Optional, List
import uuid

import aiohttp
from tqdm import tqdm

API_URL = os.getenv("API_URL", "")
API_KEY = os.getenv("API_KEY", "")

# Enhanced rules for better SMS processing
UNIVERSAL_RULES = """You are an expert financial data parser for LifafaV0 — an AI financial OS that ingests SMS messages, classifies them, and extracts structured financial data.

TASK
Given a single SMS message JSON, extract ONE JSON object with transaction details. Output valid JSON only (no markdown, no comments). If a field is not confidently available, omit the field entirely.

INPUT MESSAGE JSON
{
  "channel": "sms",
  "sender": "phone/short-code/name",
  "subject": null,
  "body": "full message body",
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
    "channel": "sms",
    "sender": "string",
    "method": "IMPS | NEFT | UPI | Card | Cash | NetBanking | RTGS | MF | SIP | ATM | Wallet | Other",
    "reference_id": "txn/ref/utr/otp/folio/etc",
    "folio": "for mutual funds",
    "scheme": "for mutual funds",
    "original_text": "verbatim body"
  }
}

CLASSIFICATION RULES
- Transaction type: "credited/received/deposit/refund/loan disbursed/dividend" → credit; "debited/spent/purchased/withdrawn/payment successful/ATM cash" → debit
- For requests/pending ("has requested money", "awaiting confirmation") do NOT set transaction_type; set message_intent to "payment_request" or "pending_confirmation"
- Amount: primary monetary figure, remove commas, keep decimals
- Balance: from "Avl Bal/Available Balance/Bal:"
- Dates: use input ISO date unless body has explicit unambiguous date
- Preserve masked account formats (XXXXXXXX9855, A/cX9855, *1234)
- Counterparty: merchant/person/org (e.g., "STATION91 TECHNOLOG", "UBI ATM PBGE0110")
- Category & tags from context; 2–5 concise tags
- Confidence: 0.90–1.00 for clear transactional SMS; 0.70-0.89 for partial/pending; 0.50-0.69 for promo/info; lower for unclear
- Currency: Default to "INR" for Indian SMS
- Output ONLY one JSON object with ONLY fields you are confident about

SPECIAL CASES
- OTP/Verification: message_intent="otp", omit transaction fields
- Promotional: message_intent="promo", omit transaction fields  
- Payment Requests: message_intent="payment_request", include amount but omit transaction_type
- ATM Alerts: message_intent="alert" for informational, "transaction" for actual withdrawals
- Investment: category="investment", include folio/scheme in metadata

NOW PARSE THE INPUT MESSAGE AND RETURN ONLY THE OUTPUT JSON."""

def build_prompt(input_msg: Dict[str, Any]) -> str:
    """Build the prompt for LLM processing"""
    return UNIVERSAL_RULES + "\n\nINPUT MESSAGE JSON:\n" + json.dumps(input_msg, ensure_ascii=False)

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM response"""
    if not text:
        return None
    
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith('```') and text.endswith('```'):
        lines = text.split('\n')
        if len(lines) > 2:
            text = '\n'.join(lines[1:-1])
    
    # Remove any json language identifier
    if text.startswith('json\n'):
        text = text[5:]
    
    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Find JSON object using regex (non-greedy)
    json_match = re.search(r'\{.*?\}', text, flags=re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except Exception:
            pass
    
    # Try to find the largest valid JSON block
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    return json.loads(text[start_idx:i+1])
                except Exception:
                    continue
    
    return None

async def call_openai_style(session: aiohttp.ClientSession, model: str, prompt: str, 
                           temperature: float, max_tokens: int, top_p: float):
    """Call OpenAI-compatible API"""
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
    """Call generic endpoint"""
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
    """Extract the assistant text content and parse JSON"""
    content = None
    if not data:
        return None

    if mode == "openai":
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = None
    else:
        # Try common fields for generic endpoints
        content = (
            data.get("text")
            or data.get("output")
            or data.get("generated_text")
            or data.get("content")
        )

    return extract_json_object(content) if content else None

def safe_enrich(input_msg: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Safe enrichment of parsed data"""
    try:
        if parsed.get("message_intent") != "transaction":
            return parsed  # Only enrich actual transactions
        
        # Ensure currency is set
        if "currency" not in parsed:
            parsed["currency"] = "INR"
        
        # Set method based on content analysis
        if "metadata" not in parsed:
            parsed["metadata"] = {}
        
        if not parsed["metadata"].get("method"):
            body = input_msg.get("body", "").lower()
            method = "Other"
            if "imps" in body: method = "IMPS"
            elif "neft" in body: method = "NEFT"
            elif "rtgs" in body: method = "RTGS"
            elif "upi" in body: method = "UPI"
            elif "atm" in body: method = "ATM"
            elif "credit card" in body or "debit card" in body: method = "Card"
            elif "mutual fund" in body or "sip" in body: method = "MF"
            
            parsed["metadata"]["method"] = method
        
        # Set category if missing for transactions
        if "category" not in parsed:
            body = input_msg.get("body", "").lower()
            if "atm" in body and "withdrawn" in body:
                parsed["category"] = "atm-withdrawal"
            elif "refund" in body:
                parsed["category"] = "refund"
            elif "mutual fund" in body or "investment" in body:
                parsed["category"] = "investment"
            else:
                parsed["category"] = "transfer"
        
        return parsed
    except Exception:
        return parsed  # Never fail enrichment

async def worker(name: str, queue: asyncio.Queue, results: List[Dict[str, Any]], 
                failures: List[Dict[str, Any]], enrich_mode: str,
                model: str, mode: str, temperature: float, max_tokens: int, top_p: float):
    """Worker to process SMS messages"""
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

            # Optional safe enrichment
            if parsed and enrich_mode == "safe":
                parsed = safe_enrich(input_msg, parsed)

            if parsed:
                results.append(parsed)
            else:
                # Log failure
                raw_text = None
                if data:
                    if mode == "openai":
                        try:
                            raw_text = data["choices"][0]["message"]["content"]
                        except Exception:
                            raw_text = None
                    else:
                        raw_text = (
                            data.get("text")
                            or data.get("output")
                            or data.get("generated_text")
                            or data.get("content")
                        )
                
                failures.append({
                    "_source_id": src_id,
                    "input": input_msg,
                    "raw": raw_text
                })

            queue.task_done()

def load_sms_data(path: str) -> List[Dict[str, Any]]:
    """Load SMS data from JSON file and normalize structure"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict) and 'sms' in data:
        sms_list = data['sms']
    elif isinstance(data, list):
        sms_list = data
    else:
        raise ValueError("Invalid JSON format. Expected list or dict with 'sms' key.")
    
    # Normalize each SMS to expected format
    normalized_sms = []
    for i, sms in enumerate(sms_list):
        normalized = {
            "id": sms.get("id", str(i + 1)),
            "channel": "sms",
            "sender": sms.get("sender", ""),
            "subject": None,
            "body": sms.get("body", ""),
            "date": sms.get("date", ""),
            "type": sms.get("type", "received")
        }
        normalized_sms.append(normalized)
    
    return normalized_sms

async def process_sms_batch(input_path: str, output_path: str, model: str, mode: str, 
                           concurrency: int, temperature: float, max_tokens: int, 
                           top_p: float, failures_path: Optional[str], enrich_mode: str):
    """Process SMS batch and output JSON array"""
    
    # Load and normalize SMS data
    print(f"📱 Loading SMS data from: {input_path}")
    sms_data = load_sms_data(input_path)
    total = len(sms_data)
    print(f"📊 Loaded {total} SMS messages")
    
    # Shared results and failures lists
    results = []
    failures = []
    
    # Create async queue
    q = asyncio.Queue(maxsize=concurrency * 2)
    
    # Start workers
    tasks = [
        asyncio.create_task(worker(f"w{i}", q, results, failures, enrich_mode,
                                   model, mode, temperature, max_tokens, top_p))
        for i in range(concurrency)
    ]
    
    # Enqueue all SMS
    for sms in sms_data:
        await q.put({"id": sms["id"], "msg": sms})
    
    # Progress tracking
    pbar = tqdm(total=total, desc="Processing SMS", unit="msg")
    
    # Monitor progress
    prev_done = 0
    while not q.empty():
        done_now = total - q.qsize()
        if done_now > prev_done:
            pbar.update(done_now - prev_done)
            prev_done = done_now
        await asyncio.sleep(0.2)
    
    # Wait for all tasks to complete
    await q.join()
    pbar.update(total - prev_done)
    pbar.close()
    
    # Stop workers
    for _ in tasks:
        await q.put(None)
    await asyncio.gather(*tasks)
    
    # Save results as JSON array
    print(f"💾 Saving {len(results)} processed results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save failures if requested
    if failures_path and failures:
        print(f"⚠️  Saving {len(failures)} failures to: {failures_path}")
        with open(failures_path, "w", encoding="utf-8") as f:
            for failure in failures:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
    
    # Print summary
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"   Total SMS: {total}")
    print(f"   Successfully Processed: {len(results)} ({len(results)/total*100:.1f}%)")
    print(f"   Failed: {len(failures)} ({len(failures)/total*100:.1f}%)")
    
    if results:
        # Analyze message intents
        intent_counts = {}
        for result in results:
            intent = result.get("message_intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print(f"\n📋 Message Intent Breakdown:")
        for intent, count in sorted(intent_counts.items()):
            print(f"   {intent.title()}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Process SMS data through LLM")
    parser.add_argument("--input", required=True, help="Path to SMS JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--model", default="qwen3:8b", help="Model name")
    parser.add_argument("--mode", choices=["openai", "generic"], default="openai",
                        help="API mode: openai or generic")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--failures", help="Path to write failures (NDJSON)")
    parser.add_argument("--enrich", choices=["off", "safe"], default="safe",
                        help="Enrichment mode")

    args = parser.parse_args()

    if not API_URL:
        raise SystemExit("❌ Set API_URL environment variable to your endpoint.")

    print(f"🚀 Starting SMS Processing")
    print(f"   Endpoint: {API_URL}")
    print(f"   Model: {args.model}")
    print(f"   Mode: {args.mode}")
    print(f"   Concurrency: {args.concurrency}")
    print(f"   Enrichment: {args.enrich}")
    print(f"   Failures Log: {args.failures or 'none'}")

    asyncio.run(process_sms_batch(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        mode=args.mode,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        failures_path=args.failures,
        enrich_mode=args.enrich,
    ))

if __name__ == "__main__":
    main()
