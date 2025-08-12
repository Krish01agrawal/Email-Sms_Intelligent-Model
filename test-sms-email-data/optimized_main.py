#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized SMS Processing Script for LifafaV0
============================================

Processes SMS data through LLM with efficient batch processing:
- 5 parallel batches of 2 SMS each (configurable)
- Real-time progress tracking
- Server-friendly processing to avoid overload
"""

import os
import re
import json
import time
import argparse
import asyncio
from typing import Any, Dict, Optional, List
import math

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
    """Call OpenAI-compatible API with retry logic"""
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

    for attempt in range(3):  # Reduced retries for faster processing
        try:
            async with session.post(API_URL, json=payload, headers=headers, timeout=30, ssl=False) as resp:
                if resp.status in (429, 500, 502, 503, 504):
                    await asyncio.sleep(min(10, 2 ** attempt))
                    continue
                data = await resp.json()
                return data
        except Exception as e:
            if attempt == 2:  # Last attempt
                print(f"⚠️  API call failed after 3 attempts: {e}")
            await asyncio.sleep(min(5, 2 ** attempt))
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

async def process_sms_batch(sms_batch: List[Dict[str, Any]], batch_id: int, 
                           session: aiohttp.ClientSession, model: str, mode: str,
                           temperature: float, max_tokens: int, top_p: float, 
                           enrich_mode: str, pbar: tqdm) -> tuple:
    """Process a batch of SMS messages"""
    results = []
    failures = []
    
    print(f"🔄 Processing Batch {batch_id} ({len(sms_batch)} SMS)")
    
    # Process each SMS in the batch
    for sms_data in sms_batch:
        src_id = sms_data.get("id")
        input_msg = sms_data
        
        prompt = build_prompt(input_msg)
        
        if mode == "openai":
            data = await call_openai_style(session, model, prompt, temperature, max_tokens, top_p)
        else:
            # Generic mode not implemented in this version
            data = None
        
        parsed = parse_response(data, mode)
        
        # Optional safe enrichment
        if parsed and enrich_mode == "safe":
            parsed = safe_enrich(input_msg, parsed)
        
        if parsed:
            results.append(parsed)
            print(f"  ✅ SMS {src_id}: {parsed.get('message_intent', 'unknown')}")
        else:
            # Log failure
            raw_text = None
            if data and mode == "openai":
                try:
                    raw_text = data["choices"][0]["message"]["content"]
                except Exception:
                    raw_text = None
            
            failures.append({
                "_source_id": src_id,
                "batch_id": batch_id,
                "input": input_msg,
                "raw": raw_text
            })
            print(f"  ❌ SMS {src_id}: Processing failed")
        
        # Update progress bar
        pbar.update(1)
        
        # Small delay to avoid overwhelming the server
        await asyncio.sleep(0.1)
    
    print(f"✅ Batch {batch_id} completed: {len(results)} success, {len(failures)} failed")
    return results, failures

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

def create_batches(sms_list: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    """Create batches from SMS list"""
    batches = []
    for i in range(0, len(sms_list), batch_size):
        batch = sms_list[i:i + batch_size]
        batches.append(batch)
    return batches

async def process_all_batches(input_path: str, output_path: str, model: str, mode: str,
                             batch_size: int, max_parallel_batches: int,
                             temperature: float, max_tokens: int, top_p: float, 
                             failures_path: Optional[str], enrich_mode: str):
    """Process all SMS in optimized batches"""
    
    # Load and normalize SMS data
    print(f"📱 Loading SMS data from: {input_path}")
    sms_data = load_sms_data(input_path)
    total_sms = len(sms_data)
    print(f"📊 Loaded {total_sms} SMS messages")
    
    # Create batches
    batches = create_batches(sms_data, batch_size)
    total_batches = len(batches)
    print(f"📦 Created {total_batches} batches of {batch_size} SMS each")
    print(f"🔄 Processing {max_parallel_batches} batches in parallel")
    
    # Initialize results
    all_results = []
    all_failures = []
    
    # Create progress bar
    pbar = tqdm(total=total_sms, desc="Processing SMS", unit="msg", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Process batches in parallel groups
    async with aiohttp.ClientSession() as session:
        for i in range(0, total_batches, max_parallel_batches):
            # Get the next group of batches to process in parallel
            batch_group = batches[i:i + max_parallel_batches]
            
            print(f"\n🚀 Starting parallel processing of batches {i+1}-{min(i+len(batch_group), total_batches)}")
            
            # Create tasks for parallel processing
            tasks = []
            for j, batch in enumerate(batch_group):
                batch_id = i + j + 1
                task = process_sms_batch(
                    batch, batch_id, session, model, mode,
                    temperature, max_tokens, top_p, enrich_mode, pbar
                )
                tasks.append(task)
            
            # Wait for all batches in this group to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"❌ Batch failed with error: {result}")
                    continue
                
                results, failures = result
                all_results.extend(results)
                all_failures.extend(failures)
            
            # Show progress summary
            print(f"📊 Completed {min(i+max_parallel_batches, total_batches)}/{total_batches} batch groups")
            print(f"   Success: {len(all_results)}, Failures: {len(all_failures)}")
            
            # Brief pause between batch groups to avoid overwhelming server
            if i + max_parallel_batches < total_batches:
                print("⏸️  Brief pause before next batch group...")
                await asyncio.sleep(1)
    
    pbar.close()
    
    # Save results as JSON array
    print(f"\n💾 Saving {len(all_results)} processed results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Save failures if requested
    if failures_path and all_failures:
        print(f"⚠️  Saving {len(all_failures)} failures to: {failures_path}")
        with open(failures_path, "w", encoding="utf-8") as f:
            for failure in all_failures:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
    
    # Print final summary
    print(f"\n📊 FINAL PROCESSING SUMMARY:")
    print(f"   Total SMS: {total_sms}")
    print(f"   Successfully Processed: {len(all_results)} ({len(all_results)/total_sms*100:.1f}%)")
    print(f"   Failed: {len(all_failures)} ({len(all_failures)/total_sms*100:.1f}%)")
    print(f"   Processing Method: {max_parallel_batches} parallel batches of {batch_size} SMS each")
    
    if all_results:
        # Analyze message intents
        intent_counts = {}
        for result in all_results:
            intent = result.get("message_intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print(f"\n📋 Message Intent Breakdown:")
        for intent, count in sorted(intent_counts.items()):
            print(f"   {intent.title()}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Process SMS data through LLM with optimized batching")
    parser.add_argument("--input", required=True, help="Path to SMS JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--model", default="qwen3:8b", help="Model name")
    parser.add_argument("--mode", choices=["openai", "generic"], default="openai",
                        help="API mode: openai or generic")
    parser.add_argument("--batch-size", type=int, default=2, help="SMS per batch (default: 2)")
    parser.add_argument("--parallel-batches", type=int, default=5, help="Parallel batches (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--failures", help="Path to write failures (NDJSON)")
    parser.add_argument("--enrich", choices=["off", "safe"], default="safe",
                        help="Enrichment mode")

    args = parser.parse_args()

    if not API_URL:
        raise SystemExit("❌ Set API_URL environment variable to your endpoint.")

    print(f"🚀 Starting Optimized SMS Processing")
    print(f"   Endpoint: {API_URL}")
    print(f"   Model: {args.model}")
    print(f"   Mode: {args.mode}")
    print(f"   Batch Size: {args.batch_size} SMS per batch")
    print(f"   Parallel Batches: {args.parallel_batches}")
    print(f"   Enrichment: {args.enrich}")
    print(f"   Failures Log: {args.failures or 'none'}")

    asyncio.run(process_all_batches(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        mode=args.mode,
        batch_size=args.batch_size,
        max_parallel_batches=args.parallel_batches,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        failures_path=args.failures,
        enrich_mode=args.enrich,
    ))

if __name__ == "__main__":
    main()
