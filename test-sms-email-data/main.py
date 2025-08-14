#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed Optimized SMS Processing Script for LifafaV0
=================================================

FIXES:
1. Improved JSON extraction to handle LLM reasoning text
2. Better error handling for API timeouts
3. Enhanced prompt clarity for structured responses
4. Robust response parsing with multiple fallback strategies
"""

import os
import re
import json
import time
import argparse
import asyncio
from typing import Any, Dict, Optional, List
import math
from datetime import datetime

import aiohttp
from tqdm import tqdm

API_URL = os.getenv("API_URL", "")
API_KEY = os.getenv("API_KEY", "")

# IMPROVED PROMPT - More explicit about JSON structure
UNIVERSAL_RULES = """You are an expert financial data parser for LifafaV0. 

CRITICAL INSTRUCTIONS:
1. Output ONLY valid JSON - no explanations, no markdown, no thinking process

MANDATORY FIELDS (must include):
- currency: "INR" (always required)
- message_intent: one of the specified values (always required)
- transaction_type: "credit" or "debit" (for actual transactions)
- amount: numeric value (for transactions)
- account: bank and account number details
2. Use the EXACT field names and structure shown below
3. If unsure about a field, omit it entirely rather than guess

INPUT: SMS message JSON
OUTPUT: Single JSON object with transaction details

REQUIRED OUTPUT STRUCTURE:
{
  "transaction_type": "credit | debit",
  "amount": number,
  "currency": "INR", 
  "transaction_date": "ISO 8601 timestamp",
  "account": {
    "bank": "bank_name",
    "account_number": "account_number"
  },
  "counterparty": "merchant/person/organization",
  "balance": number,
  "category": "investment | transfer | atm-withdrawal | other",
  "tags": ["tag1", "tag2"],
  "summary": "brief description under 10 words",
  "confidence_score": 0.0-1.0,
  "message_intent": "transaction | payment_request | pending_confirmation | otp | promo | alert | other",
  "metadata": {
    "channel": "sms",
    "sender": "sender_name",
    "method": "UPI | IMPS | NEFT | RTGS | ATM | Card | MF | Other",
    "reference_id": "transaction_reference",
    "original_text": "complete_sms_body"
  }
}

CLASSIFICATION RULES:
- "credited/received/deposit" → transaction_type: "credit"
- "debited/withdrawn/paid" → transaction_type: "debit" 
- Payment requests → message_intent: "payment_request", omit transaction_type
- OTP messages → message_intent: "otp", omit transaction fields
- Promotional → message_intent: "promo", omit transaction fields
- Currency: Always "INR" for Indian SMS
- Extract amounts as numbers (remove Rs., commas)
- Preserve masked account formats (XXXXXXXX9855, A/cX9855)

EXAMPLES:
Credit SMS: "Your a/c XXXXXXXX9855 is credited by Rs.60000.00 on 02-07-25 by STATION91"
→ {"transaction_type": "credit", "amount": 60000.0, "currency": "INR", ...}

Debit SMS: "Rs.2000 withdrawn at ATM from A/cX9855"  
→ {"transaction_type": "debit", "amount": 2000, "currency": "INR", ...}

RESPOND WITH ONLY THE JSON OBJECT - NO OTHER TEXT."""

def build_prompt(input_msg: Dict[str, Any]) -> str:
    """Build the prompt for LLM processing"""
    return UNIVERSAL_RULES + f"\n\nSMS TO PARSE:\n{json.dumps(input_msg, ensure_ascii=False)}\n\nJSON OUTPUT:"

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Enhanced JSON extraction with multiple fallback strategies"""
    if not text:
        return None
    
    text = text.strip()
    
    # Remove common LLM prefixes/suffixes
    text = re.sub(r'^```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```$', '', text)
    text = re.sub(r'^json\s*', '', text, flags=re.IGNORECASE)
    
    # Remove thinking tags and content
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove explanatory text before JSON
    text = re.sub(r'^[^{]*?(?=\{)', '', text)
    
    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Strategy 1: Find complete JSON object with proper brace matching
    brace_count = 0
    start_idx = -1
    best_json = None
    
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                try:
                    candidate = text[start_idx:i+1]
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict) and len(parsed) > 1:  # Must have multiple fields
                        best_json = parsed
                        break
                except Exception:
                    continue
    
    if best_json:
        return best_json
    
    # Strategy 2: Find JSON with regex and fix common issues
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
        r'\{.*?\}',  # Simple objects
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Fix common JSON issues
                fixed_match = match.strip()
                # Fix trailing commas
                fixed_match = re.sub(r',(\s*[}\]])', r'\1', fixed_match)
                # Fix unquoted keys
                fixed_match = re.sub(r'(\w+):', r'"\1":', fixed_match)
                
                parsed = json.loads(fixed_match)
                if isinstance(parsed, dict) and len(parsed) > 1:
                    return parsed
            except Exception:
                continue
    
    # Strategy 3: Extract key-value pairs and construct JSON
    try:
        # Look for key-value patterns
        kv_pattern = r'"([^"]+)":\s*([^,}]+)'
        matches = re.findall(kv_pattern, text)
        
        if matches:
            result = {}
            for key, value in matches:
                try:
                    # Try to parse value as JSON
                    if value.strip().startswith('"') and value.strip().endswith('"'):
                        result[key] = json.loads(value.strip())
                    elif value.strip() in ['true', 'false']:
                        result[key] = json.loads(value.strip())
                    elif re.match(r'^-?\d+\.?\d*$', value.strip()):
                        result[key] = float(value.strip()) if '.' in value else int(value.strip())
                    else:
                        result[key] = value.strip().strip('"')
                except:
                    result[key] = value.strip().strip('"')
            
            if len(result) > 1:
                return result
    except Exception:
        pass
    
    return None

async def call_openai_style(session: aiohttp.ClientSession, model: str, prompt: str, 
                           temperature: float, max_tokens: int, top_p: float):
    """Enhanced API call with better error handling"""
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

    for attempt in range(3):
        try:
            # Much longer timeout for better reliability
            timeout = aiohttp.ClientTimeout(total=120, connect=20)
            async with session.post(API_URL, json=payload, headers=headers, timeout=timeout, ssl=False) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data
                elif resp.status in (429, 500, 502, 503, 504):
                    wait_time = min(30, 5 ** attempt)  # Longer exponential backoff
                    print(f"  ⏳ API rate limit/error {resp.status}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"  ❌ API error {resp.status}: {await resp.text()}")
                    return None
        except asyncio.TimeoutError:
            print(f"  ⏳ API timeout on attempt {attempt + 1}")
            await asyncio.sleep(min(10, 2 ** attempt))
        except Exception as e:
            print(f"  ❌ API error on attempt {attempt + 1}: {str(e)[:100]}")
            await asyncio.sleep(min(5, 2 ** attempt))
    
    return None

def parse_response(data: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
    """Enhanced response parsing with better error handling"""
    content = None
    if not data:
        return None

    try:
        if mode == "openai":
            content = data["choices"][0]["message"]["content"]
        else:
            content = (
                data.get("text") or 
                data.get("output") or 
                data.get("generated_text") or 
                data.get("content")
            )
        
        if content:
            # Clean up the content before extraction
            content = content.strip()
            return extract_json_object(content)
    except Exception as e:
        print(f"  ❌ Response parsing error: {e}")
    
    return None

def safe_enrich(input_msg: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced enrichment with validation"""
    try:
        # Ensure currency is always set
        if "currency" not in parsed:
            parsed["currency"] = "INR"
        
        # Only enrich if we have meaningful data
        if not parsed.get("message_intent"):
            return parsed
        
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
        
        # Ensure metadata has required fields
        if "channel" not in parsed["metadata"]:
            parsed["metadata"]["channel"] = "sms"
        if "sender" not in parsed["metadata"]:
            parsed["metadata"]["sender"] = input_msg.get("sender", "")
        if "original_text" not in parsed["metadata"]:
            parsed["metadata"]["original_text"] = input_msg.get("body", "")
        
        return parsed
    except Exception as e:
        print(f"  ⚠️  Enrichment error: {e}")
        return parsed

def mark_sms_as_processed(input_path: str, source_id: str, success: bool = True):
    """Mark SMS as processed in the input file for resume capability"""
    try:
        # Read the input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Find and update the SMS
        if isinstance(data, dict) and 'financial_sms' in data:
            sms_list = data['financial_sms']
        elif isinstance(data, list):
            sms_list = data
        else:
            return
        
        # Find SMS by unique_id - this is the ONLY reliable way
        sms_found = False
        
        for sms in sms_list:
            if sms.get('unique_id') == source_id:
                sms['isprocessed'] = True
                sms['processing_timestamp'] = datetime.now().isoformat()
                sms['processing_status'] = 'success' if success else 'failed'
                print(f"  🔄 Marked SMS {source_id} as processed in input file")
                sms_found = True
                break
        
        if not sms_found:
            print(f"  ❌ ERROR: Could not find SMS {source_id} by unique_id in input file")
            print(f"  🔍 Available unique_ids: {[sms.get('unique_id', 'NO_ID') for sms in sms_list[:3]]}...")
        
        # Write back to file
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"⚠️  Warning: Could not update input file for SMS {source_id}: {e}")

def update_input_file_progress(input_path: str, processed_sms: List[Dict[str, Any]], failed_sms: List[Dict[str, Any]]):
    """Update input file with processing progress for all SMS in current batch"""
    try:
        # Read the input file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Find and update the SMS
        if isinstance(data, dict) and 'financial_sms' in data:
            sms_list = data['financial_sms']
        elif isinstance(data, list):
            sms_list = data
        else:
            return
        
        # Mark successful SMS as processed
        for sms in processed_sms:
            source_id = sms.get('_source_id')
            if source_id:
                # Find by unique_id - this is the ONLY reliable way
                sms_found = False
                for input_sms in sms_list:
                    if input_sms.get('unique_id') == source_id:
                        input_sms['isprocessed'] = True
                        input_sms['processing_timestamp'] = datetime.now().isoformat()
                        input_sms['processing_status'] = 'success'
                        sms_found = True
                        break
                
                if not sms_found:
                    print(f"  ❌ ERROR: Could not find SMS {source_id} by unique_id in batch progress update")
        
        # Mark failed SMS as processed (so they won't be retried)
        for sms in failed_sms:
            source_id = sms.get('_source_id')
            if source_id:
                # Find by unique_id - this is the ONLY reliable way
                sms_found = False
                for input_sms in sms_list:
                    if input_sms.get('unique_id') == source_id:
                        input_sms['isprocessed'] = True
                        input_sms['processing_timestamp'] = datetime.now().isoformat()
                        input_sms['processing_status'] = 'failed'
                        sms_found = True
                        break
                
                if not sms_found:
                    print(f"  ❌ ERROR: Could not find SMS {source_id} by unique_id in batch progress update")
        
        # Write back to file
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"⚠️  Warning: Could not update input file progress: {e}")

def write_results_real_time(output_path: str, results: List[Dict[str, Any]], mode: str = "append"):
    """Write results to output file in real-time"""
    try:
        if mode == "append" and os.path.exists(output_path):
            # Read existing results and append new ones
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            existing_results.extend(results)
            results_to_write = existing_results
            print(f"  💾 Real-time: Appended {len(results)} new results to existing {len(existing_results)} results")
        else:
            # Write all results (overwrite mode)
            results_to_write = results
            print(f"  💾 Real-time: Wrote {len(results)} results (overwrite mode)")
        
        # Write results with pretty formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_to_write, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"  ❌ Error writing results to {output_path}: {e}")

def write_failures_real_time(failures_path: str, failures: List[Dict[str, Any]], mode: str = "append"):
    """Write failures to failures file in real-time"""
    try:
        if mode == "append" and os.path.exists(failures_path):
            # Read existing failures and append new ones
            existing_failures = []
            try:
                with open(failures_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            existing_failures.append(json.loads(line.strip()))
            except:
                pass  # File might be empty or corrupted
            
            existing_failures.extend(failures)
            failures_to_write = existing_failures
        else:
            # Write all failures (overwrite mode)
            failures_to_write = failures
        
        # Write failures in NDJSON format
        with open(failures_path, 'w', encoding='utf-8') as f:
            for failure in failures_to_write:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
        
        print(f"  💾 Real-time: Updated {failures_path} with {len(failures)} new failures")
        
    except Exception as e:
        print(f"  ❌ Error writing failures to {failures_path}: {e}")

def cleanup_empty_failures_file(failures_path: str):
    """Delete failures file if it's empty (all SMS processed successfully)"""
    try:
        if os.path.exists(failures_path):
            with open(failures_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                os.remove(failures_path)
                print(f"  🗑️  Deleted empty failures file: {failures_path}")
                print(f"  ✅ All SMS processed successfully!")
            else:
                print(f"  📊 Failures file contains {len(content.split(chr(10)))} failure records")
                
    except Exception as e:
        print(f"  ⚠️  Could not check/cleanup failures file: {e}")

async def process_sms_batch(sms_batch: List[Dict[str, Any]], batch_id: int, 
                           session: aiohttp.ClientSession, model: str, mode: str,
                           temperature: float, max_tokens: int, top_p: float, 
                           enrich_mode: str, pbar: tqdm, input_path: str) -> tuple:
    """Enhanced batch processing with better error handling"""
    results = []
    failures = []
    
    print(f"🔄 Processing Batch {batch_id} ({len(sms_batch)} SMS)")
    
    for sms_data in sms_batch:
        src_id = sms_data.get("_source_id")  # Use _source_id (unique_id) instead of id
        input_msg = sms_data
        
        try:
            prompt = build_prompt(input_msg)
            
            if mode == "openai":
                data = await call_openai_style(session, model, prompt, temperature, max_tokens, top_p)
            else:
                data = None
            
            parsed = parse_response(data, mode)
            
            # Enhanced validation
            if parsed and isinstance(parsed, dict) and len(parsed) > 1:
                # Validate that we have essential fields
                essential_fields = ['currency', 'message_intent']
                has_essential = all(field in parsed for field in essential_fields)
                
                if has_essential:
                    # Safe enrichment
                    if enrich_mode == "safe":
                        parsed = safe_enrich(input_msg, parsed)
                    
                    results.append(parsed)
                    intent = parsed.get('message_intent', 'unknown')
                    amount = parsed.get('amount', 'N/A')
                    print(f"  ✅ SMS {src_id}: {intent} (₹{amount})")
                    
                    # Real-time persistence: Mark as processed in input file
                    mark_sms_as_processed(input_path, src_id, success=True)
                else:
                    # Missing essential fields - treat as failure
                    print(f"  ⚠️  SMS {src_id}: Missing essential fields")
                    failure_info = {
                        "_source_id": src_id,
                        "batch_id": batch_id,
                        "input": input_msg,
                        "parsing_error": "Missing essential fields (currency, message_intent)",
                        "partial_result": parsed
                    }
                    failures.append(failure_info)
                    
                    # Real-time persistence: Mark as processed in input file
                    mark_sms_as_processed(input_path, src_id, success=False)
            else:
                # Enhanced failure logging
                raw_text = None
                if data and mode == "openai":
                    try:
                        raw_text = data["choices"][0]["message"]["content"]
                    except:
                        raw_text = str(data)
                
                failure_info = {
                    "_source_id": src_id,
                    "batch_id": batch_id,
                    "input": input_msg,
                    "raw_response": raw_text[:500] if raw_text else None,  # Truncate long responses
                    "parsing_error": "Failed to extract valid JSON" if raw_text else "No API response"
                }
                failures.append(failure_info)
                print(f"  ❌ SMS {src_id}: Processing failed")
                
                # Real-time persistence: Mark as processed in input file
                mark_sms_as_processed(input_path, src_id, success=False)
            
        except Exception as e:
            print(f"  ❌ SMS {src_id}: Exception - {str(e)[:50]}")
            failures.append({
                "_source_id": src_id,
                "batch_id": batch_id,
                "input": input_msg,
                "error": str(e)
            })
        
        # Update progress
        pbar.update(1)
        
        # Much longer delay to avoid overwhelming server
        await asyncio.sleep(3.0)
    
    success_count = len(results)
    failure_count = len(failures)
    print(f"✅ Batch {batch_id} completed: {success_count} success, {failure_count} failed")
    
    return results, failures

def load_sms_data(path: str) -> List[Dict[str, Any]]:
    """Load SMS data from JSON file with resume capability"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        if 'financial_sms' in data:
            sms_list = data['financial_sms']
        elif 'sms' in data:
            sms_list = data['sms']
        else:
            sms_list = list(data.values())
    elif isinstance(data, list):
        sms_list = data
    else:
        raise ValueError("Invalid JSON format. Expected list or dict with 'sms' key.")
    
    # Filter only unprocessed SMS for resume capability
    unprocessed_sms = [sms for sms in sms_list if not sms.get('isprocessed', False)]
    
    if len(unprocessed_sms) < len(sms_list):
        print(f"🔄 Resume mode: {len(sms_list) - len(unprocessed_sms)} SMS already processed, continuing with {len(unprocessed_sms)} unprocessed SMS")
    
    # Normalize each SMS to expected format while preserving original structure
    normalized_sms = []
    for i, sms in enumerate(unprocessed_sms):
        # Create a copy of the original SMS to preserve all fields
        normalized = sms.copy()
        
        # Add required fields if missing
        if "id" not in normalized:
            normalized["id"] = str(i + 1)
        if "channel" not in normalized:
            normalized["channel"] = "sms"
        if "subject" not in normalized:
            normalized["subject"] = None
        if "type" not in normalized:
            normalized["type"] = "received"
        
        # Ensure _source_id is set for persistence (ALWAYS use unique_id if available)
        if "unique_id" in normalized:
            normalized["_source_id"] = normalized["unique_id"]
        elif "_source_id" not in normalized:
            normalized["_source_id"] = normalized.get("id", str(i + 1))
        
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
    """Enhanced batch processing with better progress tracking"""
    
    print(f"📱 Loading SMS data from: {input_path}")
    sms_data = load_sms_data(input_path)
    total_sms = len(sms_data)
    print(f"📊 Loaded {total_sms} SMS messages")
    
    # Initialize output files for real-time updates
    if failures_path:
        # Create empty failures file
        with open(failures_path, 'w', encoding='utf-8') as f:
            pass  # Create empty file
        print(f"📝 Initialized failures file: {failures_path}")
    
    # Initialize results file - PRESERVE existing results if file exists
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            print(f"📝 Preserved existing results file: {output_path} with {len(existing_results)} previous results")
        except:
            # If file is corrupted, start fresh
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print(f"📝 Started fresh results file: {output_path}")
    else:
        # Create new results file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        print(f"📝 Created new results file: {output_path}")
    
    batches = create_batches(sms_data, batch_size)
    total_batches = len(batches)
    print(f"📦 Created {total_batches} batches of {batch_size} SMS each")
    print(f"🔄 Processing {max_parallel_batches} batches in parallel")
    
    all_results = []
    all_failures = []
    
    # Enhanced progress bar
    pbar = tqdm(
        total=total_sms, 
        desc="Processing SMS", 
        unit="msg",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Success: {postfix}'
    )
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, total_batches, max_parallel_batches):
            batch_group = batches[i:i + max_parallel_batches]
            
            print(f"\n🚀 Processing batch group {i//max_parallel_batches + 1}/{math.ceil(total_batches/max_parallel_batches)}")
            
            # Process batches in parallel
            tasks = []
            for j, batch in enumerate(batch_group):
                batch_id = i + j + 1
                task = process_sms_batch(
                    batch, batch_id, session, model, mode,
                    temperature, max_tokens, top_p, enrich_mode, pbar, input_path
                )
                tasks.append(task)
            
            # Wait for batch group completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            batch_results_collected = []
            batch_failures_collected = []
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"❌ Batch failed: {result}")
                    continue
                
                results, failures = result
                batch_results_collected.extend(results)
                batch_failures_collected.extend(failures)
                all_results.extend(results)
                all_failures.extend(failures)
            
            # Real-time persistence: Update input file with batch progress
            if batch_results_collected or batch_failures_collected:
                update_input_file_progress(input_path, batch_results_collected, batch_failures_collected)
            
            # Real-time file updates: Write results and failures immediately
            if batch_results_collected:
                write_results_real_time(output_path, batch_results_collected, mode="append")
            if batch_failures_collected:
                write_failures_real_time(failures_path, batch_failures_collected, mode="append")
            
            # Update progress bar postfix
            pbar.set_postfix_str(f"{len(all_results)}/{total_sms}")
            
            # Longer pause between batch groups
            if i + max_parallel_batches < total_batches:
                await asyncio.sleep(5)
    
    pbar.close()
    
    # Final cleanup and summary
    print(f"\n💾 Final results file: {output_path}")
    print(f"📊 Total results written: {len(all_results)}")
    
    # Cleanup failures file if empty
    if failures_path:
        cleanup_empty_failures_file(failures_path)
    
    # Final summary
    success_rate = (len(all_results) / total_sms) * 100
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"   Total SMS: {total_sms}")
    print(f"   Successfully Processed: {len(all_results)} ({success_rate:.1f}%)")
    print(f"   Failed: {len(all_failures)} ({100-success_rate:.1f}%)")
    
    if all_results:
        # Intent breakdown
        intent_counts = {}
        for result in all_results:
            intent = result.get("message_intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print(f"\n📋 Message Intent Breakdown:")
        for intent, count in sorted(intent_counts.items()):
            print(f"   {intent.title()}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Fixed Optimized SMS Processing")
    parser.add_argument("--input", required=True, help="SMS JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--model", default="qwen3:8b", help="Model name")
    parser.add_argument("--mode", choices=["openai", "generic"], default="openai")
    parser.add_argument("--batch-size", type=int, default=1, help="SMS per batch")
    parser.add_argument("--parallel-batches", type=int, default=2, help="Parallel batches (reduced for stability)")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens (optimized for complete responses)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--failures", help="Failures log path (NDJSON)")
    parser.add_argument("--enrich", choices=["off", "safe"], default="safe")

    args = parser.parse_args()

    if not API_URL:
        raise SystemExit("❌ Set API_URL environment variable")

    print(f"🚀 Starting FIXED Optimized SMS Processing")
    print(f"   Endpoint: {API_URL}")
    print(f"   Model: {args.model}")
    print(f"   Batch Config: {args.parallel_batches} parallel × {args.batch_size} SMS")
    print(f"   Max Tokens: {args.max_tokens} (optimized)")

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
