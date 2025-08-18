#!/usr/bin/env python3
"""
MongoDB Pipeline for LifafaV0
=============================

Complete pipeline from MongoDB SMS data to processed financial transactions:
1. Read SMS from sms_data collection
2. Filter financial SMS using sms_financial_filter.py
3. Extract financial array using extract_financial_array.py
4. Process through LLM using main.py logic
5. Store results in financial_transactions collection
6. Update SMS status in sms_data collection
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from bson import ObjectId
from mongodb_operations import MongoDBOperations
from sms_financial_filter import SMSFinancialFilter
from extract_financial_array import extract_financial_array
from main import process_all_batches
import asyncio
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle MongoDB ObjectId and other non-serializable types"""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def run_mongodb_pipeline(user_id: str = None, limit: int = None, 
                        model: str = "qwen3:8b", batch_size: int = 1):
    """Run complete MongoDB pipeline"""
    
    print("🚀 Starting MongoDB Pipeline for LifafaV0")
    print("=" * 60)
    
    try:
        # Step 1: Connect to MongoDB
        print("📡 Connecting to MongoDB...")
        mongo_ops = MongoDBOperations()
        
        # Step 2: Get SMS data from MongoDB
        print("📱 Retrieving SMS data from MongoDB...")
        if user_id:
            sms_list = mongo_ops.get_user_sms_data(user_id, limit=limit, unprocessed_only=True)
            print(f"   User: {user_id}")
        else:
            sms_list = mongo_ops.get_all_sms_data(limit=limit, unprocessed_only=True)
            print(f"   All users")
        
        print(f"   Retrieved {len(sms_list)} unprocessed SMS")
        
        if not sms_list:
            print("✅ No unprocessed SMS found. Pipeline complete!")
            return
        
        # Step 3: Assign unique user IDs if missing
        print("🆔 Assigning unique user IDs...")
        sms_list = assign_unique_user_ids(sms_list)
        
        # Step 4: Filter financial SMS using sms_financial_filter.py
        print("🔍 Filtering financial SMS...")
        filter_instance = SMSFinancialFilter()
        filtered_data = filter_instance.filter_sms_dataset(sms_list)
        
        financial_sms = filtered_data['financial_sms']
        stats = filtered_data['statistics']
        
        print(f"   Financial SMS: {len(financial_sms)} out of {len(sms_list)}")
        print(f"   Financial percentage: {stats['financial_percentage']}%")
        
        if not financial_sms:
            print("✅ No financial SMS found. Pipeline complete!")
            return
        
        # Step 4.5: Store financial SMS in sms_fin_rawdata collection for tracking
        print("💾 Storing financial SMS in processing collection...")
        stored_count = mongo_ops.store_financial_raw_sms(financial_sms)
        print(f"   ✅ Stored {stored_count} financial SMS in sms_fin_rawdata collection")
        
        # Step 5: Extract financial array using extract_financial_array.py
        print("📋 Extracting financial array...")
        
        # Create temporary files for the pipeline steps
        temp_filtered = f"temp_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        temp_array = f"temp_array_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        temp_output = f"temp_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        temp_failures = f"temp_failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson"
        
        try:
            # Save filtered data (with financial_sms structure)
            with open(temp_filtered, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False, cls=JSONEncoder)
            
            # Extract financial array (convert to simple array format)
            extract_financial_array(temp_filtered, temp_array)
            
            # Verify the extracted array
            with open(temp_array, 'r', encoding='utf-8') as f:
                financial_array = json.load(f)
            
            print(f"   Extracted {len(financial_array)} financial SMS to array format")
            
            # Step 6: Process through LLM using main.py (file-based mode, not MongoDB mode)
            print("🤖 Processing financial SMS through LLM...")
            
            # Process through main.py (file-based mode for consistency)
            asyncio.run(process_all_batches(
                input_path=temp_array,
                output_path=temp_output,
                model=model,
                mode="openai",
                batch_size=batch_size,
                max_parallel_batches=1,
                temperature=0.1,
                max_tokens=4096,
                top_p=0.9,
                failures_path=f"temp_failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson",  # Provide default failures file
                enrich_mode="safe",
                use_mongodb=False,  # Use file-based mode for consistency
                user_id=user_id,
            ))
            
            # Step 7: Store results in MongoDB financial_transactions collection
            print("💾 Storing results in MongoDB...")
            
            # Read the processed results
            with open(temp_output, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            if results:
                # Store transactions in MongoDB
                stored_count = mongo_ops.store_financial_transactions_batch(results)
                print(f"   ✅ Stored {stored_count} transactions in financial_transactions collection")
                
                # Mark SMS as processed in sms_fin_rawdata collection
                success_count = 0
                for result in results:
                    source_id = result.get('_source_id')
                    if source_id:
                        success = mongo_ops.mark_financial_sms_as_processed(source_id, "success")
                        if success:
                            success_count += 1
                        else:
                            print(f"   ⚠️  Failed to mark SMS {source_id} as processed")
                    else:
                        print(f"   ⚠️  Result missing _source_id: {result.get('unique_id', 'NO_ID')}")
                
                print(f"   ✅ Updated SMS processing status in sms_fin_rawdata collection: {success_count}/{len(results)} SMS marked as processed")
            else:
                print("   ⚠️  No results to store")
            
            print("✅ MongoDB pipeline completed successfully!")
            
        finally:
            # Cleanup temp files
            for temp_file in [temp_filtered, temp_array, temp_output, temp_failures]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"🗑️  Cleaned up: {temp_file}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise
    finally:
        if 'mongo_ops' in locals():
            mongo_ops.close_connection()

def assign_unique_user_ids(sms_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign unique user IDs to SMS data if missing"""
    print("   🔍 Checking user_id assignment...")
    
    # Check if any SMS already has user_id
    has_user_id = any(sms.get('user_id') for sms in sms_list)
    
    if has_user_id:
        print("   ✅ User IDs already present in SMS data")
        return sms_list
    
    # Generate a unique user ID for this batch with transaction counter
    base_user_id = f"temp_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"   🆔 Assigning temporary user ID base: {base_user_id}")
    
    # Assign unique user_id to each SMS in this batch
    for i, sms in enumerate(sms_list):
        # Create unique user_id for each transaction
        sms['user_id'] = f"{base_user_id}_txn_{i+1:03d}"
        # Also ensure email_id is set (can be null for SMS)
        if 'email_id' not in sms:
            sms['email_id'] = None
    
    print(f"   ✅ Assigned unique user IDs to {len(sms_list)} SMS")
    print(f"   📋 Example user IDs: {sms_list[0]['user_id']}, {sms_list[1]['user_id'] if len(sms_list) > 1 else 'N/A'}")
    return sms_list

def main():
    parser = argparse.ArgumentParser(description='MongoDB Pipeline for LifafaV0')
    parser.add_argument('--user-id', help='Process SMS for specific user ID')
    parser.add_argument('--limit', type=int, help='Limit number of SMS to process')
    parser.add_argument('--model', default='qwen3:8b', help='LLM model to use')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Check environment variables
    required_env_vars = ['API_URL', 'MONGODB_URI']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("Please set:")
        for var in missing_vars:
            if var == 'API_URL':
                print(f"   export {var}='your_llm_endpoint'")
            elif var == 'MONGODB_URI':
                print(f"   export {var}='mongodb://localhost:27017/'")
        return 1
    
    try:
        run_mongodb_pipeline(
            user_id=args.user_id,
            limit=args.limit,
            model=args.model,
            batch_size=args.batch_size
        )
        return 0
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
