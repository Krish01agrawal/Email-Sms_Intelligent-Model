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
                        model: str = "qwen3:8b", batch_size: int = 5):
    """Run complete MongoDB pipeline with true parallel processing"""
    
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
        
        # Step 2.5: Filter out SMS that are already processed in sms_fin_rawdata
        print("🔍 Checking for already processed SMS...")
        already_processed_sms = mongo_ops.get_already_processed_sms_ids()
        if already_processed_sms:
            print(f"   Found {len(already_processed_sms)} already processed SMS IDs")
            print(f"   Already processed IDs: {', '.join(already_processed_sms[:5])}{'...' if len(already_processed_sms) > 5 else ''}")
            
            # Filter out already processed SMS by checking if they exist in sms_fin_rawdata
            original_count = len(sms_list)
            filtered_sms_list = []
            
            for sms in sms_list:
                sms_id = sms.get('unique_id') or sms.get('id')
                if not mongo_ops.is_sms_already_processed(sms_id):
                    filtered_sms_list.append(sms)
                else:
                    print(f"   ⏭️  Skipping already processed SMS: {sms_id}")
            
            sms_list = filtered_sms_list
            filtered_count = len(sms_list)
            
            if original_count != filtered_count:
                print(f"   ⚠️  Filtered out {original_count - filtered_count} already processed SMS")
                print(f"   ✅ Remaining unprocessed SMS: {filtered_count}")
            else:
                print(f"   ✅ No duplicate SMS found - all {filtered_count} SMS are new")
        else:
            print(f"   ✅ No already processed SMS found - all {len(sms_list)} SMS are new")
        
        if not sms_list:
            print("✅ All SMS have already been processed. Pipeline complete!")
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
            
            print(f"   ✅ Extracted {len(financial_array)} financial SMS to array format")
            print(f"   🎯 ONLY these {len(financial_array)} financial SMS will be processed by LLM")
            
            # CRITICAL VERIFICATION: Ensure we're not processing all SMS
            if len(financial_array) == len(sms_list):
                print(f"   ⚠️  WARNING: Financial SMS count equals total SMS count!")
                print(f"   🔍 This suggests the filter may not be working correctly")
                print(f"   📊 Total SMS: {len(sms_list)}, Financial SMS: {len(financial_array)}")
            else:
                print(f"   ✅ Filter working correctly: {len(financial_array)} financial out of {len(sms_list)} total")
            
            # Step 6: Process through LLM using main.py with TRUE PARALLEL PROCESSING
            print("🤖 Processing financial SMS through LLM...")
            print(f"   🚀 Batch Size: {batch_size} SMS per batch")
            print(f"   🔄 Parallel Batches: {batch_size} (enabling true concurrent processing)")
            
            # Calculate optimal parallel processing settings
            optimal_batch_size = min(batch_size, 10)  # Cap at 10 for stability
            parallel_batches = max(1, min(5, len(financial_array) // optimal_batch_size))  # Dynamic parallel processing
            
            print(f"   ⚡ Optimal Settings: {optimal_batch_size} SMS/batch, {parallel_batches} parallel batches")
            print(f"   📊 Total SMS to process: {len(financial_array)} (NOT {len(sms_list)})")
            
            # Process through main.py with TRUE PARALLEL PROCESSING
            print(f"   🔧 Calling process_all_batches with:")
            print(f"      - input_path: {temp_array} (contains ONLY {len(financial_array)} financial SMS)")
            print(f"      - output_path: {temp_output}")
            print(f"      - batch_size: {optimal_batch_size}")
            print(f"      - max_parallel_batches: {parallel_batches}")
            
            try:
                asyncio.run(process_all_batches(
                    input_path=temp_array,
                    output_path=temp_output,
                    model=model,
                    mode="openai",
                    batch_size=optimal_batch_size,  # Use optimal batch size
                    max_parallel_batches=parallel_batches,  # Enable true parallel processing
                    temperature=0.1,
                    max_tokens=4096,
                    top_p=0.9,
                    failures_path=f"temp_failures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson",
                    enrich_mode="safe",
                    use_mongodb=True,  # FIXED: Enable MongoDB updates
                    user_id=user_id,
                ))
                print(f"   ✅ process_all_batches completed successfully")
            except Exception as e:
                print(f"   ❌ process_all_batches failed: {e}")
                print(f"   🔍 Error type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                raise
            
            # Step 7: MongoDB storage now happens in real-time during processing
            print("💾 MongoDB storage completed in real-time during processing!")
            
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

def resume_processing(user_id: str = None) -> bool:
    """Resume processing from last checkpoint after server crash"""
    try:
        print("🔄 Checking for resume points...")
        
        # Initialize MongoDB operations
        mongo_ops = MongoDBOperations()
        
        # First, check for any data inconsistencies and recover them
        if user_id:
            print(f"🔍 Checking for data inconsistencies for user: {user_id}")
            
            # Check if there are processed SMS without transactions
            processed_sms_count = mongo_ops.fin_raw_collection.count_documents({
                "user_id": user_id,
                "isprocessed": True
            })
            
            transactions_count = mongo_ops.transactions_collection.count_documents({
                "user_id": user_id
            })
            
            print(f"  📊 Data consistency check:")
            print(f"     Processed SMS: {processed_sms_count}")
            print(f"     Stored transactions: {transactions_count}")
            
            if processed_sms_count > transactions_count:
                print(f"  ⚠️  Data inconsistency detected! {processed_sms_count - transactions_count} transactions missing")
                print(f"  🔄 Attempting to recover lost transactions...")
                
                recovered_count = mongo_ops.recover_unstored_transactions(user_id)
                if recovered_count > 0:
                    print(f"  ✅ Successfully recovered {recovered_count} transactions")
                else:
                    print(f"  ❌ Failed to recover transactions")
            else:
                print(f"  ✅ Data consistency verified")
        
        # Get resume point
        resume_point = mongo_ops.get_resume_point(user_id or "all_users")
        
        if not resume_point:
            print("✅ No resume point found - starting fresh")
            return False
        
        print(f"🔄 Resume point found:")
        print(f"   User: {resume_point['user_id']}")
        print(f"   Batch: {resume_point['batch_id']}")
        print(f"   Progress: {resume_point['processed_sms']}/{resume_point['total_sms']} SMS")
        print(f"   Last processed: {resume_point.get('last_processed_id', 'N/A')}")
        print(f"   Timestamp: {resume_point['checkpoint_timestamp']}")
        
        # Check if we can resume
        if resume_point['status'] == 'completed':
            print("✅ Processing already completed - no resume needed")
            return True
        
        # Resume processing
        print(f"🔄 Resuming processing from checkpoint...")
        
        # Get remaining SMS to process
        remaining_sms = resume_point['total_sms'] - resume_point['processed_sms']
        print(f"   Remaining SMS to process: {remaining_sms}")
        
        # Continue with remaining processing
        # This will automatically skip already processed SMS due to the filtering logic
        return True
        
    except Exception as e:
        print(f"❌ Error during resume: {e}")
        return False
    finally:
        if 'mongo_ops' in locals():
            mongo_ops.close_connection()

def main():
    parser = argparse.ArgumentParser(description='MongoDB Pipeline for LifafaV0 with True Parallel Processing')
    parser.add_argument('--user-id', help='Process SMS for specific user ID')
    parser.add_argument('--limit', type=int, help='Limit number of SMS to process')
    parser.add_argument('--model', default='qwen3:8b', help='LLM model to use')
    parser.add_argument('--batch-size', type=int, default=5, help='SMS per batch (default: 5, max: 10)')
    parser.add_argument('--parallel-batches', type=int, help='Number of parallel batches (auto-calculated if not specified)')
    parser.add_argument('--resume', action='store_true', help='Resume processing from last checkpoint')
    
    args = parser.parse_args()
    
    # Check for resume option
    if args.resume:
        print("🔄 RESUME MODE: Checking for processing checkpoints...")
        if resume_processing(args.user_id):
            print("🔄 Resume successful - continuing with processing")
        else:
            print("🔄 No resume point found - starting fresh processing")
    
    # Validate batch size
    if args.batch_size > 10:
        print(f"⚠️  Warning: Batch size {args.batch_size} exceeds recommended maximum of 10")
        print(f"   Setting batch size to 10 for stability")
        args.batch_size = 10
    
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
