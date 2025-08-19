#!/usr/bin/env python3
"""
SMS MongoDB Uploader
===================

Upload SMS data from JSON files to MongoDB database.
Supports batch uploads, duplicate detection, and data validation.

Usage:
    python3 sms_mongodb_uploader.py --input test_sms.json
    python3 sms_mongodb_uploader.py --input test_sms.json --batch-size 100
    python3 sms_mongodb_uploader.py --input test_sms.json --clear-collection
"""

import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
import os

try:
    from pymongo import MongoClient
    from pymongo.errors import BulkWriteError, ConnectionFailure, DuplicateKeyError
except ImportError:
    print("❌ PyMongo not installed. Install with: pip install pymongo")
    sys.exit(1)

# MongoDB Configuration
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "pluto_money"
COLLECTION_NAME = "sms_data"

class SMSMongoUploader:
    """SMS MongoDB Uploader with batch processing and error handling"""
    
    def __init__(self, connection_string: str = MONGODB_URI, 
                 db_name: str = DATABASE_NAME, collection_name: str = COLLECTION_NAME):
        self.connection_string = connection_string
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None
        
    def connect(self) -> bool:
        """Connect to MongoDB"""
        try:
            print(f"🔌 Connecting to MongoDB: {self.connection_string}")
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            
            print(f"✅ Connected to database: {self.db_name}")
            print(f"📁 Using collection: {self.collection_name}")
            return True
            
        except ConnectionFailure as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            print("💡 Make sure MongoDB is running on localhost:27017")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("🔌 Disconnected from MongoDB")
    
    def validate_sms_data(self, sms_data: List[Dict[str, Any]], user_id: str = None) -> List[Dict[str, Any]]:
        """Validate and normalize SMS data with user_id assignment"""
        validated_data = []
        errors = []
        
        # Generate user_id from filename if not provided
        if not user_id:
            user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"   🆔 Assigning user_id: {user_id}")
        
        for i, sms in enumerate(sms_data):
            try:
                # Create normalized document
                doc = {
                    "sender": sms.get("sender", ""),
                    "body": sms.get("body", ""),
                    "date": sms.get("date", ""),
                    "type": sms.get("type", "received"),
                    "user_id": user_id,  # Assign user_id to all SMS
                    "email_id": None,    # SMS don't have email_id
                    "uploaded_at": datetime.utcnow(),
                    "source_index": i + 1
                }
                
                # Add optional fields if present
                if "id" in sms:
                    doc["original_id"] = sms["id"]
                
                # Validate required fields
                if not doc["sender"] and not doc["body"]:
                    errors.append(f"SMS {i+1}: Missing both sender and body")
                    continue
                
                # Create unique identifier for duplicate detection
                doc["unique_id"] = f"{user_id}_sms_{i+1:06d}"
                
                validated_data.append(doc)
                
            except Exception as e:
                errors.append(f"SMS {i+1}: Validation error - {e}")
        
        if errors:
            print(f"⚠️  Found {len(errors)} validation errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"   - {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more errors")
        
        print(f"✅ Validated {len(validated_data)} SMS documents for user: {user_id}")
        return validated_data
    
    def create_indexes(self):
        """Create indexes for better performance"""
        try:
            # Create indexes
            indexes = [
                ("user_id", 1),      # For user-based queries
                ("unique_id", 1),    # For duplicate detection
                ("sender", 1),       # For sender-based queries
                ("date", -1),        # For date-based queries (descending)
                ("uploaded_at", -1), # For upload tracking
                ("type", 1)          # For type-based filtering
            ]
            
            for field, direction in indexes:
                try:
                    self.collection.create_index([(field, direction)], background=True)
                    print(f"📊 Created index on: {field}")
                except Exception as e:
                    print(f"⚠️  Index creation warning for {field}: {e}")
            
            # Create unique index for duplicate prevention
            try:
                self.collection.create_index(
                    [("unique_id", 1)], 
                    unique=True, 
                    background=True,
                    name="unique_sms_idx"
                )
                print("🔒 Created unique index for duplicate prevention")
            except Exception as e:
                print(f"⚠️  Unique index warning: {e}")
                
        except Exception as e:
            print(f"❌ Error creating indexes: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            stats = self.db.command("collStats", self.collection_name)
            count = self.collection.count_documents({})
            
            return {
                "document_count": count,
                "storage_size": stats.get("storageSize", 0),
                "avg_document_size": stats.get("avgObjSize", 0),
                "total_indexes": stats.get("nindexes", 0)
            }
        except Exception as e:
            print(f"⚠️  Could not get collection stats: {e}")
            return {"document_count": 0}
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            result = self.collection.delete_many({})
            print(f"🗑️  Cleared {result.deleted_count} documents from collection")
            return True
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
            return False
    
    def upload_batch(self, documents: List[Dict[str, Any]]) -> Dict[str, int]:
        """Upload a batch of documents"""
        try:
            result = self.collection.insert_many(documents, ordered=False)
            return {
                "inserted": len(result.inserted_ids),
                "duplicates": 0,
                "errors": 0
            }
        except BulkWriteError as e:
            # Handle bulk write errors (mostly duplicates)
            inserted = e.details.get("nInserted", 0)
            duplicates = len([err for err in e.details.get("writeErrors", []) 
                            if err.get("code") == 11000])  # Duplicate key error
            errors = len(e.details.get("writeErrors", [])) - duplicates
            
            return {
                "inserted": inserted,
                "duplicates": duplicates,
                "errors": errors
            }
        except Exception as e:
            print(f"❌ Batch upload error: {e}")
            return {"inserted": 0, "duplicates": 0, "errors": len(documents)}
    
    def upload_sms_data(self, sms_data: List[Dict[str, Any]], batch_size: int = 100) -> Dict[str, int]:
        """Upload SMS data in batches"""
        print(f"📤 Starting upload of {len(sms_data)} SMS documents...")
        print(f"📦 Batch size: {batch_size}")
        
        total_stats = {"inserted": 0, "duplicates": 0, "errors": 0}
        
        # Process in batches
        for i in range(0, len(sms_data), batch_size):
            batch = sms_data[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(sms_data) + batch_size - 1) // batch_size
            
            print(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            batch_stats = self.upload_batch(batch)
            
            # Update totals
            for key in total_stats:
                total_stats[key] += batch_stats[key]
            
            print(f"   ✅ Inserted: {batch_stats['inserted']}, "
                  f"Duplicates: {batch_stats['duplicates']}, "
                  f"Errors: {batch_stats['errors']}")
        
        return total_stats
    
    def load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load SMS data from JSON file"""
        try:
            print(f"📁 Loading SMS data from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                if 'sms' in data:
                    sms_list = data['sms']
                elif 'data' in data:
                    sms_list = data['data']
                else:
                    # Assume the dict itself is the data
                    sms_list = [data]
            elif isinstance(data, list):
                sms_list = data
            else:
                raise ValueError("Invalid JSON format")
            
            print(f"📊 Loaded {len(sms_list)} SMS records")
            return sms_list
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return []
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description="Upload SMS data to MongoDB")
    parser.add_argument("--input", required=True, help="Path to SMS JSON file")
    parser.add_argument("--user-id", help="User ID to assign to all SMS in this file")
    parser.add_argument("--connection", default=MONGODB_URI, help="MongoDB connection string")
    parser.add_argument("--database", default=DATABASE_NAME, help="Database name")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Collection name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for uploads")
    parser.add_argument("--clear-collection", action="store_true", help="Clear collection before upload")
    parser.add_argument("--create-indexes", action="store_true", help="Create performance indexes")
    parser.add_argument("--stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    # Generate user_id from filename if not provided
    if not args.user_id:
        filename = os.path.basename(args.input)
        # Extract name from filename (e.g., sms_data_divyam.json -> divyam)
        if filename.startswith("sms_data_") and filename.endswith(".json"):
            args.user_id = filename[10:-5]  # Remove "sms_data_" prefix and ".json" suffix
        else:
            args.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🆔 Using user_id: {args.user_id}")
    
    # Initialize uploader
    uploader = SMSMongoUploader(
        connection_string=args.connection,
        db_name=args.database,
        collection_name=args.collection
    )
    
    try:
        # Connect to MongoDB
        if not uploader.connect():
            return 1
        
        # Show initial stats
        if args.stats:
            print("\n📊 INITIAL COLLECTION STATISTICS:")
            stats = uploader.get_collection_stats()
            print(f"   Documents: {stats['document_count']:,}")
            print(f"   Storage Size: {stats.get('storage_size', 0):,} bytes")
            print(f"   Indexes: {stats.get('total_indexes', 0)}")
        
        # Clear collection if requested
        if args.clear_collection:
            print(f"\n🗑️  Clearing collection '{args.collection}'...")
            uploader.clear_collection()
        
        # Create indexes if requested
        if args.create_indexes:
            print("\n📊 Creating performance indexes...")
            uploader.create_indexes()
        
        # Load SMS data
        print(f"\n📁 Loading SMS data from: {args.input}")
        sms_data = uploader.load_json_file(args.input)
        
        if not sms_data:
            print("❌ No SMS data to upload")
            return 1
        
        # Validate data with user_id
        print(f"\n✅ Validating {len(sms_data)} SMS records...")
        validated_data = uploader.validate_sms_data(sms_data, args.user_id)
        
        if not validated_data:
            print("❌ No valid SMS data to upload")
            return 1
        
        # Upload data
        print(f"\n📤 Uploading {len(validated_data)} SMS records...")
        upload_stats = uploader.upload_sms_data(validated_data, args.batch_size)
        
        # Show results
        print(f"\n📊 UPLOAD SUMMARY:")
        print(f"   User ID: {args.user_id}")
        print(f"   Total Records: {len(sms_data)}")
        print(f"   Valid Records: {len(validated_data)}")
        print(f"   Successfully Inserted: {upload_stats['inserted']}")
        print(f"   Duplicates Skipped: {upload_stats['duplicates']}")
        print(f"   Errors: {upload_stats['errors']}")
        
        success_rate = (upload_stats['inserted'] / len(validated_data)) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
        
        # Show final stats
        if args.stats:
            print(f"\n📊 FINAL COLLECTION STATISTICS:")
            stats = uploader.get_collection_stats()
            print(f"   Total Documents: {stats['document_count']:,}")
            print(f"   Storage Size: {stats.get('storage_size', 0):,} bytes")
        
        print(f"\n✅ Upload completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Upload interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    finally:
        uploader.disconnect()

if __name__ == "__main__":
    exit(main())
