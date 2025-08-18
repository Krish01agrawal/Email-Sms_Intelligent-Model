#!/usr/bin/env python3
"""
MongoDB Operations for LifafaV0
===============================

Handles MongoDB operations for:
1. Reading SMS data from sms_data collection
2. Storing processed financial transactions in financial_transactions collection
3. Managing user-specific data and batch processing
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import logging
from pymongo import UpdateOne, InsertOne

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MongoDBOperations:
    """MongoDB operations for LifafaV0 financial data pipeline"""
    
    def __init__(self, connection_string: str = None, db_name: str = None):
        """Initialize MongoDB connection"""
        try:
            # Use environment variable or default
            if connection_string is None:
                connection_string = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            
            if db_name is None:
                db_name = os.getenv('MONGODB_DB', 'pluto_money')
            
            self.client = MongoClient(connection_string)
            self.db = self.client[db_name]
            self.db_name = db_name
            
            # Collections
            self.sms_collection = self.db['sms_data']  # Raw SMS (never modified)
            self.fin_raw_collection = self.db['sms_fin_rawdata']  # Financial SMS with processing status
            self.transactions_collection = self.db['financial_transactions']  # Final processed results
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"✅ Connected to MongoDB: {connection_string}")
            logger.info(f"✅ Database: {self.db_name}")
            
            # Create indexes
            self._create_indexes()
            
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ MongoDB initialization error: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # SMS data collection indexes
            self.sms_collection.create_index([("user_id", 1)])
            self.sms_collection.create_index([("timestamp", -1)])
            
            # Financial raw data collection indexes
            self.fin_raw_collection.create_index([("unique_id", 1)], unique=True)
            self.fin_raw_collection.create_index([("isprocessed", 1)])
            self.fin_raw_collection.create_index([("user_id", 1)])
            self.fin_raw_collection.create_index([("processing_timestamp", -1)])
            
            # Financial transactions collection indexes
            self.transactions_collection.create_index([("unique_id", 1)], unique=True)
            self.transactions_collection.create_index([("user_id", 1)])
            self.transactions_collection.create_index([("transaction_date", -1)])
            
            logger.info("✅ MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
            raise
    
    def store_financial_raw_sms(self, financial_sms: List[Dict[str, Any]]) -> int:
        """Store financial SMS in sms_fin_rawdata collection"""
        try:
            if not financial_sms:
                return 0
            
            # Prepare financial SMS for storage
            for sms in financial_sms:
                # Ensure required fields
                if 'isprocessed' not in sms:
                    sms['isprocessed'] = False
                if 'processing_timestamp' not in sms:
                    sms['processing_timestamp'] = None
                if 'processing_status' not in sms:
                    sms['processing_status'] = None
                if 'created_at' not in sms:
                    sms['created_at'] = datetime.now()
                if 'updated_at' not in sms:
                    sms['updated_at'] = datetime.now()
            
            # Use bulk operations for better performance
            bulk_operations = []
            for sms in financial_sms:
                bulk_operations.append(
                    UpdateOne(
                        {"unique_id": sms["unique_id"]},
                        {"$set": sms},
                        upsert=True
                    )
                )
            
            if bulk_operations:
                result = self.fin_raw_collection.bulk_write(bulk_operations)
                logger.info(f"💾 Stored {len(financial_sms)} financial SMS in sms_fin_rawdata collection")
                return len(financial_sms)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error storing financial raw SMS: {e}")
            return 0
    
    def get_financial_raw_sms(self, user_id: str = None, unprocessed_only: bool = True, limit: int = None) -> List[Dict[str, Any]]:
        """Get financial SMS from sms_fin_rawdata collection"""
        try:
            # Build query
            query = {}
            if user_id:
                query["user_id"] = user_id
            if unprocessed_only:
                query["isprocessed"] = {"$ne": True}
            
            # Execute query
            cursor = self.fin_raw_collection.find(query)
            
            if limit:
                cursor = cursor.limit(limit)
            
            sms_list = list(cursor)
            logger.info(f"📱 Retrieved {len(sms_list)} financial SMS from sms_fin_rawdata collection")
            return sms_list
            
        except Exception as e:
            logger.error(f"❌ Error retrieving financial raw SMS: {e}")
            return []
    
    def mark_financial_sms_as_processed(self, unique_id: str, status: str = "success") -> bool:
        """Mark financial SMS as processed in sms_fin_rawdata collection"""
        try:
            result = self.fin_raw_collection.update_one(
                {"unique_id": unique_id},
                {
                    "$set": {
                        "isprocessed": True,
                        "processing_timestamp": datetime.now(),
                        "processing_status": status,
                        "updated_at": datetime.now()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"🔄 Marked financial SMS {unique_id} as processed ({status})")
                return True
            else:
                logger.warning(f"⚠️  Financial SMS {unique_id} not found for marking as processed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error marking financial SMS as processed: {e}")
            return False
    
    def get_user_sms_data(self, user_id: str, limit: int = None, unprocessed_only: bool = True) -> List[Dict[str, Any]]:
        """Get SMS data for a specific user"""
        try:
            query = {"user_id": user_id}
            if unprocessed_only:
                query["isprocessed"] = {"$ne": True}
            
            cursor = self.sms_collection.find(query).sort("date", -1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            sms_list = list(cursor)
            logger.info(f"📱 Retrieved {len(sms_list)} SMS for user {user_id}")
            return sms_list
            
        except Exception as e:
            logger.error(f"❌ Error retrieving SMS data: {e}")
            return []
    
    def get_all_sms_data(self, unprocessed_only: bool = True, limit: int = None) -> List[Dict[str, Any]]:
        """Get all SMS data (for batch processing)"""
        try:
            query = {}
            if unprocessed_only:
                query["isprocessed"] = {"$ne": True}
            
            cursor = self.sms_collection.find(query).sort("date", -1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            sms_list = list(cursor)
            logger.info(f"📱 Retrieved {len(sms_list)} SMS (all users)")
            return sms_list
            
        except Exception as e:
            logger.error(f"❌ Error retrieving all SMS data: {e}")
            return []
    
    def store_financial_transaction(self, transaction_data: Dict[str, Any]) -> bool:
        """Store a single financial transaction"""
        try:
            # Add metadata
            transaction_data["created_at"] = datetime.now()
            transaction_data["updated_at"] = datetime.now()
            
            # Ensure unique_id is unique
            if "unique_id" in transaction_data:
                # Use upsert to avoid duplicates
                result = self.transactions_collection.update_one(
                    {"unique_id": transaction_data["unique_id"]},
                    {"$set": transaction_data},
                    upsert=True
                )
                
                if result.upserted_id:
                    logger.info(f"💾 Stored new transaction: {transaction_data.get('unique_id', 'NO_ID')}")
                else:
                    logger.info(f"🔄 Updated existing transaction: {transaction_data.get('unique_id', 'NO_ID')}")
                
                return True
            else:
                # Insert without unique_id
                result = self.transactions_collection.insert_one(transaction_data)
                logger.info(f"💾 Stored transaction with ID: {result.inserted_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error storing transaction: {e}")
            return False
    
    def store_financial_transactions_batch(self, transactions: List[Dict[str, Any]]) -> int:
        """Store multiple financial transactions in batch"""
        try:
            if not transactions:
                return 0
            
            # Add metadata to each transaction
            for transaction in transactions:
                transaction["created_at"] = datetime.now()
                transaction["updated_at"] = datetime.now()
            
            # Use bulk operations for better performance
            bulk_operations = []
            for transaction in transactions:
                # Clean the transaction document for MongoDB
                clean_transaction = self._clean_transaction_document(transaction)
                
                if "unique_id" in clean_transaction:
                    # Upsert operation - use proper MongoDB syntax
                    bulk_operations.append(
                        UpdateOne(
                            {"unique_id": clean_transaction["unique_id"]},
                            {"$set": clean_transaction},
                            upsert=True
                        )
                    )
                else:
                    # Insert operation - use proper MongoDB syntax
                    bulk_operations.append(
                        InsertOne(clean_transaction)
                    )
            
            if bulk_operations:
                result = self.transactions_collection.bulk_write(bulk_operations)
                logger.info(f"💾 Batch stored {len(transactions)} transactions")
                return len(transactions)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error in batch transaction storage: {e}")
            # Try individual inserts as fallback
            return self._fallback_individual_inserts(transactions)
    
    def _clean_transaction_document(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Clean transaction document for MongoDB storage"""
        try:
            # Create a clean copy
            clean_transaction = {}
            
            for key, value in transaction.items():
                # Skip None values
                if value is None:
                    continue
                
                # Handle different data types
                if isinstance(value, (str, int, float, bool)):
                    clean_transaction[key] = value
                elif isinstance(value, dict):
                    # Recursively clean nested dictionaries
                    clean_transaction[key] = self._clean_transaction_document(value)
                elif isinstance(value, list):
                    # Clean list items
                    clean_transaction[key] = [
                        self._clean_transaction_document(item) if isinstance(item, dict) else item
                        for item in value if item is not None
                    ]
                elif isinstance(value, datetime):
                    clean_transaction[key] = value
                else:
                    # Convert other types to string
                    clean_transaction[key] = str(value)
            
            return clean_transaction
            
        except Exception as e:
            logger.error(f"❌ Error cleaning transaction document: {e}")
            return transaction
    
    def _fallback_individual_inserts(self, transactions: List[Dict[str, Any]]) -> int:
        """Fallback to individual inserts if bulk operations fail"""
        try:
            success_count = 0
            for transaction in transactions:
                try:
                    clean_transaction = self._clean_transaction_document(transaction)
                    result = self.transactions_collection.insert_one(clean_transaction)
                    if result.inserted_id:
                        success_count += 1
                except Exception as e:
                    logger.error(f"❌ Failed to insert individual transaction: {e}")
                    continue
            
            logger.info(f"💾 Fallback: Inserted {success_count} transactions individually")
            return success_count
            
        except Exception as e:
            logger.error(f"❌ Fallback individual inserts failed: {e}")
            return 0
    
    def mark_sms_as_processed(self, unique_id: str, status: str = "success") -> bool:
        """Mark SMS as processed in the database"""
        try:
            result = self.sms_collection.update_one(
                {"unique_id": unique_id},
                {
                    "$set": {
                        "isprocessed": True,
                        "processing_timestamp": datetime.now(),
                        "processing_status": status
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"🔄 Marked SMS {unique_id} as processed ({status})")
                return True
            else:
                logger.warning(f"⚠️  SMS {unique_id} not found for marking as processed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error marking SMS as processed: {e}")
            return False
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        try:
            # Raw SMS collection stats
            total_sms = self.sms_collection.count_documents({})
            
            # Financial raw data collection stats
            total_financial_sms = self.fin_raw_collection.count_documents({})
            processed_financial_sms = self.fin_raw_collection.count_documents({"isprocessed": True})
            unprocessed_financial_sms = self.fin_raw_collection.count_documents({"isprocessed": {"$ne": True}})
            
            # Financial transactions collection stats
            total_transactions = self.transactions_collection.count_documents({})
            
            stats = {
                "total_raw_sms": total_sms,
                "total_financial_sms": total_financial_sms,
                "processed_financial_sms": processed_financial_sms,
                "unprocessed_financial_sms": unprocessed_financial_sms,
                "total_transactions": total_transactions,
                "financial_processing_percentage": round((processed_financial_sms / total_financial_sms * 100), 2) if total_financial_sms > 0 else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting processing stats: {e}")
            return {}
    
    def close_connection(self):
        """Close MongoDB connection"""
        try:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")
        except Exception as e:
            logger.error(f"❌ Error closing MongoDB connection: {e}")

def test_mongodb_connection():
    """Test MongoDB connection and basic operations"""
    try:
        mongo_ops = MongoDBOperations()
        
        # Test basic operations
        stats = mongo_ops.get_processing_stats()
        print(f"📊 MongoDB Connection Test Results:")
        print(f"   Database: {mongo_ops.db_name}")
        print(f"   SMS Collection: {mongo_ops.sms_collection.name}")
        print(f"   Transactions Collection: {mongo_ops.transactions_collection.name}")
        print(f"   Total SMS: {stats.get('total_sms', 0)}")
        print(f"   Total Transactions: {stats.get('total_transactions', 0)}")
        
        mongo_ops.close_connection()
        print("✅ MongoDB connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_mongodb_connection()
