"""
MongoDB Integration for Email-SMS Intelligent Model
Handles storing processed emails and extracted financial data
"""

import pymongo
from pymongo import MongoClient
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from bson import ObjectId
import os
from dotenv import load_dotenv

from config import mongodb_config
from data_preprocessing import TextPreprocessor, DataExtractor
from model import EmailSMSModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBManager:
    """MongoDB connection and operations manager"""
    
    def __init__(self, connection_string: str = None, database_name: str = None):
        self.connection_string = connection_string or mongodb_config.connection_string
        self.database_name = database_name or mongodb_config.database_name
        
        # Initialize connection
        self.client = MongoClient(self.connection_string)
        self.db = self.client[self.database_name]
        
        # Collections
        self.email_logs = self.db[mongodb_config.email_logs_collection]
        self.financial_transactions = self.db[mongodb_config.financial_transactions_collection]
        self.sms_logs = self.db[mongodb_config.sms_logs_collection]
        
        # Create indexes for better performance
        self._create_indexes()
        
        logger.info(f"Connected to MongoDB: {self.database_name}")
    
    def _create_indexes(self):
        """Create database indexes for better performance"""
        try:
            # Email logs indexes
            self.email_logs.create_index([("user_id", pymongo.ASCENDING)])
            self.email_logs.create_index([("email_id", pymongo.ASCENDING)])
            self.email_logs.create_index([("created_at", pymongo.DESCENDING)])
            self.email_logs.create_index([("is_financial", pymongo.ASCENDING)])
            
            # Financial transactions indexes
            self.financial_transactions.create_index([("user_id", pymongo.ASCENDING)])
            self.financial_transactions.create_index([("email_id", pymongo.ASCENDING)])
            self.financial_transactions.create_index([("transaction_date", pymongo.DESCENDING)])
            self.financial_transactions.create_index([("transaction_type", pymongo.ASCENDING)])
            self.financial_transactions.create_index([("merchant_canonical", pymongo.ASCENDING)])
            self.financial_transactions.create_index([("amount", pymongo.ASCENDING)])
            
            # SMS logs indexes
            self.sms_logs.create_index([("user_id", pymongo.ASCENDING)])
            self.sms_logs.create_index([("created_at", pymongo.DESCENDING)])
            self.sms_logs.create_index([("is_financial", pymongo.ASCENDING)])
            
            logger.info("Database indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes: {e}")
    
    def store_email_data(self, email_data: Dict[str, Any]) -> str:
        """Store email data in MongoDB"""
        try:
            # Add metadata
            email_data['created_at'] = datetime.utcnow()
            email_data['updated_at'] = datetime.utcnow()
            
            # Insert into email_logs collection
            result = self.email_logs.insert_one(email_data)
            
            logger.info(f"Email stored with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error storing email data: {e}")
            raise
    
    def store_financial_transaction(self, transaction_data: Dict[str, Any]) -> str:
        """Store financial transaction data in MongoDB"""
        try:
            # Add metadata
            transaction_data['created_at'] = datetime.utcnow()
            transaction_data['updated_at'] = datetime.utcnow()
            
            # Insert into financial_transactions collection
            result = self.financial_transactions.insert_one(transaction_data)
            
            logger.info(f"Financial transaction stored with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error storing financial transaction: {e}")
            raise
    
    def store_sms_data(self, sms_data: Dict[str, Any]) -> str:
        """Store SMS data in MongoDB"""
        try:
            # Add metadata
            sms_data['created_at'] = datetime.utcnow()
            sms_data['updated_at'] = datetime.utcnow()
            
            # Insert into sms_logs collection
            result = self.sms_logs.insert_one(sms_data)
            
            logger.info(f"SMS stored with ID: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error storing SMS data: {e}")
            raise
    
    def get_user_transactions(self, user_id: str, start_date: datetime = None, 
                            end_date: datetime = None, transaction_type: str = None) -> List[Dict]:
        """Get financial transactions for a user with optional filters"""
        try:
            query = {"user_id": user_id}
            
            # Add date filters
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query["transaction_date"] = date_filter
            
            # Add transaction type filter
            if transaction_type:
                query["transaction_type"] = transaction_type
            
            # Execute query
            transactions = list(self.financial_transactions.find(query).sort("transaction_date", -1))
            
            logger.info(f"Retrieved {len(transactions)} transactions for user {user_id}")
            return transactions
            
        except Exception as e:
            logger.error(f"Error retrieving transactions: {e}")
            raise
    
    def get_user_emails(self, user_id: str, is_financial: bool = None, 
                       start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get emails for a user with optional filters"""
        try:
            query = {"user_id": user_id}
            
            # Add financial filter
            if is_financial is not None:
                query["is_financial"] = is_financial
            
            # Add date filters
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query["created_at"] = date_filter
            
            # Execute query
            emails = list(self.email_logs.find(query).sort("created_at", -1))
            
            logger.info(f"Retrieved {len(emails)} emails for user {user_id}")
            return emails
            
        except Exception as e:
            logger.error(f"Error retrieving emails: {e}")
            raise
    
    def get_user_sms(self, user_id: str, is_financial: bool = None,
                    start_date: datetime = None, end_date: datetime = None) -> List[Dict]:
        """Get SMS for a user with optional filters"""
        try:
            query = {"user_id": user_id}
            
            # Add financial filter
            if is_financial is not None:
                query["is_financial"] = is_financial
            
            # Add date filters
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                query["created_at"] = date_filter
            
            # Execute query
            sms_list = list(self.sms_logs.find(query).sort("created_at", -1))
            
            logger.info(f"Retrieved {len(sms_list)} SMS for user {user_id}")
            return sms_list
            
        except Exception as e:
            logger.error(f"Error retrieving SMS: {e}")
            raise
    
    def get_transaction_summary(self, user_id: str, start_date: datetime = None, 
                              end_date: datetime = None) -> Dict:
        """Get transaction summary for a user"""
        try:
            pipeline = [
                {"$match": {"user_id": user_id}}
            ]
            
            # Add date filter
            if start_date or end_date:
                date_filter = {}
                if start_date:
                    date_filter["$gte"] = start_date
                if end_date:
                    date_filter["$lte"] = end_date
                pipeline[0]["$match"]["transaction_date"] = date_filter
            
            # Add aggregation stages
            pipeline.extend([
                {
                    "$group": {
                        "_id": "$transaction_type",
                        "count": {"$sum": 1},
                        "total_amount": {"$sum": "$amount"},
                        "avg_amount": {"$avg": "$amount"},
                        "min_amount": {"$min": "$amount"},
                        "max_amount": {"$max": "$amount"}
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "total_transactions": {"$sum": "$count"},
                        "total_amount": {"$sum": "$total_amount"},
                        "transaction_types": {
                            "$push": {
                                "type": "$_id",
                                "count": "$count",
                                "total_amount": "$total_amount",
                                "avg_amount": "$avg_amount",
                                "min_amount": "$min_amount",
                                "max_amount": "$max_amount"
                            }
                        }
                    }
                }
            ])
            
            result = list(self.financial_transactions.aggregate(pipeline))
            
            if result:
                summary = result[0]
                summary.pop("_id", None)
            else:
                summary = {
                    "total_transactions": 0,
                    "total_amount": 0,
                    "transaction_types": []
                }
            
            logger.info(f"Generated transaction summary for user {user_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating transaction summary: {e}")
            raise

class EmailSMSProcessor:
    """Process emails and SMS using the trained model and store in MongoDB"""
    
    def __init__(self, model_path: str = None, mongodb_manager: MongoDBManager = None):
        # Initialize model
        if model_path and os.path.exists(model_path):
            self.model = EmailSMSModel()
            self.model.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            self.model = EmailSMSModel()
            logger.info("Using default model")
        
        # Initialize MongoDB manager
        self.mongodb = mongodb_manager or MongoDBManager()
        
        # Initialize preprocessing components
        self.preprocessor = TextPreprocessor()
        self.data_extractor = DataExtractor()
    
    def process_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single email and store results"""
        try:
            # Extract email content
            subject = email_data.get('subject', '')
            body = email_data.get('body', '')
            sender = email_data.get('sender', '')
            recipient = email_data.get('recipient', '')
            email_id = email_data.get('email_id', '')
            user_id = email_data.get('user_id', '')
            
            # Combine subject and body for analysis
            full_text = f"{subject} {body}".strip()
            
            # Predict using model
            prediction = self.model.predict(full_text)
            
            # Prepare email document
            email_doc = {
                "user_id": user_id,
                "email_id": email_id,
                "sender": sender,
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "full_text": full_text,
                "cleaned_text": prediction['cleaned_text'],
                "is_financial": prediction['is_financial'],
                "financial_confidence": prediction['financial_confidence'],
                "prediction_metadata": {
                    "model_used": self.model.model_name,
                    "prediction_timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Store email in MongoDB
            email_doc_id = self.mongodb.store_email_data(email_doc)
            email_doc['_id'] = email_doc_id
            
            # If financial, extract and store transaction data
            if prediction['is_financial'] and prediction['extraction']:
                transaction_doc = self._create_transaction_document(
                    email_data, prediction, email_doc_id
                )
                transaction_id = self.mongodb.store_financial_transaction(transaction_doc)
                email_doc['transaction_id'] = transaction_id
            
            logger.info(f"Processed email {email_id} - Financial: {prediction['is_financial']}")
            return email_doc
            
        except Exception as e:
            logger.error(f"Error processing email: {e}")
            raise
    
    def process_sms(self, sms_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single SMS and store results"""
        try:
            # Extract SMS content
            message = sms_data.get('message', '')
            sender = sms_data.get('sender', '')
            recipient = sms_data.get('recipient', '')
            sms_id = sms_data.get('sms_id', '')
            user_id = sms_data.get('user_id', '')
            
            # Predict using model
            prediction = self.model.predict(message)
            
            # Prepare SMS document
            sms_doc = {
                "user_id": user_id,
                "sms_id": sms_id,
                "sender": sender,
                "recipient": recipient,
                "message": message,
                "cleaned_text": prediction['cleaned_text'],
                "is_financial": prediction['is_financial'],
                "financial_confidence": prediction['financial_confidence'],
                "prediction_metadata": {
                    "model_used": self.model.model_name,
                    "prediction_timestamp": datetime.utcnow().isoformat()
                }
            }
            
            # Store SMS in MongoDB
            sms_doc_id = self.mongodb.store_sms_data(sms_doc)
            sms_doc['_id'] = sms_doc_id
            
            # If financial, extract and store transaction data
            if prediction['is_financial'] and prediction['extraction']:
                transaction_doc = self._create_transaction_document(
                    sms_data, prediction, sms_doc_id, is_sms=True
                )
                transaction_id = self.mongodb.store_financial_transaction(transaction_doc)
                sms_doc['transaction_id'] = transaction_id
            
            logger.info(f"Processed SMS {sms_id} - Financial: {prediction['is_financial']}")
            return sms_doc
            
        except Exception as e:
            logger.error(f"Error processing SMS: {e}")
            raise
    
    def _create_transaction_document(self, original_data: Dict, prediction: Dict, 
                                   source_id: str, is_sms: bool = False) -> Dict:
        """Create transaction document from extracted data"""
        extraction = prediction['extraction']
        
        # Determine source type
        source_type = "sms" if is_sms else "email"
        
        # Create transaction document
        transaction_doc = {
            "user_id": original_data.get('user_id', ''),
            f"{source_type}_id": source_id,
            "transaction_type": extraction.get('transaction_type', 'unknown'),
            "amount": extraction.get('amount'),
            "currency": extraction.get('currency', 'INR'),
            "transaction_date": extraction.get('date'),
            "merchant_canonical": extraction.get('merchant'),
            "merchant_original": extraction.get('merchant'),
            "merchant_patterns": [extraction.get('merchant')] if extraction.get('merchant') else [],
            "service_category": self._categorize_service(extraction.get('merchant', '')),
            "service_name": extraction.get('merchant', ''),
            "payment_method": None,
            "payment_status": "completed",
            "transaction_reference": None,
            "invoice_number": None,
            "order_id": None,
            "receipt_number": None,
            "bank_name": None,
            "account_number": None,
            "upi_id": None,
            "is_subscription": False,
            "subscription_frequency": None,
            "next_renewal_date": None,
            "is_automatic_payment": False,
            "total_amount": extraction.get('amount'),
            "base_amount": None,
            "tax_amount": None,
            "discount_amount": None,
            "late_fee_amount": None,
            "processing_fee": None,
            "cashback_amount": None,
            "billing_period_start": None,
            "billing_period_end": None,
            "bank_details": {
                "bank_name": None,
                "account_number": None
            },
            "upi_details": {
                "transaction_flow": {
                    "direction": None,
                    "description": None
                },
                "receiver": {
                    "upi_id": None,
                    "name": None,
                    "upi_app": None
                }
            },
            "card_details": {},
            "subscription_details": {
                "is_subscription": False,
                "product_name": None,
                "category": None,
                "type": None,
                "confidence_score": 0,
                "detection_reasons": []
            },
            "primary_category": "finance",
            "secondary_category": self._categorize_service(extraction.get('merchant', '')),
            "tertiary_category": "Banking",
            "confidence_score": extraction.get('extraction_confidence', 0),
            "extraction_confidence": extraction.get('extraction_confidence', 0),
            "source_type": source_type,
            "extraction_metadata": {
                "model_used": self.model.model_name,
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "extracted_fields": extraction.get('extracted_fields', [])
            }
        }
        
        return transaction_doc
    
    def _categorize_service(self, merchant: str) -> str:
        """Categorize service based on merchant name"""
        if not merchant:
            return "Other"
        
        merchant_lower = merchant.lower()
        
        # Banking
        if any(bank in merchant_lower for bank in ['hdfc', 'sbi', 'icici', 'axis', 'kotak', 'yes bank']):
            return "Banking"
        
        # Investment
        if any(inv in merchant_lower for inv in ['groww', 'zerodha', 'upstox', 'angel', 'mutual fund']):
            return "Investment"
        
        # E-commerce
        if any(ec in merchant_lower for ec in ['amazon', 'flipkart', 'myntra', 'nykaa']):
            return "E-commerce"
        
        # Food Delivery
        if any(fd in merchant_lower for fd in ['zomato', 'swiggy', 'blinkit', 'dunzo']):
            return "Food_Delivery"
        
        # Transportation
        if any(trans in merchant_lower for trans in ['uber', 'ola', 'rapido', 'bmtc']):
            return "Transportation"
        
        # Entertainment
        if any(ent in merchant_lower for ent in ['netflix', 'prime', 'hotstar', 'sony']):
            return "Entertainment"
        
        # Utilities
        if any(util in merchant_lower for util in ['electricity', 'water', 'gas', 'internet']):
            return "Utilities"
        
        return "Other"
    
    def batch_process_emails(self, emails: List[Dict]) -> List[Dict]:
        """Process multiple emails in batch"""
        results = []
        for email in emails:
            try:
                result = self.process_email(email)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing email: {e}")
                results.append({"error": str(e), "email_data": email})
        return results
    
    def batch_process_sms(self, sms_list: List[Dict]) -> List[Dict]:
        """Process multiple SMS in batch"""
        results = []
        for sms in sms_list:
            try:
                result = self.process_sms(sms)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing SMS: {e}")
                results.append({"error": str(e), "sms_data": sms})
        return results

if __name__ == "__main__":
    # Test the MongoDB integration
    mongodb = MongoDBManager()
    
    # Test connection
    print("Testing MongoDB connection...")
    print(f"Database: {mongodb.database_name}")
    print(f"Collections: {mongodb.db.list_collection_names()}")
    
    # Test processor
    processor = EmailSMSProcessor(mongodb_manager=mongodb)
    
    # Test email processing
    test_email = {
        "user_id": "test_user_123",
        "email_id": "test_email_456",
        "sender": "alerts@hdfcbank.net",
        "recipient": "user@example.com",
        "subject": "UPI Transaction Alert",
        "body": "Dear Customer, Rs.500.00 has been debited from your account for UPI transaction to Zomato."
    }
    
    result = processor.process_email(test_email)
    print(f"Email processing result: {result}") 