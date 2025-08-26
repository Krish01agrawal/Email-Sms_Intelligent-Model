#!/usr/bin/env python3
"""
MongoDB Aggregation Pipeline Template
Easy to configure - just change 4 values to test different aggregations!
"""

import pymongo
import json
from typing import List, Dict, Any
from datetime import datetime, timezone  # ⬅️ add this import

START = datetime(2025, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
END   = datetime(2025, 8, 1, 0, 0, 0, tzinfo=timezone.utc)

# ============================================================================
# 🔧 CONFIGURATION - CHANGE ONLY THESE 4 VALUES
# ============================================================================

# 1. MongoDB Connection URL
MONGODB_URL = "mongodb://localhost:27017/"

# 2. Database Name  
DATABASE_NAME = "pluto_money"

# 3. Collection Name
COLLECTION_NAME = "user_financial_transactions"

# 4. MongoDB Aggregation Pipeline (replace this with your pipeline)
AGGREGATION_PIPELINE = [
    {
        "$match": {
            "user_id": "ivyam",
            "transaction_date": {
                "$gte": "2025-07-27T16:44:36.305847+00:00",
                "$lt": "2025-08-26T16:44:36.305847+00:00"
            },
            "transaction_type": "debit"
        }
    },
    {
        "$set": {
            "debitAmount": {
                "$cond": [
                    {
                        "$eq": [
                            "$transaction_type",
                            "debit"
                        ]
                    },
                    {
                        "$abs": "$amount"
                    },
                    0
                ]
            }
        }
    },
    {
        "$group": {
            "_id": "$category",
            "totalSpending": {
                "$sum": "$debitAmount"
            }
        }
    },
    {
        "$group": {
            "_id": None,
            "totalBudget": {
                "$sum": "$totalSpending"
            },
            "categories": {
                "$push": {
                    "category": "$_id",
                    "total": "$totalSpending"
                }
            }
        }
    },
    {
        "$unwind": "$categories"
    },
    {
        "$set": {
            "categories.percentage": {
                "$multiply": [
                    {
                        "$divide": [
                            "$categories.total",
                            "$totalBudget"
                        ]
                    },
                    100
                ]
            }
        }
    },
    {
        "$replaceRoot": {
            "newRoot": {
                "totalBudget": "$totalBudget",
                "categories": "$categories"
            }
        }
    }
]




# ============================================================================
# 🚀 MAIN EXECUTION CODE (DON'T CHANGE BELOW THIS LINE)
# ============================================================================

class MongoDBAggregator:
    def __init__(self):
        """Initialize MongoDB connection using configuration values"""
        try:
            self.client = pymongo.MongoClient(MONGODB_URL)
            self.db = self.client[DATABASE_NAME]
            self.collection = self.db[COLLECTION_NAME]
            print(f"✅ Connected to MongoDB: {DATABASE_NAME}.{COLLECTION_NAME}")
            print(f"🔗 Connection: {MONGODB_URL}")
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise

    def run_aggregation(self) -> List[Dict[str, Any]]:
        """Run the configured aggregation pipeline"""
        try:
            print(f"🔍 Running aggregation pipeline...")
            print(f"📊 Pipeline stages: {len(AGGREGATION_PIPELINE)}")
            results = list(self.collection.aggregate(AGGREGATION_PIPELINE))
            print(f"✅ Found {len(results)} result(s)")
            return results
        except Exception as e:
            print(f"❌ Error running aggregation: {e}")
            raise

    def print_results(self, results: List[Dict[str, Any]]) -> None:
        """Print aggregation results in a formatted way"""
        if not results:
            print("📊 No results found")
            return
        
        print("\n📊 AGGREGATION RESULTS")
        print("=" * 60)
        
        # Print the raw results structure
        for i, result in enumerate(results):
            print(f"\n🔍 Result {i+1}:")
            print(json.dumps(result, indent=2, default=str))
        
        print("=" * 60)

    def export_results(self, results: List[Dict[str, Any]], filename: str = "aggregation_results_test.json") -> None:
        """Export results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str, ensure_ascii=False)
            print(f"💾 Results exported to {filename}")
        except Exception as e:
            print(f"❌ Error exporting results: {e}")

    def close_connection(self) -> None:
        """Close MongoDB connection"""
        if hasattr(self, 'client'):
            self.client.close()
            print("🔌 MongoDB connection closed")

def main():
    """Main function to run the aggregation"""
    try:
        print("🚀 MongoDB Aggregation Template")
        print("=" * 40)
        print(f"📁 Database: {DATABASE_NAME}")
        print(f"📋 Collection: {COLLECTION_NAME}")
        print(f"🔗 URL: {MONGODB_URL}")
        print("=" * 40)
        
        # Initialize and run
        aggregator = MongoDBAggregator()
        results = aggregator.run_aggregation()
        aggregator.print_results(results)
        aggregator.export_results(results)
        
    except Exception as e:
        print(f"❌ Script failed: {e}")
    finally:
        if 'aggregator' in locals():
            aggregator.close_connection()

if __name__ == "__main__":
    main()
