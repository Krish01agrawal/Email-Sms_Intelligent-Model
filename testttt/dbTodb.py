#!/usr/bin/env python3
"""
MongoDB Migration Script
Migrates all collections from local MongoDB to MongoDB Atlas

Source: mongodb://localhost:27017/ (db: pluto_money)
Target: mongodb+srv://divyamverma:geMnO2HtgXwOrLsW@cluster0.gzbouvi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0 (db: pluto_money)

This script will:
1. Connect to both source and target databases
2. List all collections in the source database
3. Drop existing collections in target (overwrite mode)
4. Copy all documents from each collection
5. Create indexes if they exist
"""

import pymongo
import sys
from datetime import datetime
from bson import ObjectId
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
SOURCE_URI = "mongodb://localhost:27017/"
TARGET_URI = "mongodb+srv://divyamverma:geMnO2HtgXwOrLsW@cluster0.gzbouvi.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "pluto_money"

# SSL configuration for MongoDB Atlas
import ssl

class MongoMigrator:
    def __init__(self, source_uri, target_uri, db_name):
        self.source_uri = source_uri
        self.target_uri = target_uri
        self.db_name = db_name
        self.source_client = None
        self.target_client = None
        self.source_db = None
        self.target_db = None
        
    def connect(self):
        """Establish connections to source and target databases"""
        try:
            logger.info("Connecting to source database...")
            self.source_client = pymongo.MongoClient(self.source_uri)
            self.source_db = self.source_client[self.db_name]
            
            # Test source connection
            self.source_client.admin.command('ping')
            logger.info(f"✓ Connected to source: {self.source_uri}")
            
            logger.info("Connecting to target database...")
            # Configure TLS settings for MongoDB Atlas - using PyMongo 4.x parameters
            self.target_client = pymongo.MongoClient(
                self.target_uri,
                tls=True,
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True,
                serverSelectionTimeoutMS=30000
            )
            self.target_db = self.target_client[self.db_name]
            
            # Test target connection
            self.target_client.admin.command('ping')
            logger.info(f"✓ Connected to target: MongoDB Atlas")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def get_collections(self):
        """Get list of all collections in source database"""
        try:
            collections = self.source_db.list_collection_names()
            logger.info(f"Found {len(collections)} collections: {collections}")
            return collections
        except Exception as e:
            logger.error(f"Failed to get collections: {e}")
            return []
    
    def copy_collection(self, collection_name):
        """Copy a single collection from source to target"""
        try:
            logger.info(f"\n--- Migrating collection: {collection_name} ---")
            
            source_collection = self.source_db[collection_name]
            target_collection = self.target_db[collection_name]
            
            # Get document count
            doc_count = source_collection.count_documents({})
            logger.info(f"Documents to migrate: {doc_count}")
            
            if doc_count == 0:
                logger.info(f"Collection {collection_name} is empty, skipping...")
                return True
            
            # Drop existing collection in target (overwrite mode)
            logger.info(f"Dropping existing collection {collection_name} in target...")
            target_collection.drop()
            
            # Copy documents in batches
            batch_size = 1000
            migrated_count = 0
            
            cursor = source_collection.find({})
            batch = []
            
            for document in cursor:
                batch.append(document)
                
                if len(batch) >= batch_size:
                    target_collection.insert_many(batch)
                    migrated_count += len(batch)
                    logger.info(f"Migrated {migrated_count}/{doc_count} documents...")
                    batch = []
            
            # Insert remaining documents
            if batch:
                target_collection.insert_many(batch)
                migrated_count += len(batch)
            
            logger.info(f"✓ Successfully migrated {migrated_count} documents")
            
            # Copy indexes
            self.copy_indexes(collection_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate collection {collection_name}: {e}")
            return False
    
    def copy_indexes(self, collection_name):
        """Copy indexes from source to target collection"""
        try:
            source_collection = self.source_db[collection_name]
            target_collection = self.target_db[collection_name]
            
            # Get all indexes except the default _id index
            indexes = list(source_collection.list_indexes())
            custom_indexes = [idx for idx in indexes if idx['name'] != '_id_']
            
            if not custom_indexes:
                logger.info(f"No custom indexes found for {collection_name}")
                return
            
            logger.info(f"Copying {len(custom_indexes)} indexes...")
            
            for index in custom_indexes:
                try:
                    # Extract index specification
                    keys = list(index['key'].items())
                    options = {k: v for k, v in index.items() if k not in ['key', 'v', 'ns']}
                    
                    # Create index on target
                    target_collection.create_index(keys, **options)
                    logger.info(f"✓ Created index: {index['name']}")
                    
                except Exception as e:
                    logger.warning(f"Failed to create index {index['name']}: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to copy indexes for {collection_name}: {e}")
    
    def migrate_all(self):
        """Migrate all collections from source to target"""
        start_time = datetime.now()
        logger.info(f"=== Starting migration at {start_time} ===")
        
        if not self.connect():
            return False
        
        collections = self.get_collections()
        if not collections:
            logger.error("No collections found to migrate")
            return False
        
        success_count = 0
        failed_collections = []
        
        for collection_name in collections:
            if self.copy_collection(collection_name):
                success_count += 1
            else:
                failed_collections.append(collection_name)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"\n=== Migration Summary ===")
        logger.info(f"Total collections: {len(collections)}")
        logger.info(f"Successfully migrated: {success_count}")
        logger.info(f"Failed: {len(failed_collections)}")
        if failed_collections:
            logger.info(f"Failed collections: {failed_collections}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Completed at: {end_time}")
        
        return len(failed_collections) == 0
    
    def close_connections(self):
        """Close database connections"""
        if self.source_client:
            self.source_client.close()
        if self.target_client:
            self.target_client.close()
        logger.info("Database connections closed")

def main():
    """Main function to run the migration"""
    logger.info("MongoDB Migration Tool")
    logger.info("=" * 50)
    
    # Confirm migration
    print("\n⚠️  WARNING: This will OVERWRITE all collections in the target database!")
    print(f"Source: {SOURCE_URI} (db: {DB_NAME})")
    print(f"Target: MongoDB Atlas (db: {DB_NAME})")
    
    confirm = input("\nDo you want to continue? (yes/no): ").lower().strip()
    if confirm not in ['yes', 'y']:
        print("Migration cancelled.")
        return
    
    migrator = MongoMigrator(SOURCE_URI, TARGET_URI, DB_NAME)
    
    try:
        success = migrator.migrate_all()
        if success:
            logger.info("✅ Migration completed successfully!")
            print("\n✅ Migration completed successfully!")
        else:
            logger.error("❌ Migration completed with errors!")
            print("\n❌ Migration completed with errors! Check migration.log for details.")
            
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
        print("\n🛑 Migration interrupted by user")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        
    finally:
        migrator.close_connections()

if __name__ == "__main__":
    main()
