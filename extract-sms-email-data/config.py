#!/usr/bin/env python3
"""
Configuration Management for LifafaV0 SMS Processing System
==========================================================

Centralized configuration for all system parameters including:
- API settings
- Rate limiting
- Database configuration
- Processing parameters
- Error handling
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Centralized configuration management"""
    
    def __init__(self):
        self.api_config = self._load_api_config()
        self.database_config = self._load_database_config()
        self.processing_config = self._load_processing_config()
        self.rate_limiting_config = self._load_rate_limiting_config()
        self.error_handling_config = self._load_error_handling_config()
    
    def _load_api_config(self) -> Dict[str, Any]:
        """Load API configuration"""
        return {
            "url": os.getenv("API_URL", ""),
            "key": os.getenv("API_KEY", ""),
            "model": os.getenv("DEFAULT_MODEL", "qwen3:8b"),
            "temperature": float(os.getenv("API_TEMPERATURE", "0.1")),
            "max_tokens": int(os.getenv("API_MAX_TOKENS", "4096")),
            "top_p": float(os.getenv("API_TOP_P", "0.9")),
            "timeout": {
                "total": int(os.getenv("API_TIMEOUT_TOTAL", "60")),
                "connect": int(os.getenv("API_TIMEOUT_CONNECT", "20")),
                "read": int(os.getenv("API_TIMEOUT_READ", "40"))
            },
            "retry_attempts": int(os.getenv("API_RETRY_ATTEMPTS", "3")),
            "backoff_factor": float(os.getenv("API_BACKOFF_FACTOR", "2.0"))
        }
    
    def _load_database_config(self) -> Dict[str, Any]:
        """Load database configuration"""
        return {
            "uri": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
            "database": os.getenv("MONGODB_DB", "pluto_money"),
            "connection_pool": {
                "max_pool_size": int(os.getenv("MONGODB_MAX_POOL_SIZE", "50")),
                "min_pool_size": int(os.getenv("MONGODB_MIN_POOL_SIZE", "10")),
                "max_idle_time_ms": int(os.getenv("MONGODB_MAX_IDLE_TIME_MS", "30000")),
                "wait_queue_timeout_ms": int(os.getenv("MONGODB_WAIT_QUEUE_TIMEOUT_MS", "5000")),
                "server_selection_timeout_ms": int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000")),
                "connect_timeout_ms": int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "10000")),
                "socket_timeout_ms": int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", "30000")),
                "heartbeat_frequency_ms": int(os.getenv("MONGODB_HEARTBEAT_FREQUENCY_MS", "10000"))
            },
            "collections": {
                "sms_data": "sms_data",
                "financial_raw": "sms_fin_rawdata",
                "transactions": "financial_transactions"
            }
        }
    
    def _load_processing_config(self) -> Dict[str, Any]:
        """Load processing configuration"""
        return {
            "batch_size": int(os.getenv("BATCH_SIZE", "2")),
            "max_parallel_batches": int(os.getenv("MAX_PARALLEL_BATCHES", "1")),
            "enrich_mode": os.getenv("ENRICH_MODE", "safe"),
            "resume_processing": os.getenv("RESUME_PROCESSING", "true").lower() == "true",
            "real_time_persistence": os.getenv("REAL_TIME_PERSISTENCE", "true").lower() == "true",
            "adaptive_batching": os.getenv("ADAPTIVE_BATCHING", "true").lower() == "true",
            "parallel_processing": os.getenv("PARALLEL_PROCESSING", "true").lower() == "true"
        }
    
    def _load_rate_limiting_config(self) -> Dict[str, Any]:
        """Load rate limiting configuration"""
        return {
            "min_delay": float(os.getenv("RATE_LIMIT_MIN_DELAY", "0.5")),
            "max_delay": float(os.getenv("RATE_LIMIT_MAX_DELAY", "10.0")),
            "initial_delay": float(os.getenv("RATE_LIMIT_INITIAL_DELAY", "1.0")),
            "target_response_time": float(os.getenv("RATE_LIMIT_TARGET_RESPONSE_TIME", "5.0")),
            "performance_window": int(os.getenv("RATE_LIMIT_PERFORMANCE_WINDOW", "10")),
            "success_threshold": float(os.getenv("RATE_LIMIT_SUCCESS_THRESHOLD", "0.9")),
            "delay_adjustment_factor": float(os.getenv("RATE_LIMIT_ADJUSTMENT_FACTOR", "1.2"))
        }
    
    def _load_error_handling_config(self) -> Dict[str, Any]:
        """Load error handling configuration"""
        return {
            "max_retries": int(os.getenv("ERROR_MAX_RETRIES", "3")),
            "backoff_factor": float(os.getenv("ERROR_BACKOFF_FACTOR", "2.0")),
            "max_retry_delay": float(os.getenv("ERROR_MAX_RETRY_DELAY", "60.0")),
            "permanent_errors": [
                "validation_error",
                "parsing_error", 
                "missing_essential_fields"
            ],
            "save_retry_queue": os.getenv("ERROR_SAVE_RETRY_QUEUE", "true").lower() == "true",
            "save_dead_letter_queue": os.getenv("ERROR_SAVE_DEAD_LETTER_QUEUE", "true").lower() == "true"
        }
    
    def validate(self) -> bool:
        """Validate all configuration parameters"""
        errors = []
        
        # Validate API configuration
        if not self.api_config["url"]:
            errors.append("API_URL is required")
        
        if self.api_config["temperature"] < 0 or self.api_config["temperature"] > 2:
            errors.append("API temperature must be between 0 and 2")
        
        if self.api_config["max_tokens"] < 1 or self.api_config["max_tokens"] > 8192:
            errors.append("API max_tokens must be between 1 and 8192")
        
        # Validate database configuration
        if not self.database_config["uri"]:
            errors.append("MONGODB_URI is required")
        
        # Validate processing configuration
        if self.processing_config["batch_size"] < 1:
            errors.append("Batch size must be at least 1")
        
        if self.processing_config["max_parallel_batches"] < 1:
            errors.append("Max parallel batches must be at least 1")
        
        # Validate rate limiting configuration
        if self.rate_limiting_config["min_delay"] < 0:
            errors.append("Min delay must be non-negative")
        
        if self.rate_limiting_config["max_delay"] < self.rate_limiting_config["min_delay"]:
            errors.append("Max delay must be greater than min delay")
        
        # Validate error handling configuration
        if self.error_handling_config["max_retries"] < 0:
            errors.append("Max retries must be non-negative")
        
        if self.error_handling_config["backoff_factor"] < 1:
            errors.append("Backoff factor must be at least 1")
        
        if errors:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        print("✅ Configuration validation passed")
        return True
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 SYSTEM CONFIGURATION:")
        print(f"   API: {self.api_config['model']} at {self.api_config['url']}")
        print(f"   Database: {self.database_config['database']} at {self.database_config['uri']}")
        print(f"   Processing: {self.processing_config['batch_size']} SMS per batch, {self.processing_config['max_parallel_batches']} parallel")
        print(f"   Rate Limiting: {self.rate_limiting_config['initial_delay']:.1f}s delay, adaptive")
        print(f"   Error Handling: {self.error_handling_config['max_retries']} retries with exponential backoff")
        
        # Show enterprise capabilities
        enterprise = self.get_enterprise_capabilities()
        print(f"\n🚀 ENTERPRISE SCALING CAPABILITIES:")
        print(f"   Max Concurrent Users: {enterprise['max_concurrent_users']}")
        print(f"   Max SMS per User: {enterprise['max_sms_per_user']}")
        print(f"   Max Batch Size: {enterprise['max_batch_size']}")
        print(f"   Database Pool: {enterprise['database_pool']['min_size']}-{enterprise['database_pool']['max_size']}")
        
        # Show scalability examples
        print(f"\n📊 SCALABILITY EXAMPLES:")
        test_scenarios = [100, 1000, 5000, 10000]
        for sms_count in test_scenarios:
            config = self.get_optimal_batch_config(sms_count)
            print(f"   {sms_count} SMS: {config['batch_size']} SMS/batch, {config['parallel_batches']} parallel, ~{config['estimated_time_minutes']:.1f} min")
    
    def get_optimal_batch_config(self, total_sms: int, user_count: int = 1) -> dict:
        """Calculate optimal batch configuration for enterprise scaling"""
        batch_size = self.processing_config["batch_size"]
        max_batch_size = self.processing_config.get("max_batch_size", 20)
        min_batch_size = self.processing_config.get("min_batch_size", 5)
        max_parallel = self.processing_config["max_parallel_batches"]
        
        # Calculate optimal batch size based on scale
        if total_sms <= 100:
            optimal_batch_size = min_batch_size
            parallel_batches = 2
        elif total_sms <= 1000:
            optimal_batch_size = batch_size
            parallel_batches = 3
        else:
            optimal_batch_size = min(max_batch_size, batch_size * 2)
            parallel_batches = max_parallel
        
        return {
            "batch_size": optimal_batch_size,
            "parallel_batches": parallel_batches,
            "estimated_time_minutes": (total_sms * 5.0) / (optimal_batch_size * parallel_batches) / 60
        }
    
    def get_enterprise_capabilities(self) -> dict:
        """Get enterprise scaling capabilities"""
        return {
            "max_concurrent_users": self.processing_config.get("max_concurrent_users", 1000),
            "max_sms_per_user": self.processing_config.get("max_sms_per_user", 10000),
            "max_batch_size": self.processing_config.get("max_batch_size", 20),
            "max_parallel_batches": self.processing_config["max_parallel_batches"],
            "database_pool": {
                "max_size": self.database_config["connection_pool"]["max_pool_size"],
                "min_size": self.database_config["connection_pool"]["min_pool_size"]
            }
        }

# Global configuration instance
config = Config()

# Validate configuration on import
if not config.validate():
    raise ValueError("Invalid configuration detected")

if __name__ == "__main__":
    config.print_config()
