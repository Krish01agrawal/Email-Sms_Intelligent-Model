#!/usr/bin/env python3
"""
Enterprise Configuration for LifafaV0 SMS Processing System
=========================================================

Optimized for 1000+ users with 5000+ SMS each
"""

import os

class EnterpriseConfig:
    """Enterprise-grade configuration for maximum scalability"""
    
    def __init__(self):
        # Processing Configuration
        self.DEFAULT_BATCH_SIZE = 10
        self.MAX_BATCH_SIZE = 20
        self.MIN_BATCH_SIZE = 5
        self.MAX_PARALLEL_BATCHES = 5
        
        # Database Configuration
        self.MAX_POOL_SIZE = 100  # For 1000+ users
        self.MIN_POOL_SIZE = 20
        
        # Scaling Configuration
        self.MAX_CONCURRENT_USERS = 1000
        self.MAX_SMS_PER_USER = 10000
        
        # Rate Limiting
        self.INITIAL_RATE_LIMIT_DELAY = 1.0
        self.MIN_RATE_LIMIT_DELAY = 0.5
        self.MAX_RATE_LIMIT_DELAY = 10.0
    
    def get_optimal_batch_config(self, total_sms: int, user_count: int = 1) -> dict:
        """Calculate optimal batch configuration"""
        if total_sms <= 100:
            batch_size = self.MIN_BATCH_SIZE
            parallel_batches = 2
        elif total_sms <= 1000:
            batch_size = self.DEFAULT_BATCH_SIZE
            parallel_batches = 3
        else:
            batch_size = self.MAX_BATCH_SIZE
            parallel_batches = self.MAX_PARALLEL_BATCHES
        
        return {
            "batch_size": batch_size,
            "parallel_batches": parallel_batches,
            "estimated_time_minutes": (total_sms * 5.0) / (batch_size * parallel_batches) / 60
        }

# Global instance
enterprise_config = EnterpriseConfig()

if __name__ == "__main__":
    print("🏢 ENTERPRISE CONFIGURATION FOR LIFAFAV0")
    print("=" * 50)
    
    print(f"⚡ Processing Configuration:")
    print(f"   Batch Size: {enterprise_config.MIN_BATCH_SIZE}-{enterprise_config.MAX_BATCH_SIZE}")
    print(f"   Parallel Batches: {enterprise_config.MAX_PARALLEL_BATCHES}")
    print(f"   Default Batch Size: {enterprise_config.DEFAULT_BATCH_SIZE}")
    
    print(f"\n🚀 Scaling Configuration:")
    print(f"   Max Concurrent Users: {enterprise_config.MAX_CONCURRENT_USERS}")
    print(f"   Max SMS per User: {enterprise_config.MAX_SMS_PER_USER}")
    print(f"   Database Pool: {enterprise_config.MIN_POOL_SIZE}-{enterprise_config.MAX_POOL_SIZE}")
    
    print(f"\n🧪 SCALABILITY TESTING:")
    print("=" * 30)
    
    test_scenarios = [
        {"sms_count": 100, "users": 1, "description": "Small scale"},
        {"sms_count": 1000, "users": 10, "description": "Medium scale"},
        {"sms_count": 5000, "users": 100, "description": "Large scale"},
        {"sms_count": 10000, "users": 1000, "description": "Enterprise scale"}
    ]
    
    for scenario in test_scenarios:
        config = enterprise_config.get_optimal_batch_config(scenario["sms_count"], scenario["users"])
        print(f"\n📊 {scenario['description']}: {scenario['sms_count']} SMS, {scenario['users']} users")
        print(f"   Batch Size: {config['batch_size']}")
        print(f"   Parallel Batches: {config['parallel_batches']}")
        print(f"   Estimated Time: {config['estimated_time_minutes']:.1f} minutes")
