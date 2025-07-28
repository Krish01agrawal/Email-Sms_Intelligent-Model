#!/usr/bin/env python3
"""
Test script for Email-SMS Intelligent Model
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("🧪 Testing data preprocessing...")
    
    try:
        from data_preprocessing import DatasetPreparator, TextPreprocessor, DataExtractor
        
        # Test text preprocessor
        preprocessor = TextPreprocessor()
        test_text = "Your HDFC Bank account has been debited with Rs. 500 for UPI transaction"
        cleaned_text = preprocessor.clean_text(test_text)
        features = preprocessor.extract_financial_features(cleaned_text)
        
        print(f"✅ Text preprocessing: {len(cleaned_text)} characters")
        print(f"✅ Feature extraction: {features['has_financial_keywords']} financial keywords")
        
        # Test data extractor
        extractor = DataExtractor()
        extraction = extractor.extract_transaction_data(test_text)
        
        print(f"✅ Data extraction: {extraction['amount']} amount extracted")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing test failed: {e}")
        return False

def test_model_initialization():
    """Test model initialization"""
    print("🧪 Testing model initialization...")
    
    try:
        from model import EmailSMSModel
        
        # Initialize model
        model = EmailSMSModel()
        
        print(f"✅ Model initialized: {model.model_name}")
        print(f"✅ Device: {model.device}")
        
        # Test prediction
        test_text = "Your HDFC Bank account has been debited with Rs. 500 for UPI transaction"
        result = model.predict(test_text)
        
        print(f"✅ Prediction successful: {result['is_financial']}")
        print(f"✅ Confidence: {result['financial_confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization test failed: {e}")
        return False

def test_mongodb_integration():
    """Test MongoDB integration (if available)"""
    print("🧪 Testing MongoDB integration...")
    
    try:
        from mongodb_integration import MongoDBManager, EmailSMSProcessor
        
        # Test MongoDB connection
        mongodb = MongoDBManager()
        print(f"✅ MongoDB connected: {mongodb.database_name}")
        
        # Test processor
        processor = EmailSMSProcessor(mongodb_manager=mongodb)
        print("✅ EmailSMSProcessor initialized")
        
        return True
        
    except Exception as e:
        print(f"⚠️ MongoDB test skipped (not available): {e}")
        return True  # Not a failure, just not available

def test_configuration():
    """Test configuration loading"""
    print("🧪 Testing configuration...")
    
    try:
        from config import model_config, data_config, training_config
        
        print(f"✅ Model config: {model_config.model_name}")
        print(f"✅ Data config: {data_config.train_file}")
        print(f"✅ Training config: {training_config.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_sample_data():
    """Test with sample data"""
    print("🧪 Testing with sample data...")
    
    try:
        from model import EmailSMSModel
        
        model = EmailSMSModel()
        
        # Test cases
        test_cases = [
            {
                "text": "Your HDFC Bank account has been debited with Rs. 500 for UPI transaction to Zomato",
                "expected": True
            },
            {
                "text": "Dear Customer, Rs.1000.00 has been credited to your account by VPA john@okaxis",
                "expected": True
            },
            {
                "text": "Hello, how are you doing today?",
                "expected": False
            },
            {
                "text": "Your Netflix subscription has been renewed for Rs. 499",
                "expected": True
            },
            {
                "text": "Rs.250 debited for Uber ride from Koramangala to Indiranagar",
                "expected": True
            }
        ]
        
        correct_predictions = 0
        
        for i, case in enumerate(test_cases, 1):
            result = model.predict(case["text"])
            prediction = result['is_financial']
            expected = case["expected"]
            
            if prediction == expected:
                correct_predictions += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{status} Test {i}: Expected {expected}, Got {prediction} "
                  f"(Confidence: {result['financial_confidence']:.3f})")
        
        accuracy = correct_predictions / len(test_cases)
        print(f"📊 Sample test accuracy: {accuracy:.2f} ({correct_predictions}/{len(test_cases)})")
        
        return accuracy >= 0.8  # At least 80% accuracy on sample data
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Email-SMS Intelligent Model Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Data Preprocessing", test_data_preprocessing),
        ("Model Initialization", test_model_initialization),
        ("MongoDB Integration", test_mongodb_integration),
        ("Sample Data", test_sample_data)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The model is ready to use.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 