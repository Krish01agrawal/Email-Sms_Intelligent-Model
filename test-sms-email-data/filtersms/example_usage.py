#!/usr/bin/env python3
"""
Example Usage of SMS Financial Filter
====================================

This script demonstrates how to use the SMS financial filter
with sample data and different input formats.
"""

import json
from sms_financial_filter import SMSFinancialFilter

def create_sample_sms_data():
    """Create sample SMS data for testing"""
    return [
        # Financial SMS examples
        {
            "id": "1",
            "sender_name": "AX-SBIUPI-S",
            "message_body": "Dear UPI user A/C X9855 debited by 150.0 on date 24Jul25 trf to HALLI MANE DONNE Refno 447965688556. If not u? call 1800111109. -SBI",
            "received_date": "2025-07-24T14:07:02.449",
            "type": "received"
        },
        {
            "id": "2", 
            "sender_name": "AX-CBSSBI-S",
            "message_body": "Your A/C XXXXX869855 Credited INR 11,022.00 on 29/05/25 -Deposit by transfer from Mr. DivyamVerma. Avl Bal INR 11,063.21-SBI",
            "received_date": "2025-05-29T13:44:00.623",
            "type": "received"
        },
        {
            "id": "3",
            "sender_name": "HDFC-Bank",
            "message_body": "Your HDFC Bank Credit Card ending 1234 has been charged Rs.2,500.00 on 25/07/25. Available credit limit: Rs.47,500.00",
            "received_date": "2025-07-25T10:30:00.000",
            "type": "received"
        },
        # Non-financial SMS examples
        {
            "id": "4",
            "sender_name": "OTP-Service",
            "message_body": "Your OTP is 123456. Valid for 10 minutes. Do not share with anyone.",
            "received_date": "2025-07-25T11:00:00.000",
            "type": "received"
        },
        {
            "id": "5",
            "sender_name": "Promo-Store",
            "message_body": "🎉 FLAT 50% OFF on all items! Limited time offer. Shop now at our store!",
            "received_date": "2025-07-25T12:00:00.000",
            "type": "received"
        },
        {
            "id": "6",
            "sender_name": "Data-Provider",
            "message_body": "You have used 75% of your daily data. 2.5GB remaining out of 10GB.",
            "received_date": "2025-07-25T13:00:00.000",
            "type": "received"
        },
        {
            "id": "7",
            "sender_name": "Shopping-App",
            "message_body": "Your order #ORD123456 has been shipped! Track your delivery here.",
            "received_date": "2025-07-25T14:00:00.000",
            "type": "received"
        },
        {
            "id": "8",
            "sender_name": "ICICI-Bank",
            "message_body": "Your ICICI Bank account XXXXX1234 has been credited with Rs.50,000.00 via NEFT transfer. Available balance: Rs.1,25,000.00",
            "received_date": "2025-07-25T15:00:00.000",
            "type": "received"
        }
    ]

def test_filter_with_sample_data():
    """Test the filter with sample data"""
    print("🧪 Testing SMS Financial Filter with Sample Data")
    print("=" * 60)
    
    # Create sample data
    sample_sms = create_sample_sms_data()
    
    # Initialize filter
    filter_instance = SMSFinancialFilter()
    
    # Filter SMS
    filtered_data = filter_instance.filter_sms_dataset(sample_sms)
    
    # Display results
    stats = filtered_data['statistics']
    print(f"\n📊 Results:")
    print(f"   Total SMS: {stats['total_sms']}")
    print(f"   Financial SMS: {stats['financial_sms_count']}")
    print(f"   Excluded SMS: {stats['excluded_sms_count']}")
    print(f"   Financial Percentage: {stats['financial_percentage']}%")
    
    print(f"\n📋 Exclusion Breakdown:")
    for category, count in stats['exclusion_breakdown'].items():
        print(f"   {category.title()}: {count}")
    
    print(f"\n💰 Financial SMS Details:")
    for i, sms in enumerate(filtered_data['financial_sms'], 1):
        print(f"   {i}. {sms['sender_name']}: {sms['message_body'][:80]}...")
    
    return filtered_data

def save_sample_data():
    """Save sample data to JSON file for testing"""
    sample_data = create_sample_sms_data()
    
    # Save as user_sms.json format
    with open('sample_user_sms.json', 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print("💾 Sample data saved to 'sample_user_sms.json'")
    print("   You can now test: python sms_financial_filter.py sample_user_sms.json")

if __name__ == "__main__":
    # Test with sample data
    test_filter_with_sample_data()
    
    print("\n" + "=" * 60)
    
    # Save sample data for testing
    save_sample_data()
    
    print("\n🚀 To test with your own data:")
    print("   python sms_financial_filter.py your_user_sms.json")
    print("   python sms_financial_filter.py your_user_sms.json -o my_filtered.json")
    print("   python sms_financial_filter.py your_user_sms.json -v")
