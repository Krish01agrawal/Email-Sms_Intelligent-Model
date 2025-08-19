# SMS Financial Filter

A high-performance, rule-based SMS filtering system designed to extract financial SMS from large datasets while efficiently filtering out non-financial messages like OTPs, promotional content, data usage alerts, and shopping notifications.

## 🎯 Purpose

This script is specifically designed for **financial wealth building, tracking, and management products** that need to process large volumes of SMS data (5000+ messages) and extract only the financially relevant information.

## 🚀 Features

- **Efficient Rule-Based Filtering**: Fast pattern matching using regex
- **Comprehensive Financial Detection**: Covers banking, investments, credit cards, loans, payments
- **Smart Exclusion Logic**: Automatically filters out OTP, promotional, data usage, shopping SMS
- **Bank Pattern Recognition**: Identifies major Indian banks and financial institutions
- **Amount Detection**: Recognizes currency amounts and transaction values
- **Detailed Statistics**: Provides comprehensive filtering analytics
- **Flexible Input Formats**: Handles various JSON structures
- **Performance Optimized**: Designed for processing 5000+ SMS efficiently

## 📁 Files

- `sms_financial_filter.py` - Main filtering script
- `example_usage.py` - Example usage and testing
- `SMS_FILTER_README.md` - This documentation

## 🛠️ Installation

No additional dependencies required! The script uses only Python standard library:

```bash
# Ensure you have Python 3.7+
python3 --version

# Make the script executable
chmod +x sms_financial_filter.py
```

## 📊 Input Format

The script accepts JSON files with SMS data in these formats:

### Format 1: Direct SMS Array
```json
[
  {
    "id": "1",
    "sender_name": "AX-SBIUPI-S",
    "message_body": "Your A/C debited by Rs.150.00...",
    "received_date": "2025-07-24T14:07:02.449",
    "type": "received"
  }
]
```

### Format 2: Nested SMS Structure
```json
{
  "sms": [
    {
      "id": "1",
      "sender_name": "AX-SBIUPI-S",
      "message_body": "Your A/C debited by Rs.150.00...",
      "received_date": "2025-07-24T14:07:02.449",
      "type": "received"
    }
  ]
}
```

## 🎮 Usage

### Basic Usage
```bash
python3 sms_financial_filter.py user_sms.json
```

### Custom Output File
```bash
python3 sms_financial_filter.py user_sms.json -o filtered_financial.json
```

### Verbose Logging
```bash
python3 sms_financial_filter.py user_sms.json -v
```

### Help
```bash
python3 sms_financial_filter.py --help
```

## 🔍 What Gets Filtered

### ✅ Financial SMS (KEPT)
- **Bank Transactions**: Credits, debits, transfers, UPI, IMPS, NEFT, RTGS
- **Investment**: Mutual funds, stocks, FDs, insurance, dividends
- **Credit Cards**: Payments, statements, rewards, alerts
- **Loans**: EMIs, repayments, interest, penalties
- **Payments**: Bills, utilities, subscriptions, rent

### ❌ Non-Financial SMS (FILTERED OUT)
- **OTP**: Verification codes, authentication messages
- **Promotional**: Offers, discounts, sales, contests
- **Data Usage**: Internet usage alerts, bandwidth notifications
- **Shopping**: Order confirmations, delivery tracking
- **Social**: WhatsApp, social media notifications
- **System**: App updates, maintenance alerts

## 📈 Output Structure

The script generates a comprehensive JSON file with:

```json
{
  "statistics": {
    "total_sms": 5000,
    "financial_sms_count": 1250,
    "excluded_sms_count": 3750,
    "financial_percentage": 25.0,
    "exclusion_breakdown": {
      "otp": 1500,
      "promotional": 800,
      "data_usage": 400,
      "shopping": 600,
      "social": 300,
      "other": 150
    },
    "processing_timestamp": "2025-01-27T10:30:00.000000"
  },
  "financial_sms": [
    {
      "id": "1",
      "sender_name": "AX-SBIUPI-S",
      "message_body": "Your A/C debited by Rs.150.00...",
      "received_date": "2025-07-24T14:07:02.449",
      "type": "received"
    }
  ]
}
```

## 🧪 Testing

### Run Example with Sample Data
```bash
python3 example_usage.py
```

### Test with Your Data
```bash
# First, create sample data
python3 example_usage.py

# Then test the filter
python3 sms_financial_filter.py sample_user_sms.json
```

## ⚡ Performance

- **Processing Speed**: ~1000 SMS/second on standard hardware
- **Memory Usage**: Minimal memory footprint, processes SMS one-by-one
- **Scalability**: Efficiently handles 5000+ SMS datasets
- **Pattern Matching**: Optimized regex patterns for fast filtering

## 🔧 Customization

### Adding New Financial Patterns
Edit the `_initialize_financial_patterns()` method in `SMSFinancialFilter` class:

```python
"new_category": [
    r"pattern1|pattern2|pattern3",
    r"another\s*pattern"
]
```

### Adding New Exclusion Patterns
Edit the `_initialize_exclusion_patterns()` method:

```python
"new_exclusion": [
    r"pattern\s*to\s*exclude",
    r"another\s*exclusion\s*pattern"
]
```

### Adjusting Filtering Threshold
Modify the threshold in `is_financial_sms()` method:

```python
# Current threshold: financial_score >= 2
# Make it stricter: financial_score >= 3
# Make it looser: financial_score >= 1
return financial_score >= 2
```

## 📊 Expected Results

For a typical 5000 SMS dataset:
- **Total SMS**: 5000
- **Financial SMS**: 1000-1500 (20-30%)
- **Excluded SMS**: 3500-4000 (70-80%)
- **Processing Time**: 5-10 seconds
- **Output File Size**: 200-500 KB

## 🚨 Common Issues & Solutions

### Issue: "Invalid JSON format"
**Solution**: Ensure your JSON file is properly formatted and contains valid SMS data.

### Issue: "Input file not found"
**Solution**: Check the file path and ensure the file exists in the specified location.

### Issue: Low financial SMS count
**Solution**: Review your SMS data format and ensure `message_body` and `sender_name` fields are populated.

### Issue: High false positives
**Solution**: Adjust the filtering threshold or add more specific exclusion patterns.

## 🔄 Integration with LLM

After filtering, use the `filtered_financial.json` with your LLM system:

```python
# Load filtered financial SMS
with open('filtered_financial.json', 'r') as f:
    data = json.load(f)

# Extract financial SMS for LLM processing
financial_sms = data['financial_sms']

# Process with LLM for data extraction
for sms in financial_sms:
    # Send to LLM for transaction extraction
    llm_result = process_with_llm(sms)
```

## 📝 Example Workflow

1. **Prepare Data**: Export 5000 SMS to `user_sms.json`
2. **Run Filter**: `python3 sms_financial_filter.py user_sms.json`
3. **Review Results**: Check statistics and financial SMS count
4. **LLM Processing**: Use `filtered_financial.json` with your LLM system
5. **Data Extraction**: Extract transaction details, amounts, categories

## 🎉 Benefits

- **Eliminates Noise**: Removes 70-80% of irrelevant SMS
- **Improves LLM Accuracy**: Cleaner data leads to better extraction
- **Reduces Processing Time**: Focus only on financial messages
- **Cost Effective**: Lower LLM API calls for non-financial SMS
- **Scalable**: Handles large datasets efficiently
- **Maintainable**: Easy to update patterns and rules

## 🤝 Support

For questions or issues:
1. Check the example usage script
2. Review the filtering patterns
3. Test with sample data first
4. Ensure proper JSON format

---

**Happy Financial SMS Filtering! 🚀💰**
