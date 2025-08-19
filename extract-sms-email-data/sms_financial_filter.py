#!/usr/bin/env python3
"""
SMS Financial Filter
====================

Efficient rule-based SMS filtering to extract financial SMS from large datasets.
Filters out OTP, promotional, data usage, shopping, and other non-financial SMS.
"""

import json
import re
import logging
from typing import List, Dict, Any, Set
from datetime import datetime
import argparse
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SMSFinancialFilter:
    """Advanced SMS filtering for financial transactions"""
    
    def __init__(self):
        self.financial_patterns = self._initialize_financial_patterns()
        self.exclusion_patterns = self._initialize_exclusion_patterns()
        self.bank_patterns = self._initialize_bank_patterns()
        self.amount_patterns = self._initialize_amount_patterns()
        
    def _initialize_financial_patterns(self) -> Dict[str, List[str]]:
        """Financial transaction patterns for wealth management"""
        return {
            "bank_transactions": [
                r"credited|debited|deposited|withdrawn|transfer|trf|trnsfr",
                r"account|a/c|ac\s*no|acc\s*no",
                r"balance|bal|avl\s*bal|available\s*balance",
                r"upi|imps|neft|rtgs|cheque|dd|demand\s*draft",
                r"ref\s*no|reference|transaction\s*id|txn\s*id",
                r"successful|failed|completed|processed"
            ],
            "investment": [
                r"mutual\s*fund|mf|sip|systematic\s*investment",
                r"stock|equity|shares|trading|brokerage",
                r"fd|fixed\s*deposit|recurring\s*deposit|rd",
                r"insurance|premium|policy|claim|settlement",
                r"dividend|interest|maturity|redemption",
                r"purchase\s*request.*scheme|fund.*growth|fund.*direct\s*plan",
                r"nav|units.*allotted|folio|investor|quant|hdfc\s*fund|sbi\s*fund",
                r"small\s*cap\s*fund|large\s*cap\s*fund|mid\s*cap\s*fund|elss"
            ],
            "credit_cards": [
                r"credit\s*card|cc|card\s*ending|card\s*number",
                r"payment\s*due|minimum\s*amount|statement|bill",
                r"reward\s*points|cashback",
                r"transaction\s*alert|fraud\s*alert|suspicious"
            ],
            "loans": [
                r"loan|emi|equated\s*monthly\s*installment",
                r"repayment|overdue|late\s*fee|penalty",
                r"principal|interest|outstanding|due\s*amount",
                r"sanctioned|disbursed|approved|rejected"
            ],
            "payments": [
                r"payment.*(?:successful|completed|confirmed|received)",
                r"payment.*rs\.\s*\d+.*(?:awaiting|confirmation)",
                r"bill|utility|electricity|water|gas|internet",
                r"rent|subscription|renewal|expiry|expired"
            ],
            "atm_banking": [
                r"atm.*transaction|transaction.*atm",
                r"free\s*transaction|transaction.*charge|atm.*charge",
                r"non-sbi\s*atm|other\s*bank\s*atm"
            ]
        }
    
    def _initialize_exclusion_patterns(self) -> Dict[str, List[str]]:
        """Patterns to exclude non-financial SMS"""
        return {
            "otp": [
                r"otp|one\s*time\s*password|verification\s*code",
                r"\d{4,6}\s*is\s*your\s*otp|\d{4,6}\s*otp",
                r"verify.*account|verify.*zara|authenticate.*account",
                r"login\s*code|access\s*code"
            ],
            "promotional": [
                r"offer|discount|sale|deal|limited\s*time",
                r"free\s*(?:gift|voucher|coupon|delivery|shipping)",
                r"complimentary|gift|voucher|coupon",
                r"win|winner|prize|contest|lucky\s*draw",
                r"unlock|activate|claim|redeem",
                r"avail\s*(?:now|today|this|offer|discount|free)",
                r"buy\s*1\s*get\s*1|payday\s*sale|flat\s*\d+%\s*off"
            ],
            "data_usage": [
                r"data\s*usage|internet\s*usage|bandwidth",
                r"\d+%\s*data\s*used|\d+%\s*remaining",
                r"data\s*exhausted|data\s*renewal|data\s*plan"
            ],
            "shopping_non_payment": [
                r"order.*(?:placed|confirmed|shipped|being\s*processed)",
                r"delivery|tracking|shipping|courier",
                r"product|item|goods|merchandise",
                r"shopping|cart|wishlist"
            ],
            "payment_requests": [
                r"has\s*requested\s*money|money\s*request",
                r"will\s*be\s*debited.*on\s*approving",
                r"pending.*complete.*checkout"
            ],
            "social": [
                r"whatsapp|telegram|facebook|instagram|twitter",
                r"message|chat|conversation|group|channel",
                r"friend|contact|connection|follow|like"
            ],
            "system": [
                r"system\s*maintenance|server\s*update|app\s*update",
                r"backup|sync|restore|reset|password\s*reset",
                r"login|logout|session|timeout|expired"
            ],
            "government_info": [
                r"dot-goi|government|संचार\s*साथी|sancharsaathi",
                r"download.*app|एंड्राइड|आईओएस|दूरसंचार\s*विभाग"
            ]
        }
    
    def _initialize_bank_patterns(self) -> List[str]:
        """Bank and financial institution patterns"""
        return [
            r"sbi|state\s*bank|statebank",
            r"hdfc|hdfc\s*bank",
            r"icici|icici\s*bank",
            r"axis|axis\s*bank",
            r"kotak|kotak\s*bank",
            r"yes\s*bank|yesbank",
            r"idfc|idfc\s*bank",
            r"indusind|indus\s*ind",
            r"federal|federal\s*bank",
            r"canara|canara\s*bank",
            r"pnb|punjab\s*national\s*bank",
            r"boi|bank\s*of\s*india",
            r"union|union\s*bank",
            r"central|central\s*bank",
            r"paytm|phonepe|gpay|amazon\s*pay",
            r"razorpay|stripe|payu|ccavenue",
            r"quant|qntamc|hdfc\s*amc|sbi\s*mf|icici\s*pru",
            r"cult\.?fit|cultfit",
            r"atm.*sbi|sbi.*atm"
        ]
    
    def _initialize_amount_patterns(self) -> List[str]:
        """Amount and currency patterns"""
        return [
            r"rs\.?\s*\d+[,\d]*\.?\d*|₹\s*\d+[,\d]*\.?\d*",
            r"inr\s*\d+[,\d]*\.?\d*|\$\s*\d+[,\d]*\.?\d*",
            r"amount\s*rs\.?\s*\d+[,\d]*\.?\d*",
            r"debited\s*by\s*\d+[,\d]*\.?\d*",
            r"credited\s*inr\s*\d+[,\d]*\.?\d*"
        ]
    
    def is_financial_sms(self, sms_data: Dict[str, Any]) -> bool:
        """Determine if SMS is financial-related"""
        # Handle different JSON structures
        message_body = sms_data.get('message_body') or sms_data.get('body', '')
        sender = sms_data.get('sender_name') or sms_data.get('sender', '')
        
        message_body = message_body.lower()
        sender = sender.lower()
        
        # Check exclusion patterns first (faster rejection)
        for category, patterns in self.exclusion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_body, re.IGNORECASE):
                    return False
        
        # Check financial patterns
        financial_score = 0
        for category, patterns in self.financial_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_body, re.IGNORECASE):
                    financial_score += 1
        
        # Check bank patterns
        bank_found = any(re.search(pattern, message_body, re.IGNORECASE) 
                        for pattern in self.bank_patterns)
        if bank_found:
            financial_score += 2
        
        # Check amount patterns
        amount_found = any(re.search(pattern, message_body, re.IGNORECASE) 
                          for pattern in self.amount_patterns)
        if amount_found:
            financial_score += 2
        
        # Check sender patterns
        if any(re.search(pattern, sender, re.IGNORECASE) 
               for pattern in self.bank_patterns):
            financial_score += 1
        
        # Threshold for financial classification
        return financial_score >= 2
    
    def filter_sms_dataset(self, sms_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Filter SMS dataset and return financial SMS with statistics"""
        total_sms = len(sms_list)
        financial_sms = []
        excluded_categories = defaultdict(int)
        
        logger.info(f"🔍 Processing {total_sms} SMS messages...")
        
        for i, sms in enumerate(sms_list):
            if self.is_financial_sms(sms):
                sms['unique_id'] = f"sms_{i+1:06d}"  # Add unique ID (e.g., sms_000001)
                sms['isprocessed'] = False  # Mark as unprocessed
                financial_sms.append(sms)
            else:
                # Categorize exclusion reason
                message_body = (sms.get('message_body') or sms.get('body', '')).lower()
                for category, patterns in self.exclusion_patterns.items():
                    if any(re.search(pattern, message_body, re.IGNORECASE) 
                           for pattern in patterns):
                        excluded_categories[category] += 1
                        break
                else:
                    excluded_categories['other'] += 1
        
        # Generate statistics
        stats = {
            "total_sms": total_sms,
            "financial_sms_count": len(financial_sms),
            "excluded_sms_count": total_sms - len(financial_sms),
            "financial_percentage": round((len(financial_sms) / total_sms) * 100, 2),
            "exclusion_breakdown": dict(excluded_categories),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return {
            "statistics": stats,
            "financial_sms": financial_sms
        }
    
    def save_filtered_data(self, filtered_data: Dict[str, Any], output_path: str):
        """Save filtered data to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Filtered data saved to: {output_path}")
        except Exception as e:
            logger.error(f"❌ Error saving filtered data: {e}")
            raise

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Filter SMS dataset for financial messages')
    parser.add_argument('input_file', help='Path to input SMS JSON file')
    parser.add_argument('-o', '--output', default='filtered_financial.json', 
                       help='Output file path (default: filtered_financial.json)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"❌ Input file not found: {input_path}")
        return 1
    
    try:
        # Load SMS data
        logger.info(f"📱 Loading SMS data from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            sms_data = json.load(f)
        
        # Ensure it's a list
        if isinstance(sms_data, dict) and 'sms' in sms_data:
            sms_list = sms_data['sms']
        elif isinstance(sms_data, list):
            sms_list = sms_data
        else:
            logger.error("❌ Invalid JSON format. Expected list or dict with 'sms' key.")
            return 1
        
        # Initialize filter
        filter_instance = SMSFinancialFilter()
        
        # Filter SMS
        logger.info("🔍 Starting SMS filtering process...")
        filtered_data = filter_instance.filter_sms_dataset(sms_list)
        
        # Display statistics
        stats = filtered_data['statistics']
        print("\n" + "="*60)
        print("📊 SMS FILTERING RESULTS")
        print("="*60)
        print(f"📱 Total SMS: {stats['total_sms']:,}")
        print(f"💰 Financial SMS: {stats['financial_sms_count']:,} ({stats['financial_percentage']}%)")
        print(f"🚫 Excluded SMS: {stats['excluded_sms_count']:,}")
        print("\n📋 Exclusion Breakdown:")
        for category, count in stats['exclusion_breakdown'].items():
            percentage = round((count / stats['total_sms']) * 100, 2)
            print(f"   {category.title()}: {count:,} ({percentage}%)")
        print("="*60)
        
        # Save filtered data
        output_path = Path(args.output)
        filter_instance.save_filtered_data(filtered_data, output_path)
        
        logger.info("✅ SMS filtering completed successfully!")
        return 0
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON format: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
