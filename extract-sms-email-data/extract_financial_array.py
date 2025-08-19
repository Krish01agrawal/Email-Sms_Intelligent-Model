#!/usr/bin/env python3
"""
Extract Financial SMS Array
===========================

Extracts the financial_sms array from the filtered JSON output and saves it 
as a simple array that can be directly processed by fixed_optimized_main.py
"""

import json
import argparse
from pathlib import Path

def extract_financial_array(input_file: str, output_file: str):
    """Extract financial SMS array from filtered JSON"""
    
    print(f"📱 Loading filtered data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)
    
    # Extract the financial_sms array
    if 'financial_sms' in filtered_data:
        financial_sms = filtered_data['financial_sms']
        print(f"💰 Found {len(financial_sms)} financial SMS messages")
    else:
        print("❌ No 'financial_sms' key found in the input file")
        return
    
    # Save as simple array
    print(f"💾 Saving financial SMS array to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(financial_sms, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Successfully extracted {len(financial_sms)} financial SMS to {output_file}")
    print(f"🚀 This file can now be processed by fixed_optimized_main.py")

def main():
    parser = argparse.ArgumentParser(description='Extract financial SMS array for processing')
    parser.add_argument('input_file', help='Path to filtered JSON file (from sms_financial_filter.py)')
    parser.add_argument('-o', '--output', default='financial_sms_array.json', 
                       help='Output file path (default: financial_sms_array.json)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1
    
    try:
        extract_financial_array(args.input_file, args.output)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
