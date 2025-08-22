user_sms.json    -->      user_financial.json

(venv) krishagrawal@Mac test-sms-email-data % python3 sms_financial_filter.py sms_data.json -o financial_data_div.json



user_financial.json     -->         structured_user_finData.json

python3 main.py --input test_sms.json --output test_pretty_output.ndjson --failures failures.ndjson --enrich off




batch_processing sms to llm

python3 main.py --input test_sms.json --output optimized_test_result.json --model "qwen3:8b" --mode openai --batch-size 1 --parallel-batches 1 --temperature 0.1 --failures optimized_failures.ndjson --enrich safe












# Step 1: Filter (from your 5,072 SMS)(raw sms/email)
python3 sms_financial_filter.py sms_data.json -o filtered_financial.json

# Step 2: Extract array (extract fin_data json)
python3 extract_financial_array.py filtered_financial.json -o financial_array.json  

# Step 3: Process with AI (Data to information -> stored in mongodb)
python3 main.py --input financial_array.json --output final_results.json --failures complete_failures.ndjson



python3 main.py --input test_realtime.json --output test_results_realtime.json --failures test_failures_realtime.ndjson --batch-size 1 --parallel-batches 1


















# 1. First, upload fresh SMS data
python3 sms_mongodb_uploader.py --input test_sms.json

python3 sms_mongodb_uploader.py --input "assets/sms_data/sms_data_divyam.json" --batch-size 100 --create-indexes --stats




# 2. Run complete pipeline
python3 mongodb_pipeline.py --limit 100 --batch-size 10

python3 mongodb_pipeline.py --user-id "divyam_user" --limit 50 --batch-size 2



# 3. Verify results
python3 -c "from mongodb_operations import MongoDBOperations; mongo = MongoDBOperations(); stats = mongo.get_processing_stats(); print('📊 Final Stats:', stats); mongo.close_connection()"