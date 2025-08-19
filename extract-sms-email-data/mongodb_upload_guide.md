# SMS MongoDB Upload Guide

## 🚀 Quick Start

Upload your SMS data to MongoDB with a single command:

```bash
python3 sms_mongodb_uploader.py --input test_sms.json
```

## 📋 Prerequisites

1. **MongoDB Running**: Make sure MongoDB is running on `localhost:27017`
2. **Python Dependencies**: Install required packages:
   ```bash
   pip install pymongo
   ```

## 🎯 Usage Examples

### Basic Upload
```bash
# Upload test_sms.json to default database
python3 sms_mongodb_uploader.py --input test_sms.json
```

### Upload with Options
```bash
# Upload with custom batch size and create indexes
python3 sms_mongodb_uploader.py \
  --input test_sms.json \
  --batch-size 50 \
  --create-indexes \
  --stats
```

### Clear and Upload Fresh Data
```bash
# Clear collection and upload fresh data
python3 sms_mongodb_uploader.py \
  --input test_sms.json \
  --clear-collection \
  --create-indexes
```

### Custom Database/Collection
```bash
# Upload to custom database and collection
python3 sms_mongodb_uploader.py \
  --input test_sms.json \
  --database my_database \
  --collection my_sms_collection
```

## 📊 Default Configuration

- **Connection**: `mongodb://localhost:27017/`
- **Database**: `pluto_money`
- **Collection**: `sms_data`
- **Batch Size**: 100 documents per batch

## 🔧 Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to SMS JSON file | Required |
| `--connection` | MongoDB connection string | `mongodb://localhost:27017/` |
| `--database` | Database name | `pluto_money` |
| `--collection` | Collection name | `sms_data` |
| `--batch-size` | Documents per batch | 100 |
| `--clear-collection` | Clear collection before upload | False |
| `--create-indexes` | Create performance indexes | False |
| `--stats` | Show collection statistics | False |

## 📈 Features

### ✅ Data Validation
- Validates all SMS records before upload
- Normalizes data structure
- Adds metadata (upload timestamp, unique ID)

### 🚫 Duplicate Prevention
- Creates unique identifiers for each SMS
- Skips duplicate records automatically
- Reports duplicate count in summary

### 📊 Performance Optimization
- Batch processing for large datasets
- Optional indexes for faster queries
- Progress tracking and statistics

### 🔍 Indexes Created
When using `--create-indexes`:
- `unique_id` (unique) - Prevents duplicates
- `sender` - Fast sender-based queries
- `date` (descending) - Date-based queries
- `uploaded_at` (descending) - Upload tracking
- `type` - Type-based filtering

## 📋 Document Structure

Each SMS document in MongoDB will have:

```json
{
  "_id": "ObjectId(...)",
  "sender": "JD-SBIUPI-S",
  "body": "Dear UPI user A/C X9855 debited by 44.0...",
  "date": "2025-07-03T21:55:26.348",
  "type": "SmsMessageKind.received",
  "uploaded_at": "2025-01-27T10:30:00.000Z",
  "source_index": 1,
  "unique_id": "JD-SBIUPI-S_2025-07-03T21:55:26.348_123456789"
}
```

## 🔍 Querying Your Data

After upload, you can query your data:

```javascript
// MongoDB shell examples

// Count all SMS
db.sms_data.countDocuments()

// Find SMS by sender
db.sms_data.find({"sender": "JD-SBIUPI-S"})

// Find recent SMS (last 7 days)
db.sms_data.find({
  "uploaded_at": {
    $gte: new Date(Date.now() - 7*24*60*60*1000)
  }
})

// Find SMS containing specific text
db.sms_data.find({
  "body": {$regex: "debited", $options: "i"}
})
```

## 🚨 Troubleshooting

### MongoDB Connection Issues
```bash
# Check if MongoDB is running
brew services list | grep mongodb
# or
sudo systemctl status mongod
```

### Permission Issues
```bash
# Make sure you have write permissions
chmod +x sms_mongodb_uploader.py
```

### Large File Processing
For files with 1000+ SMS:
```bash
# Use larger batch size for better performance
python3 sms_mongodb_uploader.py \
  --input large_sms_file.json \
  --batch-size 500
```

## 📊 Expected Output

```
🔌 Connecting to MongoDB: mongodb://localhost:27017/
✅ Connected to database: pluto_money
📁 Using collection: sms_data
📁 Loading SMS data from: test_sms.json
📊 Loaded 21 SMS records
✅ Validating 21 SMS records...
✅ Validated 21 SMS documents
📤 Uploading 21 SMS records...
📦 Batch size: 100
🔄 Processing batch 1/1 (21 documents)
   ✅ Inserted: 21, Duplicates: 0, Errors: 0

📊 UPLOAD SUMMARY:
   Total Records: 21
   Valid Records: 21
   Successfully Inserted: 21
   Duplicates Skipped: 0
   Errors: 0
   Success Rate: 100.0%

✅ Upload completed successfully!
```
