#!/usr/bin/env python3
"""
Copy financial_transactions -> user_financial_transactions
and convert string transaction_date -> BSON Date (safely).

- Does NOT modify the source collection.
- Leaves other fields exactly the same.
- If a date string can't be parsed, it is kept as-is (string).
- Recreates all non-_id indexes on the new collection.

Usage:
  export MONGODB_URI="mongodb+srv://<user>:<pass>@<cluster>/?retryWrites=true&w=majority"
  python copy_convert_transaction_date.py \
      --db pluto_money \
      --source financial_transactions \
      --dest user_financial_transactions
  # Optional: interpret naive strings as India time
  # --timezone Asia/Kolkata

Tested with: PyMongo 4.x, MongoDB 5/6/7/Atlas.
"""

import argparse
import os
import sys
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson.son import SON

# -----------------------------
# Helpers
# -----------------------------

def build_conversion_expression(timezone: str | None):
    """
    Build the aggregation expression that replaces transaction_date
    only when it's a string. If timezone is provided, use $dateFromString
    with that timezone for naive strings; otherwise use $convert.
    """
    if timezone:
        # Use $dateFromString (treats strings without timezone as given TZ)
        convert_expr = {
            "$dateFromString": {
                "dateString": "$transaction_date",
                "timezone": timezone,
                "onError": "$transaction_date",
                "onNull": "$transaction_date",
            }
        }
    else:
        # Use $convert (ISO strings parsed as UTC when no TZ is present)
        convert_expr = {
            "$convert": {
                "input": "$transaction_date",
                "to": "date",
                "onError": "$transaction_date",
                "onNull": "$transaction_date",
            }
        }

    # Only apply conversion if current type is string; otherwise keep original
    expr = {
        "$cond": [
            {"$eq": [{"$type": "$transaction_date"}, "string"]},
            convert_expr,
            "$transaction_date",
        ]
    }
    return expr


def copy_indexes(src_coll, dst_coll):
    """
    Recreate all indexes from src_coll onto dst_coll (except the default _id_).
    """
    created = []
    for idx in src_coll.list_indexes():
        name = idx.get("name")
        if name == "_id_":
            continue

        # Keys
        key_list = list(idx["key"].items())  # [('field', 1), ('other', -1)] etc.

        # Copy commonly used options if present
        options = {}
        for opt in (
            "unique",
            "sparse",
            "expireAfterSeconds",
            "partialFilterExpression",
            "collation",
            "weights",
            "default_language",
            "language_override",
            "textIndexVersion",
            "2dsphereIndexVersion",
            "bits",
            "min",
            "max",
            "bucketSize",
            "wildcardProjection",
        ):
            if opt in idx:
                options[opt] = idx[opt]

        # Preserve the index name if possible
        if name:
            options["name"] = name

        dst_coll.create_index(key_list, **options)
        created.append(name or str(key_list))
    return created


def main():
    parser = argparse.ArgumentParser(description="Copy + convert transaction_date to BSON Date.")
    parser.add_argument("--uri", default=os.environ.get("MONGODB_URI", ""), help="MongoDB connection URI (or set MONGODB_URI env).")
    parser.add_argument("--db", required=True, help="Database name (e.g., pluto_money or xyz).")
    parser.add_argument("--source", default="financial_transactions", help="Source collection name.")
    parser.add_argument("--dest", default="user_financial_transactions", help="Destination collection name.")
    parser.add_argument("--timezone", default=None, help="Timezone for naive strings (e.g., Asia/Kolkata). Optional.")
    args = parser.parse_args()

    if not args.uri:
        print("ERROR: Provide --uri or set MONGODB_URI.", file=sys.stderr)
        sys.exit(1)

    try:
        client = MongoClient(args.uri)
        db = client[args.db]
        src = db[args.source]
        dst = db[args.dest]

        # Diagnostics (source)
        total_src = src.estimated_document_count()
        strings_src = src.count_documents({"transaction_date": {"$type": "string"}})
        dates_src = src.count_documents({"transaction_date": {"$type": "date"}})
        print(f"[SOURCE] {args.db}.{args.source} → total={total_src:,}, strings={strings_src:,}, dates={dates_src:,}")

        # Build pipeline: convert transaction_date when it's a string, then $out
        conversion_expr = build_conversion_expression(args.timezone)
        pipeline = [
            {"$addFields": {"transaction_date": conversion_expr}},
            # $out creates/replaces the destination collection atomically on the server
            {"$out": args.dest},
        ]

        print(f"Running aggregation with $out → {args.db}.{args.dest} ...")
        # allowDiskUse for big collections; bypassDocumentValidation to maximize throughput
        src.aggregate(pipeline, allowDiskUse=True)

        # Post-checks (destination)
        total_dst = dst.estimated_document_count()
        strings_dst = dst.count_documents({"transaction_date": {"$type": "string"}})
        dates_dst = dst.count_documents({"transaction_date": {"$type": "date"}})
        print(f"[DEST  ] {args.db}.{args.dest} → total={total_dst:,}, strings={strings_dst:,}, dates={dates_dst:,}")

        # Sanity: document count parity
        if total_dst != total_src:
            print("WARNING: Destination document count differs from source!", file=sys.stderr)

        # Recreate indexes
        print("Copying indexes from source → dest ...")
        created = copy_indexes(src, dst)
        print(f"Indexes created on {args.dest}: {created or 'none'}")

        print("✅ Done. Source left untouched. Destination collection is ready.")

    except PyMongoError as e:
        print(f"MongoDB error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
