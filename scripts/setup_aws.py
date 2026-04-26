#!/usr/bin/env python3
"""
AWS Infrastructure Setup Script for PhysioEval
Creates S3 bucket and DynamoDB table.

Usage:
    python scripts/setup_aws.py

Prerequisites:
    pip install boto3
    AWS credentials configured via .env or ~/.aws/credentials
"""

import boto3
import sys
import os
from dotenv import load_dotenv

load_dotenv()

REGION      = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "physio-eval-uploads")
TABLE_NAME  = os.getenv("DYNAMODB_TABLE_NAME", "physio-eval-results")


def create_s3_bucket():
    s3 = boto3.client("s3", region_name=REGION)
    try:
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        # Block public access
        s3.put_public_access_block(
            Bucket=BUCKET_NAME,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        print(f"✅ S3 bucket created: {BUCKET_NAME}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"ℹ️  S3 bucket already exists: {BUCKET_NAME}")
    except Exception as e:
        print(f"❌ S3 error: {e}")
        sys.exit(1)


def create_dynamodb_table():
    db = boto3.client("dynamodb", region_name=REGION)
    try:
        db.create_table(
            TableName=TABLE_NAME,
            KeySchema=[
                {"AttributeName": "evaluation_id", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "evaluation_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"✅ DynamoDB table created: {TABLE_NAME}")
        print("   Waiting for table to be active…")
        waiter = boto3.client("dynamodb", region_name=REGION).get_waiter("table_exists")
        waiter.wait(TableName=TABLE_NAME)
        print("   Table is active.")
    except db.exceptions.ResourceInUseException:
        print(f"ℹ️  DynamoDB table already exists: {TABLE_NAME}")
    except Exception as e:
        print(f"❌ DynamoDB error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print(f"\n🚀 Setting up AWS infrastructure in region: {REGION}\n")
    create_s3_bucket()
    create_dynamodb_table()
    print("\n✅ AWS setup complete!\n")
