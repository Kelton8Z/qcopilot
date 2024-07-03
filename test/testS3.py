import boto3
import streamlit as st
s3_resource = boto3.resource('s3')

region=None
aws_access_key_id = st.secrets.aws_access_key
aws_secret_access_key = st.secrets.aws_secret_key
bucket_name = "b"

try:
    s3_client = boto3.client('s3', region_name=region, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    if region is None:
        s3_client.create_bucket(Bucket=bucket_name)
    else:
        location = {'LocationConstraint': region}
        s3_client.create_bucket(Bucket=bucket_name,
                                CreateBucketConfiguration=location)
except Exception as e:
    print(e)