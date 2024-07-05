import logging
import boto3
from botocore.exceptions import ClientError
import os
import streamlit as st

aws_access_key_id = st.secrets.aws_access_key
aws_secret_access_key = st.secrets.aws_secret_key
os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
region = st.secrets.aws_region

s3_client = boto3.client('s3', region_name=region, api_version=None, use_ssl=None, verify=None, endpoint_url=None, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=None, config=None)
s3_resource = boto3.resource('s3', region_name=region, api_version=None, use_ssl=None, verify=None, endpoint_url=None, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, aws_session_token=None, config=None)

def create_bucket(bucket_name, region=None):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """

    # Create bucket
    try:
        if region is None:
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            location = {'LocationConstraint': region}
            s3_client.create_bucket(Bucket=bucket_name,
                                    CreateBucketConfiguration=location)
    except ClientError as e:
        logging.error(e)
        return False
    return True


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    try:
        response = s3_client.upload_file("./"+bucket+"/"+file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def put_object(obj, bucket, key):
    s3_client.put_object(Body=obj, Bucket=bucket, Key=key)

def create_presigned_url(bucket_name, object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL
    return response

def delete_all_objects(bucket_name):
    """
    Delete all objects in an S3 bucket.
    """
    bucket = s3_resource.Bucket(bucket_name)
    bucket.object_versions.delete()  # Deletes all versions of all objects

def delete_bucket(bucket_name):
    """
    Delete an S3 bucket after deleting all objects in it.
    """
    try:
        # Attempt to delete the bucket
        s3_client.delete_bucket(Bucket=bucket_name)
        print(f'Bucket {bucket_name} deleted successfully.')
    except Exception as e:
        print(f'Error: {e}')

def bucket_exists(bucket):
    response = s3_client.list_buckets()
    buckets = response['Buckets']
    buckets = [b['Name'] for b in buckets]
    return bucket in buckets
