from S3ops import delete_bucket, s3_client, delete_all_objects

buckets = s3_client.list_buckets()['Buckets']
buckets = [b['Name'] for b in buckets]
print(buckets)
for bucket_name in buckets:
    try:
        delete_all_objects(bucket_name)
        delete_bucket(bucket_name)
    except:
        pass