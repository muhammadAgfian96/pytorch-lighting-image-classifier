from clearml import StorageManager

manager = StorageManager()
manager.upload_file(
    local_file='/workspace/README.md',
    remote_url='s3://10.8.0.66:9000/clearml-test/test-agfian/readme.md'
)
print('done')