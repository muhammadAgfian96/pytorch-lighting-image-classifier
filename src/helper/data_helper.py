import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from minio import Minio


class MinioDatasetDownloader:
    def __init__(self, dataset, download_dir):
        self.endpoint = "10.8.0.66:9000"
        self.access_key = "bs_server_1"
        self.secret_key = "zNAYleEDeCnlzaXJsd7MvXnQhPmZehIA"
        self.bucket_name = "app-data-workflow"
        self.region = "binsho-server-2"
        self.dataset = dataset
        self.download_dir = download_dir

        # Create a Minio client with the given credentials
        self.minio_client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False,
            region=self.region,
        )

    def download_dataset(self, max_workers=10):
        # Create the download directory if it doesn't exist
        if os.path.exists(self.download_dir):
            shutil.rmtree(self.download_dir)
        os.makedirs(self.download_dir, exist_ok=True)

        # Iterate over the classes in the dataset
        for class_name, urls in self.dataset.items():
            # Create the class directory if it doesn't exist
            class_dir = os.path.join(self.download_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Use a thread pool to download each file in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for url in urls:
                    # Extract the filename from the URL
                    filename = url.split("/")[-1]

                    # Construct the object name from the URL
                    object_name = url.split(self.bucket_name + "/")[1]

                    # Download the object to the class directory
                    destination_path = os.path.join(class_dir, filename)
                    executor.submit(
                        self.minio_client.fget_object,
                        self.bucket_name,
                        object_name,
                        destination_path,
                    )


if __name__ == "__main__":
    downloader = MinioDatasetDownloader(
        # endpoint="10.8.0.66:9000",
        # access_key="bs_server_1",
        # secret_key="zNAYleEDeCnlzaXJsd7MvXnQhPmZehIA",
        # bucket_name="app-data-workflow",
        dataset={
            "Empty": [
                "s3://10.8.0.66:9000/app-data-workflow/dataset/Bousteud/val-stiched-named-revised/maturity/Empty/day4_part2_set11_flip_20220718150758_8ac9e9e4af33482d82561083d33555ff.jpg",
                "s3://10.8.0.66:9000/app-data-workflow/dataset/Bousteud/val-stiched-named-revised/maturity/Empty/day1_set5_side1_20220715143154_a418890a9e5744418841051150b60315.jpg",
            ]
        },
        download_dir="./directory",
    )

    downloader.download_dataset(max_workers=10)
