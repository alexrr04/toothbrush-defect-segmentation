import os
import shutil
import subprocess
import sys
import zipfile


def main():
    import gdown

    zip_url = "https://drive.google.com/uc?id=1K0tyRqAduDbjZNlk2hZ_iTP7D_05lZ4_"

    data_dir = "data"
    output_dir = os.path.join(data_dir, "toothbrush_dataset")
    zip_path = os.path.join(data_dir, "dataset_temp.zip")

    if os.path.exists(output_dir):
        print(f"Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(data_dir, exist_ok=True)

    print("Downloading the zipped dataset from Google Drive...")
    gdown.download(zip_url, zip_path, quiet=False)

    if not os.path.exists(zip_path):
        print("Error: Download failed.")
        return

    print(f"Extracting files to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print("Cleaning up temporary zip file...")
    os.remove(zip_path)

    print("Data fetch complete! Your fresh dataset is ready.")


if __name__ == "__main__":
    main()
