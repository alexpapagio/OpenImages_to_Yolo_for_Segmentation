import os
import requests
import zipfile
from tqdm import tqdm
import config
import argparse


def main(download_set):
    # Validate input: must be 'train', 'val', or 'test'
    if download_set not in ['train', 'val', 'test']:
        raise ValueError("Invalid download_set. Choose from 'train', 'val', or 'test'.")
    print(f"Preparing to download masks for {download_set} set...")

    # Normalize the CLASS_ID from Open Images format (e.g., '/m/09ddx') to mask filename format ('m09ddx')
    CLASS_ID = config.CLASS_ID.replace("/", "")
    print(f"Downloading masks for class ID: {CLASS_ID}")

    # Collect the set of image IDs based on the download_set (looking at actual .jpg filenames)
    image_folder = f"./Dataset/images/{download_set}"
    your_image_ids = set([fname.split('.')[0] for fname in os.listdir(image_folder) if fname.endswith('.jpg')])

    print(f"Found {len(your_image_ids)} images in {download_set} set.")

    # Destination for extracted masks
    MASK_DIR = f"./Dataset/masks/{download_set}"
    zip_dir = "mask_zips_tmp"  # Temporary folder for downloaded zip files

    print(f"Saving masks to: {MASK_DIR}")

    # Set the correct download URL pattern based on the dataset type
    base_urls = {
        "train": "https://storage.googleapis.com/openimages/v5/train-masks/train-masks-{}.zip",
        "val": "https://storage.googleapis.com/openimages/v5/validation-masks/validation-masks-{}.zip",
        "test": "https://storage.googleapis.com/openimages/v5/test-masks/test-masks-{}.zip"
    }
    BASE_URL = base_urls[download_set]

    # Open Images mask files are grouped into ZIPs by the first hex character of the ImageID
    needed_suffixes = sorted(set([img_id[0].lower() for img_id in your_image_ids]))

    # Ensure the directories exist before we begin
    os.makedirs(zip_dir, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)
    print(f"Created directories: {zip_dir} and {MASK_DIR}")

    # Process each prefix group: download and extract relevant mask files
    for suffix in tqdm(needed_suffixes, desc="Processing required suffixes"):
        relevant_ids = [img_id for img_id in your_image_ids if img_id.startswith(suffix)]
        download_and_extract_masks(suffix, relevant_ids, download_set, CLASS_ID, zip_dir, MASK_DIR, BASE_URL)

    print("🎉 Done! You now have masks for just your object images. - you can delete the .zip files now")


def download_and_extract_masks(suffix, image_ids, download_set, CLASS_ID, zip_dir, mask_dir, BASE_URL):
    # Compose the filename of the ZIP to download, based on dataset and suffix
    if download_set == 'val':
        zip_name = f"validation_{suffix}.zip"
    elif download_set == 'train':
        zip_name = f"train_{suffix}.zip"
    elif download_set == 'test':
        zip_name = f"test_{suffix}.zip"

    zip_path = os.path.join(zip_dir, zip_name)
    url = BASE_URL.format(suffix)

    # Download the ZIP if it hasn't been downloaded yet
    if not os.path.exists(zip_path):
        print(f"⬇ Downloading {zip_name}...")
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"❌ Failed to download {url}")
            return

    # Extract only relevant mask files (by image ID and class ID)
    print(f"📦 Extracting masks for {suffix}-prefixed ImageIDs...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            image_id = file.split('_')[0]
            class_id = file.split('_')[1]

            # Check if this mask file matches our desired image and class
            if (image_id in image_ids) & (class_id == CLASS_ID):
                zip_ref.extract(file, mask_dir)
                print(f"✅ Extracted: {file}")


if __name__ == "__main__":
    # Parse the --download_set argument from CLI
    parser = argparse.ArgumentParser(description="Create masks from CSV files.")
    parser.add_argument(
        "--download_set",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Specify which dataset to use: 'train', 'val', or 'test'."
    )
    args = parser.parse_args()
    main(args.download_set)
