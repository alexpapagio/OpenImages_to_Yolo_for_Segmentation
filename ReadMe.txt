
1. Find the code for the required class in oidv7-class-descriptions-boxable.csv and set it in config.py

2. run filter_csv_for_class.py, this will generate three .txt files

3. Need to run the following in CLI:
python prepare_data/downloader.py prepare_data/object_images_train.txt --download_folder Dataset/images/train --num_processes 5
python prepare_data/downloader.py prepare_data/object_images_val.txt --download_folder Dataset/images/val --num_processes 5
python prepare_data/downloader.py prepare_data/object_images_test.txt --download_folder Dataset/images/test --num_processes 5

This will use the downloader script provided by OpenImages to download the images of the selected label.
This will save the files in the corresponding folders mentioned above

4. Run the following in CLI:
python prepare_data/create_masks_from_csv.py --download_set train
python prepare_data/create_masks_from_csv.py --download_set val
python prepare_data/create_masks_from_csv.py --download_set val

This downloads the segmentation masks from Open Images that are provided for all labels as big chunks of zip files and then goes through the zip files
and extracts only the masks of the selected label and saves them in a separate masks folder (this is not the labels yet)

5. Run
python prepare_data/masks_to_polygons.py --img_set=train
python prepare_data/masks_to_polygons.py --img_set=val
python prepare_data/masks_to_polygons.py --img_set=test

to transform them in the format required from YOLO - this will actually combine multiple masks of the same image
to a sigle label file as for the training of YOLO, it expects to see one label to one image

6.
