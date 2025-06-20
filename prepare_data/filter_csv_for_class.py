import pandas as pd
import config
import os


# Define the label
object_label = config.CLASS_ID  # This should be the class ID you want to filter by
print(f"Filtering for object label: {object_label}")

# Load the annotations
train_df = pd.read_csv("./prepare_data/train-annotations-object-segmentation.csv")
val_df = pd.read_csv("./prepare_data/validation-annotations-object-segmentation.csv")
test_df = pd.read_csv('./prepare_data/test-annotations-object-segmentation.csv')


# Ensure the required files exist
for path in ["./prepare_data/train-annotations-object-segmentation.csv",
             "./prepare_data/validation-annotations-object-segmentation.csv",
             "./prepare_data/test-annotations-object-segmentation.csv"]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


# Check if the object_label is contained in the 'LabelName' column of the DataFrames
if object_label not in train_df['LabelName'].values:
    raise ValueError(f"Object label {object_label} is not found in the train DataFrame.")
if object_label not in val_df['LabelName'].values:
    raise ValueError(f"Object label {object_label} is not found in the validation DataFrame.")
if object_label not in test_df['LabelName'].values:
    raise ValueError(f"Object label {object_label} is not found in the test DataFrame.")

# Filter by object LabelName
object_df_train = train_df[train_df['LabelName'] == object_label]
object_df_val = val_df[val_df['LabelName'] == object_label]
object_df_test = test_df[test_df['LabelName'] == object_label]


# Extract unique ImageIDs
object_ids_1 = object_df_train['ImageID'].unique()
object_ids_2 = object_df_val['ImageID'].unique()
object_ids_3 = object_df_test['ImageID'].unique()


# Write them as train/ImageID format for downloader.py
with open("./prepare_data/object_images_train.txt", "w") as f:
    for image_id in object_ids_1:
        f.write(f"train/{image_id}\n")

with open("./prepare_data/object_images_val.txt", "w") as f:
    for image_id in object_ids_2:
        f.write(f"validation/{image_id}\n")

with open("./prepare_data/object_images_test.txt", "w") as f:
    for image_id in object_ids_3:
        f.write(f"test/{image_id}\n")
