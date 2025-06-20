import os
import cv2
from collections import defaultdict
import argparse

def main(img_set):
    # Validate input: must be 'train', 'val', or 'test'
    if img_set not in ['train', 'val', 'test']:
        raise ValueError("Invalid imgage set. Choose from 'train', 'val', or 'test'.")
    print(f"converting {img_set} set...")


    input_dir = f'./Dataset/masks/{img_set}'
    output_dir = f'./Dataset/labels/{img_set}'
    print(f'saving labels to: {output_dir}')

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group masks by image base ID
    grouped_masks = defaultdict(list)
    for fname in os.listdir(input_dir):
        if fname.endswith('.png'):
            image_id = fname.split('_')[0]  # adjust this if your naming differs
            grouped_masks[image_id].append(fname)

    for image_id, mask_files in grouped_masks.items():
        label_lines = []

        for mask_file in mask_files:
            mask_path = os.path.join(input_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            H, W = mask.shape

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    polygon = []
                    for point in cnt:
                        x, y = point[0]
                        polygon.append(x / W)
                        polygon.append(y / H)

                    # Format the line in YOLOv8 polygon style
                    line = '0 ' + ' '.join(f'{p:.6f}' for p in polygon)
                    label_lines.append(line)

        # Save combined label file
        if label_lines:
            with open(os.path.join(output_dir, f"{image_id}.txt"), 'w') as f:
                f.write("\n".join(label_lines) + "\n")

    print(f"Converted masks to polygons for {img_set} set.")

if __name__ == "__main__":
# Parse the --img_set argument from CLI
    parser = argparse.ArgumentParser(description="convert masks to polygons.")
    parser.add_argument(
        "--img_set",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Specify which dataset to use: 'train', 'val', or 'test'."
    )
    args = parser.parse_args()
    main(args.img_set)
