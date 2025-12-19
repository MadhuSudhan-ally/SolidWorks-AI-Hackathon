import os
import pandas as pd
from PIL import Image

CLASS_MAP = {
    "bolt": 0,
    "locatingpin": 1,
    "nut": 2,
    "washer": 3
}

def convert_bbox(row, w, h):
    xc = (row.x_min + row.x_max) / 2 / w
    yc = (row.y_min + row.y_max) / 2 / h
    bw = (row.x_max - row.x_min) / w
    bh = (row.y_max - row.y_min) / h
    return xc, yc, bw, bh

def main(dataset_path, output_path):
    df = pd.read_csv(os.path.join(dataset_path, "train_bboxes.csv"))
    train_img_path = os.path.join(dataset_path, "train")

    os.makedirs(f"{output_path}/labels/train", exist_ok=True)

    for img_name in df.image_name.unique():
        img = Image.open(os.path.join(train_img_path, img_name))
        w, h = img.size

        rows = df[df.image_name == img_name]
        label_file = os.path.join(
            output_path, "labels/train", img_name.replace(".png", ".txt")
        )

        with open(label_file, "w") as f:
            for _, r in rows.iterrows():
                cls = CLASS_MAP[r["class"]]
                bbox = convert_bbox(r, w, h)
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")

if __name__ == "__main__":
    main("DATASET_PATH", "YOLO_DATASET_PATH")
