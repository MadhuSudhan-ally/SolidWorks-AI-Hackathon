import pandas as pd
from collections import Counter
from ultralytics import YOLO
import os

def generate_submission(model_path, test_path, output_csv):
    model = YOLO(model_path)
    results = model(test_path, conf=0.25)

    rows = []
    for r in results:
        counts = Counter(r.boxes.cls.cpu().numpy().astype(int))
        rows.append({
            "image_name": os.path.basename(r.path),
            "bolt": counts.get(0, 0),
            "locatingpin": counts.get(1, 0),
            "nut": counts.get(2, 0),
            "washer": counts.get(3, 0)
        })

    df = pd.DataFrame(rows).sort_values("image_name")
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    generate_submission(
        "best.pt",
        "test/",
        "submission.csv"
    )
