import json
import os

import pandas as pd

with open("data/emnist_mapping.json") as f:
    emnist_mapping = json.load(f)

root = "data/emnist_data"
path = os.path.join(root, "train")

image_list = []
for path, sub_directories, files in os.walk(root):
    for name in files:
        image_list.append(os.path.join(path, name))


items = emnist_mapping.items()
rows = []
for image in image_list:
    character_key = image.split("\\")[2]
    for item in items:
        if item[0] == character_key:
            character = item[1]
            rows.append([character, image])

emnist_data = pd.DataFrame(rows, columns=["character", "image"])

emnist_data.to_csv("data/emnist_data_labels.csv", index=0)
