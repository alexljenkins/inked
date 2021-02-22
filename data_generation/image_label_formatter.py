import pandas as pd
from PIL import Image
import PIL.ImageOps

# Update image path in hasy_data_labels.csv
hasy_data = pd.read_csv("data/hasy_data_labels.csv", encoding="utf-8")
new_paths = []
for image in hasy_data["image"]:
    image_name = image.split("/")[2]
    image_path = "data/hasy_data_resized/" + image_name
    new_paths.append(image_path)

hasy_data["new_image"] = new_paths
hasy_data.head()

hasy_data.to_csv("data/hasy_data_inverted_labels.csv", index=0)

# Update image path in emnist_data_labels.csv
emnist_data = pd.read_csv("data/emnist_data_labels.csv", encoding="utf-8")
new_paths = []
for image in emnist_data["image"]:
    image_name = image.split("\\")[3]
    image_path = "data/emnist_data_inverted/" + image_name
    new_paths.append(image_path)

emnist_data["new_image"] = new_paths
emnist_data.head()

emnist_data.to_csv("data/emnist_data_inverted_labels.csv", index=0)


# Resize images in hasy dataset to 28x28 from 32x32
hasy_data = pd.read_csv("./data/hasy_data_labels.csv", encoding="utf-8")

hasy_images = hasy_data["image"]

for image in hasy_images:
    PIL_image = Image.open(image)
    resized_image = PIL_image.resize((28, 28))
    image_name = image.split("/")[2]
    resized_image.save(f"data/hasy_data_resized/{image_name}")


# Invert images in EMNIST dataset
emnist_data = pd.read_csv("data/emnist_data_labels.csv", encoding="utf-8")

emnist_images = emnist_data["image"]

for image in emnist_images:
    PIL_image = Image.open(image)
    inverted_image = PIL.ImageOps.invert(PIL_image)
    image_name = image.split("\\")[3]
    inverted_image.save(f"data/emnist_data_inverted/{image_name}")

# NOTE: This only inverted 697,932 images out of 814,255
