import json
import io
from pathlib import Path
import lmdb
from PIL import Image
from tqdm import tqdm


class CharSetLMDBGen:
    def __init__(self):

        self.options = {
            "sml": {"max_chars_per_dataset": 100},
            "med": {"max_chars_per_dataset": 1000},
            "lrg": {"max_chars_per_dataset": 99999999999},
        }
        with open("combined_dict.json", "r") as f:
            self.combined = json.load(f)

    def generate_all(self):
        for size in self.options.keys():
            self.generate(size)

    def generate(self, size: str):
        assert size in self.options.keys()

        env = lmdb.open(f"/opt/working/{size}/", map_size=1_000_000_000_000)

        dataset_details = {}
        LEN = 0
        with env.begin(write=True) as txn:
            for char, paths in tqdm(self.combined.items()):
                dataset_details[char] = {}
                for path in tqdm(paths, leave=False):
                    p = Path(path)
                    if not p.exists():
                        print(f"missing data: {path}")
                        continue

                    if path.startswith("data/hasy_data_resized"):
                        dataset = "hasy"
                    elif path.startswith("data/unipen_data_resized_refined"):
                        dataset = "unipen"
                    elif path.startswith("data/emnist_data_inverted"):
                        dataset = "emnist"
                    else:
                        print(f"Unknown dataset {path}")
                        continue

                    if dataset_details[char].get(dataset) is None:
                        dataset_details[char][dataset] = {"max": 0}

                    if dataset_details[char][dataset]["max"] + 1 > self.options[size]["max_chars_per_dataset"]:
                        continue
                    dataset_details[char][dataset]["max"] += 1
                    image = Image.open(p).convert("RGBA")
                    byteImgIO = io.BytesIO()
                    image.save(byteImgIO, "PNG")
                    byteImgIO.seek(0)
                    img_bytes = byteImgIO.read()

                    txn.put(f"{char}_{dataset}_{dataset_details[char][dataset]['max']:09d}".encode(), img_bytes)
                    LEN += 1

        # {
        #     "a": {"emnist": {"max": 100}, "unipen": {"max": 100}, "hasy": {"max": 100}},
        #     "b": {"emnist": {"max": 100}, "unipen": {"max": 100}, "hasy": {"max": 100}},
        # }
        with open(f"/opt/working/{size}/details.json", "w") as f:
            json.dump(dataset_details, f, indent=2, sort_keys=True)

        print(f"Made an LMDB with {LEN} images")


if __name__ == "__main__":
    c = CharSetLMDBGen()
    for d in ["sml", "med"]:
        c.generate(d)
