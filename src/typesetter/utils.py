from pathlib import Path

import numpy as np
import requests
from PIL import Image

AVAILABLE_SIZES = ["lrg", "med", "sml"]
REQUIRED_FILES = ["data.mdb", "lock.mdb", "details.json"]


def download_charset(size: str) -> None:  # pragma: no cover
    """Downloads lmdb files and details.json for either sml, med or lrg charset images

    Args:
        size (str): One of ["lrg", "med", "sml"]
    """
    assert size in AVAILABLE_SIZES
    folder = Path.home() / ".typesetter" / size
    folder.mkdir(parents=True, exist_ok=True)
    for file in REQUIRED_FILES:
        if not (folder / file).exists():
            response = requests.get(
                f"https://cg-apac-invent-sva-docclass-data-public.s3-ap-southeast-2.amazonaws.com/typesetter/charsets/{size}/{file}"
            )
            if response.status_code == 200:
                with open(folder / file, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {file}")


def white_to_transparent(img: Image.Image) -> Image.Image:
    """Converts non RGBA images into transparent images by replacing white pixels with alpha

    NOTE: This was faster to convert to numpy than loop over pixel data in Pillow

    Args:
        img (Image.Image): Image to convert

    Returns:
        Image.Image: RGBA image
    """
    if img.mode == "RGBA":
        return img

    img = img.convert("RGBA")
    x = np.asarray(img).copy()
    x[:, :, 3] = (255 * (x[:, :, :3] != 255).any(axis=2)).astype(np.uint8)

    return Image.fromarray(x)
