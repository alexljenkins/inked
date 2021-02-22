import io
from typing import Any, Dict, List, Union

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .augmentor import Augmentor
from .utils import white_to_transparent


class Character(object):
    """Utility class for the generation and combining of character images"""

    def __init__(self, text: str, image: Image.Image, metadata: Dict[str, Any] = {}):
        self.metadata = metadata
        self.text = text
        self.image = white_to_transparent(image)

    def __repr__(self) -> str:
        """Utility operator overloading to allow str() conversion of the class so prints out useful information"""
        return f"{self.__class__.__name__}: {self.text} Image: {self.image} Meta: {self.metadata}"

    def __eq__(self, other: object) -> bool:
        """Utility operator overload to check for equality of Characters"""
        if isinstance(other, Character):
            return other.text == self.text and self.image == other.image and self.metadata == other.metadata

        return False

    def __add__(self, other: Union[int, "Character", "Word"]) -> Union["Character", "Word"]:
        """Utility operator overloading which allows addition of Characters and Words - can also use sum() function"""
        # Needed for __radd__ to use inbuilt sum() function
        if other == 0:
            return self
        if not isinstance(other, (Character, Word)):
            raise TypeError("Can only add Words and Character")

        # Join two characters together with random spacing
        # NOTE: min image x,y for augments is 32x32
        new_height = max(self.image.height, other.image.height, 32)
        new_img = Image.new(
            "RGBA", (max(32, self.image.width + other.image.width), new_height), color=(255, 255, 255, 0),
        )

        new_img.paste(self.image, (0, new_height - self.image.height))
        new_img.paste(other.image, (self.image.width, new_height - other.image.height))

        self_metadata = [self.metadata] if not isinstance(self.metadata, list) else self.metadata
        other_metadata = [other.metadata] if not isinstance(other.metadata, list) else other.metadata

        return Word(self.text + other.text, new_img, self_metadata + other_metadata)

    __radd__ = __add__

    def _get_pnginfo(self) -> PngInfo:
        """This adds metadata to the PNG - can be viewed by Image.open(file).text (returns dict)"""
        pnginfo = PngInfo()
        for k, v in self.metadata.items():
            if not k == "augments":
                pnginfo.add_text(f"Char 0: {k}", str(v))

        # Add augmentation info
        for aug_key, aug_value in self.metadata.get("augments", {}).items():
            pnginfo.add_text(f"Char 0: Augmention - {aug_key}", str(aug_value))
        return pnginfo

    def save(self, path: str) -> None:
        """Saves the generated image along with associated metadata using pnginfo"""
        return self.image.save(path, pnginfo=self._get_pnginfo())

    def encode(self) -> bytes:
        """Returns the encoded image as a byte array - for use with lmdb files"""
        byteImgIO = io.BytesIO()
        self.image.save(byteImgIO, "PNG")
        byteImgIO.seek(0)
        return byteImgIO.read()

    def augment(self, augmentor: Augmentor) -> "Character":
        """Augments the character image using the augmentation settings given in the config file"""
        self.image = augmentor.augment(self)
        self.metadata["augments"] = augmentor.augments_performed
        return self


class Word(Character):
    """Utility class for the result of two or more Character being combined"""

    def __init__(self, text: str, image: Image.Image, metadata: List[Dict[str, Any]] = []):  # type: ignore
        self.metadata = metadata  # type: ignore
        self.word_metadata: Dict[str, Any] = {}
        self.text = text
        self.image = white_to_transparent(image)

    def __eq__(self, other: object) -> bool:
        """Utility operator overload to check for equality of Words"""
        if isinstance(other, Word):
            return (
                other.text == self.text
                and self.image == other.image
                and self.metadata == other.metadata
                and self.word_metadata == other.word_metadata
            )
        return False

    def _get_pnginfo(self) -> PngInfo:
        """This adds metadata to the PNG - can be viewed by Image.open(file).text (returns dict)"""
        # FIXME: Clean up - lots of duplication
        pnginfo = PngInfo()
        for char_index, char_meta in enumerate(self.metadata):
            for k, v in char_meta.items():  # type: ignore
                if not k == "augments":
                    if k == "Space":
                        pnginfo.add_text(f"Space {int((char_index-1)/2)}", str(v))
                    else:
                        pnginfo.add_text(f"Char {int((char_index))}: {k}", str(v))

            # Add augmentation info
            for aug_key, aug_value in self.word_metadata.items():
                pnginfo.add_text(f"Char {char_index}:Augmention - {aug_key}", str(aug_value))
        return pnginfo

    def augment(self, augmentor: Augmentor) -> "Word":
        """Augments the word image using the augmentation settings given in the config file"""
        self.image = augmentor.augment(self)
        self.word_metadata = augmentor.augments_performed
        return self


class RandomSpacer:
    """Utility class that creates a random width space"""

    def __init__(self, _min: int = 0, _max: int = 10) -> None:
        self._min = _min
        self._max = _max

    def __call__(self, height: int) -> Character:
        """Creates a transparent spacer image of the given height with a random width"""
        r_width = np.random.randint(self._min, self._max)
        img = Image.new("RGBA", (r_width, height), color=(255, 255, 255, 0))
        return Character("", img, {"Space": r_width})


class FixedSpacer:
    """Utility class that creates a fixed width space"""

    def __init__(self, width: int = 10):
        self.width = width

    def __call__(self, height: int) -> Character:
        """Creates a transparent spacer image of the given height with a fixed width"""
        img = Image.new("RGBA", (self.width, height), color=(255, 255, 255, 0))
        return Character("", img, {"Space": self.width})
