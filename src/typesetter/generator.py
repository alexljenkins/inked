import io
import json
import random
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Tuple, Union

import lmdb
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .augmentor import Augmentor
from .character import Character, FixedSpacer, RandomSpacer, Word
from .utils import download_charset


class CharacterUnavailableError(Exception):
    """Raised when the character requested is not a member of the possible characters to generate"""


class CharacterGenerator(object):
    """Generates fixed or random characters from font files and MNIST style character images"""

    def __init__(
        self,
        warehouses: List[str] = ["fonts"],  # TODO: replace with Warehouse class? has Font/Scribe/Setter + whitelists
        augmentor: Optional[Augmentor] = None,
        font_blacklist: List[str] = [],
        lmdb_size: str = "sml",
    ):
        assert lmdb_size in ["sml", "med", "lrg"]

        self.lmdb_size = lmdb_size
        self.augmentor = augmentor
        self.warehouses = warehouses
        self.__config_file = Path(__file__).parent / "configs" / "char_config.json"
        self.__font_whitelist = Path(__file__).parent / "configs" / "font_character_whitelist.json"
        self.font_folder = Path(__file__).parent / "data" / "fonts"
        self.__possible_fonts = {}
        self.__possible_lmdb = {}

        if "fonts" in warehouses:
            self.__possible_fonts = self._get_fonts()
        if "lmdb" in warehouses:
            self.__possible_lmdb = self._load_lmdb()

        with open(self.__config_file, "r") as f:
            self._chars_config = json.load(f)

        # list of possible characters for user to know what can be generated
        self.possible = set([*self.__possible_fonts] + list(self.__possible_lmdb.keys()))
        if self.possible == set():
            raise Exception("No data in the warehouses to choose from")

    def _load_lmdb(self):
        self._charset = Path.home() / ".typesetter" / self.lmdb_size
        download_charset(self.lmdb_size)
        self.lmdb = lmdb.open(str(self._charset))
        with open(self._charset / "details.json", "r") as f:
            return json.load(f)

    def __getitem__(self, key: str) -> Character:
        """Generates a Character either from the font files or the MNIST style images supplied

        Args:
            key (Union[Tuple[str, int], str]): Either the character to generate as a string or a tuple with the character and exact image number

        Returns:
            Character: The requested character
        """
        assert isinstance(key, str)
        # randomly choose method of image except when explicitly given
        possible_warehouses = []
        if "fonts" in self.warehouses and self.__possible_fonts.get(key):
            possible_warehouses.append("fonts")
        if "lmdb" in self.warehouses and self.__possible_lmdb.get(key):
            possible_warehouses.append("lmdb")
        if len(possible_warehouses) == 0:
            raise KeyError

        generate_from = np.random.choice(possible_warehouses)

        if generate_from == "fonts":
            random_font = np.random.choice(self.__possible_fonts[key])
            img, font = self.generate_from_font(key, font=random_font)
            metadata = {"generator": "typefont", "font": font}
        elif generate_from == "lmdb":
            rand_dataset = random.choice(list(self.__possible_lmdb[key].keys()))
            rand_char_index = random.randint(1, self.__possible_lmdb[key][rand_dataset]["max"])
            img, font = self.generate_from_lmdb(key, rand_dataset, rand_char_index)
            metadata = {"generator": "typesetter", "font": font}

        char = Character(key, image=img, metadata=metadata)
        if self.augmentor:
            char.augment(self.augmentor)

        return char

    def _get_fonts(self) -> Dict[str, List[str]]:
        """Pulls in all of the supplied fonts from the filesystem, filters out the
        characters using a whitelist and indexes them by the characters they support"""
        with open(self.__font_whitelist) as f:
            whitelist = json.load(f)

        fonts = {
            path.stem: {"glyphs": [chr(x) for x in TTFont(str(path))["cmap"].tables[-1].cmap.keys()], "path": path}
            for path in self.font_folder.glob("*.ttf")
        }

        supported: Dict[str, List[str]] = {}
        # {'a': ['Arial', ...], 'b': ['Helvetica', ...]}
        for fontname, font in fonts.items():
            for glyph in font["glyphs"]:  # type: ignore
                if glyph not in whitelist:
                    continue
                if supported.get(glyph) is None:
                    supported[glyph] = []
                supported[glyph].append(fontname)
        return supported

    def generate_from_font(
        self, glyph: str, font: Optional[Union[str, PurePath]] = None, size: int = 36
    ) -> Tuple[Image.Image, str]:
        """Generates a Image and font name for the requested glyph using font front the filesystem
        If the font is not supplied as a argument then a random font will be selected
        """
        padding = 2
        image = Image.new("RGBA", (size, size), (255, 255, 255, 0))

        font = self.font_folder / f"{np.random.choice(self.__possible_fonts[glyph])}.ttf"

        # NOTE: keeping fonts loaded into memory requires a size to be standardised
        ffont = ImageFont.truetype(str(font), size)
        drawing = ImageDraw.Draw(image)
        _, _, height, width = drawing.textbbox((0, 0), text=glyph, font=ffont, anchor="lt")

        drawing.text((0 + padding, 0 + padding), glyph, fill=(0, 0, 0, 255), font=ffont, anchor="lt")

        # glyph image out
        image = image.crop((0, 0, height + (padding * 2), width + (padding * 2)))

        return image, font.stem

    def generate_from_lmdb(self, char: str, dataset: str, num: int) -> Tuple[Image.Image, str]:
        """Generates an image from the predownloaded LMDB file for the given character, dataset and number (composed into a key)"""
        key = f"{char}_{dataset}_{num:09d}"
        with self.lmdb.begin(write=False) as txn:
            imgbuf = txn.get(key.encode())
            buf = io.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            return Image.open(buf), key


class WordGenerator(object):
    """Generates a word from with fixed or random spacing using the CharacterGenerator
    and applies any requested augmentations at both the word and character level"""

    def __init__(self, augmentor: Union[Augmentor, bool] = False, warehouses=["fonts", "lmdb"]):
        if isinstance(augmentor, Augmentor):
            self.augmentor = augmentor
        elif augmentor:
            self.augmentor = Augmentor()
        else:
            self.augmentor = None  # type: ignore

        self.chargen = CharacterGenerator(augmentor=self.augmentor, warehouses=warehouses)

    def generate(self, text: str, augment_word: bool, spacer: Union[FixedSpacer, RandomSpacer] = FixedSpacer()) -> Word:
        """Generates a Word from the requested text and spacing strategy, it will then apply any augmentations requested"""
        # Generate chars
        char_imgs = [self.chargen[char] for char in text]

        # Add in spaces
        word = []
        for i, char in enumerate(char_imgs):
            word.append(char)
            if i != len(char_imgs):
                space_char = spacer(char.image.height)
                word.append(space_char)

        # Create Word from the chars
        word_image = sum(word)
        if self.augmentor and augment_word:
            word_image = word_image.augment(self.augmentor)  # type: ignore
        return word_image  # type: ignore
