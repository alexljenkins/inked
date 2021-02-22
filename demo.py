import time
from pathlib import Path
import concurrent.futures

import typer
from PIL import Image
from tqdm import tqdm

from src.typesetter import WordGenerator, CharDict, RandomSpacer, FixedSpacer
from src.typesetter.lmdb_ctx import LMDBMaker


def main(n_words: int = 1000):

    factory = WordGenerator(augmentor=None, warehouses=["lmdb", "fonts"])
    char_dict = CharDict(
        distribution={"english_words": 0.1, "google_words": 0.6, "additional_words": 0.3,},
        possible_chars=factory.chargen.possible,
    )

    # LMDB
    # with LMDBMaker(Path("LMDB")) as lmdb:
    #     for _ in tqdm(range(n_words)):
    #         rand_word = char_dict[char_dict.random_char]
    #         word_image = factory.generate("abc", augment_word=True, spacer=RandomSpacer(0))
    #         lmdb.append(word_image.text, word_image.encode())

    # NORMAL
    st = time.time()
    for _ in tqdm(range(n_words)):
        try:
            r_char = char_dict.random_char
            rand_word = char_dict["a"]
            word_image = factory.generate("a", augment_word=True, spacer=RandomSpacer(0))
            word_image.save("demo.png")
        except KeyError:
            print(f"Character {r_char} is not available in any of the dictionaries")
    print(time.time() - st)


if __name__ == "__main__":
    typer.run(main)
