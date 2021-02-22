import random
from pathlib import Path

import lmdb

from ..typesetter import RandomSpacer, WordGenerator
from ..typesetter.lmdb_ctx import LMDBMaker

random.seed(1)


def test_lmdb():
    lmdb_path = Path("/tmp/TESTING-LMDB/")
    number_words = 1005
    factory = WordGenerator(augmentor=None, warehouses=["lmdb", "fonts"])

    # create images and save to lmdb
    with LMDBMaker(Path(lmdb_path)) as mdb:
        for _ in range(number_words):
            word_image = factory.generate("abc", augment_word=False, spacer=RandomSpacer(0))
            mdb.append(word_image.text, word_image.encode())

    # check files got created
    assert (lmdb_path / "data.mdb").exists()
    assert (lmdb_path / "lock.mdb").exists()

    # open database
    env_db = lmdb.open(str(lmdb_path))
    with env_db.begin() as txn:
        with txn.cursor() as curs:
            # check values inside
            image_count, label_count = 0, 0
            for key, value in curs:  # ergodic
                if key.decode().startswith("image"):
                    image_count += 1
                if key.decode().startswith("label"):
                    label_count += 1
                    assert value.decode() == "abc"

    env_db.close()
    assert image_count == label_count == number_words

    # remove created files
    (lmdb_path / "data.mdb").unlink()
    (lmdb_path / "lock.mdb").unlink()
    lmdb_path.rmdir()
