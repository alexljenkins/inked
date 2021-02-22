import random
from pathlib import Path

import numpy as np
from PIL import Image
from truth.truth import AssertThat

from ..typesetter.augmentor import Augmentor
from ..typesetter.character import Character, FixedSpacer, RandomSpacer, Word

random.seed(1)

IMG_CHAR = np.ones((32, 32, 4), dtype=np.uint8) * 255
IMG_WORD = np.ones((32, 64, 4), dtype=np.uint8) * 255


def test_character_init():
    img = Image.fromarray(IMG_CHAR)
    char = Character("a", img, {"example": "value"})
    assert char.text == "a"
    assert char.image == img
    assert char.metadata == {"example": "value"}


def test_word_init():
    img = Image.fromarray(IMG_CHAR)
    word = Word("a", img, [{"example": "value"}])
    assert word.text == "a"
    assert word.image == img
    assert word.metadata == [{"example": "value"}]
    assert word.word_metadata == {}


def test_character_repr():
    img = Image.fromarray(IMG_CHAR)
    char = Character("a", img, {"example": "value"})
    assert str(char).startswith("Character: a Image: <PIL.Image.Image image mode=RGBA size=32x32 at ")
    assert str(char).endswith("> Meta: {'example': 'value'}")


def test_word_repr():
    img = Image.fromarray(IMG_CHAR)
    word = Word("bob", img, [{"example": "value"}])
    assert str(word).startswith("Word: bob Image: <PIL.Image.Image image mode=RGBA size=32x32 at ")
    assert str(word).endswith("> Meta: [{'example': 'value'}]")


def test_character_add_another_character():
    img_char = Image.fromarray(IMG_CHAR)
    char1 = Character("a", img_char, {"example": "value 1"})
    char2 = Character("b", img_char, {"example": "value 2"})
    combined = char1 + char2
    img_word = Image.fromarray(IMG_WORD)
    expected = Word("ab", img_word, [{"example": "value 1"}, {"example": "value 2"}])

    assert combined == sum([char1, char2])
    assert expected.text == combined.text
    assert np.equal(np.array(expected.image), np.array(combined.image)).all()
    assert expected.metadata == combined.metadata


def test_character_add_another_word():
    img_char = Image.fromarray(IMG_CHAR)
    char1 = Character("a", img_char, {"example": "value 1"})
    word2 = Word("b", img_char, [{"example": "value 2"}])
    combined = char1 + word2
    img_word = Image.fromarray(IMG_WORD)
    expected = Word("ab", img_word, [{"example": "value 1"}, {"example": "value 2"}])

    assert combined == sum([char1, word2])
    assert expected.text == combined.text
    assert np.equal(np.array(expected.image), np.array(combined.image)).all()
    assert expected.metadata == combined.metadata


def test_word_add_another_character():
    img_char = Image.fromarray(IMG_CHAR)
    word1 = Word("b", img_char, [{"example": "value 2"}])
    char2 = Character("a", img_char, {"example": "value 1"})
    combined = word1 + char2
    img_word = Image.fromarray(IMG_WORD)
    expected = Word("ba", img_word, [{"example": "value 2"}, {"example": "value 1"}])

    assert combined == sum([word1, char2])
    assert expected.text == combined.text
    assert np.equal(np.array(expected.image), np.array(combined.image)).all()
    assert expected.metadata == combined.metadata


def test_word_add_another_word():
    img_char = Image.fromarray(IMG_CHAR)
    word1 = Word("b", img_char, [{"example": "value 2"}])
    word2 = Word("a", img_char, [{"example": "value 1"}])
    combined = word1 + word2
    img_word = Image.fromarray(IMG_WORD)
    expected = Word("ba", img_word, [{"example": "value 2"}, {"example": "value 1"}])

    assert combined == sum([word1, word2])
    assert expected.text == combined.text
    assert np.equal(np.array(expected.image), np.array(combined.image)).all()
    assert expected.metadata == combined.metadata


def test_character_encode():
    img = Image.fromarray(IMG_CHAR)
    char = Character("a", img, {"example": "value"})
    assert len(char.encode()) > 0
    assert isinstance(char.encode(), bytes)


def test_character_augment():
    img = Image.fromarray(IMG_CHAR)
    char = Character("a", img, {"example": "value"})
    img_before = char.image.copy()
    aug = char.augment(Augmentor())

    assert isinstance(aug, Character)
    assert aug.metadata.get("augments") is not None
    assert isinstance(aug.metadata["augments"], dict)
    assert img_before != aug.image


def test_word_augment():
    img = Image.fromarray(IMG_WORD)
    word = Word("a", img, [{"example": "value"}])
    img_before = word.image.copy()
    aug = word.augment(Augmentor())

    assert isinstance(aug, Word)
    assert isinstance(aug.word_metadata, dict)
    assert aug.word_metadata != {}
    assert img_before != aug.image


def test_character_save_adds_in_metadata():
    img = Image.fromarray(IMG_CHAR)
    char = Character("a", img, {"example": "value"})
    tmp_name = Path("/tmp/TESTING-CHAR.png")
    char.save(tmp_name)
    re_read = Image.open(tmp_name)
    assert re_read.text == {"Char 0: example": "value"}
    tmp_name.unlink()


def test_character_save_adds_in_metadata_post_augment():
    img = Image.fromarray(IMG_CHAR)
    char = Character("a", img, {"example": "value"})
    char = char.augment(Augmentor())
    tmp_name = Path("/tmp/TESTING-CHAR.png")
    char.save(tmp_name)
    re_read = Image.open(tmp_name)
    assert re_read.text["Char 0: example"] == "value"
    assert any(["Augmention" in k for k in re_read.text.keys()])
    tmp_name.unlink()


def test_word_save_adds_in_metadata():
    img = Image.fromarray(IMG_WORD)
    word = Word("ab", img, [{"example": "value"}, {"testing": "value"}])
    tmp_name = Path("/tmp/TESTING-WORD.png")
    word.save(tmp_name)
    re_read = Image.open(tmp_name)
    assert re_read.text == {"Char 0: example": "value", "Char 1: testing": "value"}
    tmp_name.unlink()


def test_word_save_adds_in_metadata_from_word_addition():
    img = Image.fromarray(IMG_WORD)
    char1 = Character("a", img, {"example": "value"})
    char2 = Character("b", img, {"testing": "value"})
    word = char1 + char2
    tmp_name = Path("/tmp/TESTING-WORD.png")
    word.save(tmp_name)
    re_read = Image.open(tmp_name)
    assert re_read.text == {"Char 0: example": "value", "Char 1: testing": "value"}
    tmp_name.unlink()


def test_word_save_adds_in_metadata_post_augment():
    img = Image.fromarray(IMG_WORD)
    word = Word("ab", img, [{"example": "value"}, {"testing": "value"}])
    word = word.augment(Augmentor())
    tmp_name = Path("/tmp/TESTING-WORD.png")
    word.save(tmp_name)
    re_read = Image.open(tmp_name)
    assert re_read.text["Char 0: example"] == "value"
    assert re_read.text["Char 1: testing"] == "value"
    assert any(["Augmention" in k for k in re_read.text.keys()])
    tmp_name.unlink()


def test_word_save_adds_in_metadata_post_augment_addition():
    img = Image.fromarray(IMG_WORD)
    char1 = Character("a", img, {"example": "value"})
    char2 = Character("b", img, {"testing": "value"})
    word = char1 + char2
    word = word.augment(Augmentor())
    tmp_name = Path("/tmp/TESTING-WORD.png")
    word.save(tmp_name)
    re_read = Image.open(tmp_name)
    assert re_read.text["Char 0: example"] == "value"
    assert re_read.text["Char 1: testing"] == "value"
    assert any(["Augmention" in k for k in re_read.text.keys()])
    tmp_name.unlink()


def test_fixed_spacer():
    space_char = FixedSpacer(8)(4)
    assert space_char.text == ""
    assert space_char.image == Image.new("RGBA", (8, 4), color=(255, 255, 255, 0))
    assert space_char.metadata == {"Space": 8}
    assert np.array(space_char.image).shape == (4, 8, 4)


def test_random_spacer():
    space_char = RandomSpacer(1, 10)(4)
    assert space_char.text == ""
    assert isinstance(space_char.image, Image.Image)
    assert space_char.metadata.get("Space") >= 0
    assert space_char.metadata.get("Space") <= 10
    assert np.array(space_char.image).shape[2] == 4


def test_word_with_spaces_add_random():
    img = Image.fromarray(IMG_CHAR)
    char1 = Character("a", img, {"example": "value"})
    space_char = RandomSpacer(1, 10)(4)
    word = char1 + space_char
    assert isinstance(word, Word)
    assert word.image != char1.image


def test_word_with_spaces_add_metadata_random():
    img = Image.fromarray(IMG_CHAR)
    char1 = Character("a", img, {"example": "value"})
    space_char = RandomSpacer(1, 10)(4)
    word = char1 + space_char

    tmp_name = Path("/tmp/TESTING-WORD.png")
    word.save(tmp_name)
    re_read = Image.open(tmp_name)
    print(re_read.text)
    assert re_read.text["Char 0: example"] == "value"
    assert re_read.text["Space 0"].isdigit()
    tmp_name.unlink()


def test_word_with_spaces_add_fixed():
    img = Image.fromarray(IMG_CHAR)
    char1 = Character("a", img, {"example": "value"})
    space_char = FixedSpacer(8)(4)
    word = char1 + space_char
    assert isinstance(word, Word)
    assert word.image != char1.image


def test_word_with_spaces_add_metadata_fixed():
    img = Image.fromarray(IMG_CHAR)
    char1 = Character("a", img, {"example": "value"})
    space_char = FixedSpacer(8)(4)
    word = char1 + space_char

    tmp_name = Path("/tmp/TESTING-WORD.png")
    word.save(tmp_name)
    re_read = Image.open(tmp_name)
    assert re_read.text["Char 0: example"] == "value"
    assert re_read.text["Space 0"].isdigit()
    tmp_name.unlink()


def test_char_add_error():
    img = Image.fromarray(IMG_CHAR)
    char1 = Character("a", img, {"example": "value"})
    with AssertThat(TypeError).IsRaised():
        char1 + 1
    with AssertThat(TypeError).IsRaised():
        char1 + "Testing"


def test_word_add_error():
    img = Image.fromarray(IMG_WORD)
    word1 = Word("bad", img, [{"example": "value"}, {"testing": "value"}])
    with AssertThat(TypeError).IsRaised():
        word1 + 1
    with AssertThat(TypeError).IsRaised():
        word1 + "Testing"


def test_char_eq():
    img = Image.fromarray(IMG_CHAR)
    char1 = Character("a", img, {"example": "value"})
    char2 = Character("b", img, {"testing": "value"})
    char3 = Character("b", img, {"testing": "value"})
    assert (char1 == char2) == False
    assert (char1 != char2) == True
    assert (char1 == 0) == False
    assert (char1 != 0) == True
    assert (char1 == "Testing") == False
    assert (char1 != "Testing") == True

    assert (char1 == char1) == True
    assert (char1 != char1) == False

    assert (char2 == char2) == True
    assert (char2 != char2) == False

    assert (char2 == char3) == True
    assert (char2 != char3) == False


def test_word_eq():
    img = Image.fromarray(IMG_WORD)
    word1 = Word("bad", img, [{"example": "value"}, {"testing": "value"}])
    word2 = Word("good", img, [{"example": "value"}, {"testing": "value"}])
    word3 = Word("bad", img, [{"example": "value"}, {"testing": "value"}])

    assert (word1 == word2) == False
    assert (word1 != word2) == True
    assert (word1 == 0) == False
    assert (word1 != 0) == True
    assert (word1 == "Testing") == False
    assert (word1 != "Testing") == True

    assert (word1 == word1) == True
    assert (word1 != word1) == False

    assert (word2 == word2) == True
    assert (word2 != word2) == False

    assert (word1 == word3) == True
    assert (word1 != word3) == False
