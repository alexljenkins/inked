import random

from truth.truth import AssertThat

from ..typesetter import Augmentor, CharacterGenerator, FixedSpacer, RandomSpacer, WordGenerator

random.seed(1)
LMDB_CHARACTER_SET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
FONT_CHARACTER_SET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


def test_lmdb_generator():
    factory = WordGenerator(augmentor=None, warehouses=["lmdb"])
    word_image = factory.generate(LMDB_CHARACTER_SET, augment_word=False, spacer=RandomSpacer(0))

    assert word_image.text == LMDB_CHARACTER_SET


def test_fonts_generator():
    factory = WordGenerator(augmentor=None, warehouses=["fonts"])
    word_image = factory.generate(FONT_CHARACTER_SET, augment_word=False, spacer=RandomSpacer(0))

    assert word_image.text == FONT_CHARACTER_SET


def test_either_generator():
    factory = WordGenerator(augmentor=None, warehouses=["fonts", "lmdb"])
    word_image = factory.generate(FONT_CHARACTER_SET, augment_word=False, spacer=RandomSpacer(0))

    assert word_image.text == FONT_CHARACTER_SET


def test_with_augments_and_fixed_spacing_generator():
    factory = WordGenerator(augmentor=True, warehouses=["fonts", "lmdb"])
    word_image = factory.generate(FONT_CHARACTER_SET, augment_word=True, spacer=FixedSpacer(1))

    assert word_image.text == FONT_CHARACTER_SET


def test_setting_augmentor_generator():
    factory = WordGenerator(augmentor=Augmentor(), warehouses=["fonts", "lmdb"])
    word_image = factory.generate(FONT_CHARACTER_SET, augment_word=True, spacer=FixedSpacer(1))

    assert word_image.text == FONT_CHARACTER_SET


def test_char_generator_init_error():
    with AssertThat(Exception).IsRaised():
        CharacterGenerator(warehouses=[])


def test_char_generator_index_error():
    with AssertThat(KeyError).IsRaised():
        gen = CharacterGenerator()
        gen["XX"]


def test_char_generator_with_augmentor():
    gen1 = CharacterGenerator(augmentor=Augmentor())
    gen2 = CharacterGenerator(augmentor=None)
    assert gen1["a"] != gen2["a"]


def test_word_generator_init_augment():
    gen1 = WordGenerator(augmentor=True)
    gen2 = WordGenerator(augmentor=Augmentor())
    gen3 = WordGenerator(augmentor=None)

    assert isinstance(gen1.augmentor, Augmentor)
    assert isinstance(gen2.augmentor, Augmentor)
    assert gen3.augmentor is None
