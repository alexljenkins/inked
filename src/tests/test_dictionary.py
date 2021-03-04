import string

import pytest
from truth.truth import AssertThat

from ..inked.dictionary import CharDict, index_by_char, supported


@pytest.mark.parametrize(
    "words, possible_chars, expected",
    [
        # Shouldnt work
        ([], set(), []),
        (["Dog"], set(), []),
        (["Dog", "Cat"], set(), []),
        ([], set(["Dog"]), []),
        ([], set(["D", "o", "g", "C", "a", "t"]), []),
        (["Dog"], set("dog"), []),
        (["Cat"], set("dog"), []),
        (["Dodge"], set("Dog"), []),
        # Should work
        (["Dog"], set("Dog"), ["Dog"]),
        (["Dog", "Dog"], set("Dog"), ["Dog"]),
        (["Dog", "Cat"], set("DogCat"), ["Dog", "Cat"]),
        (["Dog", "Cat"], set(string.ascii_letters), ["Dog", "Cat"]),
    ],
)
def test_supported(words, possible_chars, expected):
    assert set(supported(words, possible_chars)) == set(expected)


@pytest.mark.parametrize(
    "words, expected",
    [
        # Shouldnt work
        ([], {}),
        (["a"], {"a": ["a"]}),
        (["aaaaaaa"], {"a": ["aaaaaaa"]}),
        (["bad"], {"a": ["bad"], "b": ["bad"], "d": ["bad"]}),
        (["bad", "dad"], {"a": ["bad", "dad"], "b": ["bad"], "d": ["bad", "dad"]}),
        (["bad", "dad", "sad"], {"a": ["bad", "dad", "sad"], "b": ["bad"], "d": ["bad", "dad", "sad"], "s": ["sad"]}),
    ],
)
def test_index_by_char(words, expected):
    assert index_by_char(words) == expected


def test_CharDict_init_empty_should_be_empty_and_error_on_index_or_random():
    d = CharDict(distribution={}, possible_chars=set())
    assert d.dictionary == {}  # {"additional_words": {}, "english_words": {}, "google_words": {}}
    with AssertThat(KeyError).IsRaised():
        d["a"]
    with AssertThat(KeyError).IsRaised():
        d.random_char


def test_CharDict_indexing():
    d = CharDict(
        distribution={"english_words": 0.1, "google_words": 0.6, "additional_words": 0.3},
        possible_chars=set(string.ascii_letters),
    )
    val = d["a"]
    assert "a" in val
    assert all(x in string.ascii_letters for x in val)


def test_CharDict_loading_possible_words_valid_init():
    d = CharDict(
        distribution={"english_words": 0.1, "google_words": 0.6, "additional_words": 0.3},
        possible_chars=set(string.ascii_letters),
    )
    assert d.dictionary["additional_words"] != {}
    assert d.dictionary["english_words"] != {}
    assert d.dictionary["google_words"] != {}
    assert sorted(d.possible_chars) == sorted(list(set(string.ascii_letters)))


def test_CharDict_random_char():
    d = CharDict(
        distribution={"english_words": 0.1, "google_words": 0.6, "additional_words": 0.3},
        possible_chars=set(string.ascii_letters),
    )
    assert d.random_char != d.random_char
