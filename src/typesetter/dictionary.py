import json
import random
from pathlib import Path
from typing import Dict, List, Set

import numpy as np


class CharDict(object):
    """Utility class that allows you to draw random characters
    with a given distribution from a set of possible character lists
    """

    def __init__(
        self, distribution: Dict[str, float], possible_chars: Set[str],
    ):
        self.distribution = distribution
        self.possible_chars = possible_chars
        self.dictionary_path = Path(__file__).parent / "word_dictionaries.json"

        self.dictionary = self.__load_possible_words()

    @property
    def random_char(self) -> str:
        """Utility property to generate a random character from the list of possible characters"""
        if len(self.possible_chars) == 0:
            raise KeyError
        return random.choice(list(self.possible_chars))

    def __load_possible_words(self) -> Dict[str, Dict[str, List[str]]]:
        """Loads all of the possible words indexed by the characters they contain using the supplied dictionary"""
        with open(self.dictionary_path, "r") as f:
            full_dictionary = json.load(f)
        return {k: index_by_char(supported(v, self.possible_chars)) for k, v in full_dictionary.items()}

    def __getitem__(self, key: str) -> str:
        """Utility operator overloading to generate a word from the given character"""
        word_lists = []
        word_probs = []
        for d, prob in self.distribution.items():
            if self.dictionary[d].get(key) is not None:
                word_lists.append(d)
                word_probs.append(prob)
        if len(word_lists) == 0:
            raise KeyError(f"Character {key} is not available in any of the dictionaries")
        # Normalise
        p = np.array(word_probs)
        p /= p.sum()

        rand_dict = np.random.choice(word_lists, p=list(p))
        return random.choice(self.dictionary[rand_dict][key])


def index_by_char(words: List[str]) -> Dict[str, List[str]]:
    """Produces a lookup table to search for words that contain a given word
    Example: ["bob", "anna"] -> {"a": ["anna"], "b":["bob"], "n": ["anna"], "o": ["bob"]}
    """
    possible_chars = list(set("".join(words)))
    output: Dict[str, List[str]] = {char: [] for char in possible_chars}
    for word in words:
        for char in list(set(word)):
            output[char].append(word)
    return output


def supported(words: List[str], possible_chars: Set[str]) -> List[str]:
    """Filters out words for which we do not have the supported characters to produce"""
    all_chars = set("".join(words))
    unsupported_chars = all_chars.difference(possible_chars)
    supported_words = filter(lambda word: set(word).intersection(unsupported_chars) == set(), words)
    return list(set(supported_words))
