import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ..inked.scribe import Scribe, data_denormalization, plot_stroke, stable_softmax

data_path = str(Path(__file__).parent / "data/")


@pytest.mark.parametrize("text1, text2", [("All kittens are unique", "All kittens are unique"), ("Dogs", "Cats"),])
def test_scribe(text1, text2):
    scribe1 = Scribe()
    scribe2 = Scribe()
    # Assert generated results are different every time
    image1, _ = scribe1.generate_sequence(text=text1)
    image2, _ = scribe2.generate_sequence(text=text2)
    assert image1 != image2


def test_scribe_clashing_names():
    try:
        one_purpose_scribe = Scribe()
        image1, _ = one_purpose_scribe.generate_sequence(text="All kittens are unique")
        image2, _ = one_purpose_scribe.generate_sequence(text="All kittens are unique")
        assert False
    except Exception:
        assert True


def test_plot_stroke():
    list_of_arrays = np.load(os.path.join(data_path, "strokes.npy"), encoding="bytes", allow_pickle=True)
    image = plot_stroke(list_of_arrays[0])

    assert isinstance(image, Image.Image)
    assert (image.size[0], image.size[1]) != (0, 0)
    assert image.size != None


def test_data_denormalization():
    data = np.asarray([[[1.0 for col in range(2)] for col in range(2)] for row in range(3)])
    mean = 0.0
    std = 2.0

    output = data_denormalization(mean, std, data)
    output_expected = np.asarray([[[1.0, 2.0] for col in range(2)] for row in range(3)])

    assert np.array_equal(output, output_expected)


def test_stable_softmax():
    vec = np.array([-55.8673, 101.9347, -30.1526, 14.7185, 9999999999])
    X = torch.from_numpy(vec)
    output = stable_softmax(X, dim=0)
    assert np.array_equal(output.numpy(), np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
