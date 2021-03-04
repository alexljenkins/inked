import numpy as np
import pytest
from PIL import Image

from ..inked.utils import white_to_transparent

ALL_WHITE = np.ones((3, 3, 3), dtype=np.uint8) * 255

PARTIAL_TRANSPARENT = np.array(
    [
        [[255, 255, 255, 0], [255, 255, 255, 255], [255, 255, 255, 255]],
        [[255, 255, 255, 0], [255, 255, 255, 255], [255, 255, 255, 255]],
        [[255, 255, 255, 0], [255, 255, 255, 255], [255, 255, 255, 255]],
    ],
    dtype=np.uint8,
)

ALL_TRANSPARENT = np.array(
    [
        [[255, 255, 255, 0], [255, 255, 255, 0], [255, 255, 255, 0]],
        [[255, 255, 255, 0], [255, 255, 255, 0], [255, 255, 255, 0]],
        [[255, 255, 255, 0], [255, 255, 255, 0], [255, 255, 255, 0]],
    ],
    dtype=np.uint8,
)

ALL_WHITE_GREYSCALE = np.ones((3, 3, 1), dtype=np.uint8)[0] * 255


@pytest.mark.parametrize(
    "incoming, expected",
    [
        (ALL_WHITE, ALL_TRANSPARENT),  # Test that when we load image that is all white -> all transparent
        (ALL_TRANSPARENT, ALL_TRANSPARENT),  # Test that when we load image that is all transparent -> all transparent
        (  # Test that when we load image that is partial transparent -> partial transparent
            PARTIAL_TRANSPARENT,
            PARTIAL_TRANSPARENT,
        ),
        (  # Test that when we load image that is all white greyscale -> all transparent
            ALL_WHITE_GREYSCALE,
            ALL_TRANSPARENT,
        ),
    ],
)
def test_white_to_transparent(incoming, expected):
    incoming = Image.fromarray(incoming)
    assert np.equal(np.array(white_to_transparent(incoming)), expected).all()
