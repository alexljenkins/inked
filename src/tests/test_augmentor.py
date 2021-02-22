import random
from pathlib import Path

import cv2
from truth.truth import AssertThat

from ..typesetter import Augmentor
from ..typesetter.augmentor import (
    BackgroundImagesException,
    LoadingConfigInputException,
    _check_word_and_char_config,
    _list_background_images_from_file,
    _load_config,
)

random.seed(1)

IMG = cv2.imread(str(Path(__file__).parent / "data" / "demo.png"))


def test_augment_error():
    augmentor = Augmentor()
    with AssertThat(TypeError).IsRaised():
        augmentor.augment("alex")


def test__get_random_severity():
    augmentor = Augmentor()
    assert isinstance(augmentor._get_random_severity(1, 5), int)
    assert isinstance(augmentor._get_random_severity(1.0, 5.0), float)
    with AssertThat(TypeError).IsRaised():
        augmentor._get_random_severity(1.0, 5)


def test__add_shear():
    augmentor = Augmentor()
    augmentor.image = IMG.copy()
    augmentor._add_shear(severity=0.1)

    assert augmentor.image.all() == cv2.imread(str(Path(__file__).parent / "data" / "shear.png")).all()


def test__add_scale_x():
    augmentor = Augmentor()
    augmentor.image = IMG.copy()
    augmentor._add_scale_x(severity=2.0)

    assert augmentor.image.shape == (IMG.shape[0], IMG.shape[1] * 2, IMG.shape[2])
    assert augmentor.image.all() == cv2.imread(str(Path(__file__).parent / "data" / "scale_x.png")).all()


def test__add_scale_y():
    augmentor = Augmentor()
    augmentor.image = IMG.copy()
    augmentor._add_scale_y(severity=0.5)

    assert augmentor.image.shape == (IMG.shape[0] * 0.5, IMG.shape[1], IMG.shape[2])
    assert augmentor.image.all() == cv2.imread(str(Path(__file__).parent / "data" / "scale_x.png")).all()


def test__add_xblur():
    augmentor = Augmentor()
    augmentor.image = IMG.copy()
    augmentor._add_xblur(severity=2)
    # cv2.imwrite("xblur.png", augmentor.image)
    assert augmentor.image.shape == IMG.shape
    assert augmentor.image.all() == cv2.imread(str(Path(__file__).parent / "data" / "xblur.png")).all()


def test__add_yblur():
    augmentor = Augmentor()
    augmentor.image = IMG.copy()
    augmentor._add_yblur(severity=2)

    assert augmentor.image.shape == IMG.shape
    assert augmentor.image.all() == cv2.imread(str(Path(__file__).parent / "data" / "yblur.png")).all()


def test__add_random_pixel_noise():
    augmentor = Augmentor()
    augmentor.image = IMG.copy()
    augmentor._add_random_pixel_noise(severity=0.2)

    assert augmentor.image.shape == IMG.shape
    assert augmentor.image.all() == cv2.imread(str(Path(__file__).parent / "data" / "random_pixel_noise.png")).all()


def test__add_bg_colour():
    augmentor = Augmentor()
    augmentor.image = IMG.copy()
    augmentor._add_bg_colour(severity=0.2)

    assert augmentor.image.shape == IMG.shape
    assert augmentor.image.all() == cv2.imread(str(Path(__file__).parent / "data" / "bg_colour.png")).all()


def test__add_bg_image():
    augmentor = Augmentor()
    augmentor.image = cv2.cvtColor(IMG, cv2.COLOR_RGB2RGBA)
    augmentor._add_bg_image(severity=0.0, background=cv2.imread(str(Path(__file__).parent / "data" / "bg.png")))
    # cv2.imwrite("bg_img.png", augmentor.image)
    assert augmentor.image.shape[:2] == IMG.shape[:2]
    assert augmentor.image.all() == cv2.imread(str(Path(__file__).parent / "data" / "bg_out.png")).all()


def test__load_config():
    loaded_config = _load_config(Path(__file__).parent.parent / "typesetter" / "configs" / "augment_config.json")

    assert len(loaded_config) == 2
    assert list(loaded_config[0].keys()) == ["object", "min_augments", "max_augments", "augments"]
    assert loaded_config[0]["object"] == "Character"
    assert list(loaded_config[1].keys()) == ["object", "min_augments", "max_augments", "augments"]
    assert loaded_config[1]["object"] == "Word"


def test__load_config_exception():
    with AssertThat(FileNotFoundError).IsRaised():
        _load_config("random string")

    with AssertThat(LoadingConfigInputException).IsRaised():
        _load_config(1234)


def test__list_background_images_from_file():
    config = _load_config(Path(__file__).parent.parent / "typesetter" / "configs" / "augment_config.json")

    bg_imgs = _list_background_images_from_file(config[1])
    assert len(bg_imgs) > 0


def test__list_background_images_from_file_else():
    config = _load_config(Path(__file__).parent.parent / "typesetter" / "configs" / "augment_config.json")
    tmp = config[1]
    for aug in tmp["augments"]:
        if aug["name"] == "bg_image":
            aug["enabled"] = False
    bg_imgs = _list_background_images_from_file(tmp)
    assert len(bg_imgs) == 13


def test__list_background_images_from_file_else_error():
    config = _load_config(Path(__file__).parent.parent / "typesetter" / "configs" / "augment_config.json")
    for aug in config[1]["augments"]:
        if aug["name"] == "bg_image":
            aug["folder"] = "random_folder"
    with AssertThat(BackgroundImagesException).IsRaised():
        _list_background_images_from_file(config[1])


def test__list_background_images_from_file_none():
    config = _load_config(Path(__file__).parent.parent / "typesetter" / "configs" / "augment_config.json")
    del config[1]["augments"][3]
    bg_imgs = _list_background_images_from_file(config[1])
    assert len(bg_imgs) == 0


def test__check_word_and_char_config():
    config = _load_config(Path(__file__).parent.parent / "typesetter" / "configs" / "augment_config.json")
    cconfig, wconfig = _check_word_and_char_config(config)
    assert isinstance(cconfig, dict)
    assert isinstance(wconfig, dict)
