import json
import logging
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import imutils
import numpy as np
from PIL import Image
from skimage.util import random_noise

from .utils import random_sample_arr

logger = logging.getLogger()


class Augmentor:
    """Augmentator of images from Character and Word class objects."""

    def __init__(self, config: PurePath = Path(__file__).parent / "configs" / "augment_config.json"):
        _augment_config = _load_config(config)
        self._char_aug_config, self._word_aug_config = _check_word_and_char_config(_augment_config)
        self.bg_image_paths = _list_background_images_from_file(self._word_aug_config)
        self.text_textures = list((Path(__file__).parent / "data" / "text_textures").glob("*.png"))

        self.enabled_character_augments = [
            [aug["name"], aug["min_severity"], aug["max_severity"]]
            for aug in self._char_aug_config["augments"]
            if aug["enabled"]
        ]
        self.enabled_word_augments = [
            [aug["name"], aug["min_severity"], aug["max_severity"]]
            for aug in self._word_aug_config["augments"]
            if aug["enabled"]
        ]

    def augment(self, glyph: Union["Character", "Word"]) -> Image.Image:  # type: ignore # noqa
        """Augments a given character or word object with image augmenations such as blur, stretch, salt and pepper.

        Args:
            glyph (Union["Character", "Word"]): A Character or Word object

        Raises:
            TypeError: TypeError if a Word of Character is not given.

        Returns:
            Image.Image: A PIL image with augmentations
        """

        if glyph.__class__.__name__ == "Character":
            config = self._char_aug_config
            enabled_augments = self.enabled_character_augments
        elif glyph.__class__.__name__ == "Word":
            config = self._word_aug_config
            enabled_augments = self.enabled_word_augments
        else:
            raise TypeError("glyph must be a Word or Character")

        self.image = np.asarray(glyph.image)
        self.augments_performed = {}

        num_of_augs = np.random.randint(config["min_augments"], config["max_augments"] + 1)
        random_augs = random_sample_arr(enabled_augments, num_of_augs)
        # NOTE: this may cause the # of augments to be less than "min augments" set in config
        random_augs = _drop_mutually_exclusive_augments(random_augs)
        random_augs = _order_augments(random_augs)
        for augment in random_augs:
            severity = self._get_random_severity(augment[1], augment[2])
            logger.info(f"Adding {augment[0]} with severity {severity} to {glyph.__class__.__name__} '{glyph.text}'.")
            getattr(self, f"_add_{augment[0]}")(severity)
            self.augments_performed[augment[0]] = severity

        self.image = Image.fromarray(self.image)

        logger.info(f"Augmentations complete on {glyph.__class__.__name__} '{glyph.text}'.")
        return self.image

    def _get_random_severity(
        self, minimum: Union[int, float, list], maximum: Union[int, float, list]
    ) -> Union[int, float, tuple]:
        """Randomly choises an int between (and INCLUSIVE) of minimum and maximum (if provided with ints).
        OR if minimum and maximum are floats, will randomly (and uniformly) return a float between both numbers.

        Args:
            minimum (Union[int, float]): Lower end of the random number choice (inclusive).
            maximum (Union[int, float]): Upper end of the random number choice (inclusive).

        Raises:
            TypeError: TypeError if minimum and maximum are not the same types of numbers.

        Returns:
            Union[int, float]: int or float between minimum and maximum, depending on input type.
        """
        if isinstance(minimum, list) and isinstance(maximum, list):
            return (
                np.random.randint(minimum[0], maximum[0] + 1),
                np.random.randint(minimum[1], maximum[1] + 1),
                np.random.randint(minimum[2], maximum[2] + 1),
            )
        elif isinstance(minimum, int) and isinstance(maximum, int):
            return np.random.randint(minimum, maximum + 1)
        elif isinstance(minimum, float) and isinstance(maximum, float):
            return round(np.random.uniform(minimum, maximum), 5)
        raise TypeError("Severity values must either both be floats, ints or lists but not a combination.")

    """
    character level augments
    """

    def _add_text_fill(self, severity: List[int]) -> None:
        fill_img = np.ones_like(self.image)

        fill_img[:, :, 0], fill_img[:, :, 1], fill_img[:, :, 2] = severity

        # NOTE: method to use when image is using rgb layers to show color
        # norm_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # _, otsu = cv2.threshold(norm_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # +cv2.THRESH_OTSU for more transparent text
        # mask = cv2.bitwise_not(norm_img, otsu)

        # NOTE: method to use when image is using transparent layer to show color
        _, otsu = cv2.threshold(self.image[:, :, 3], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.bitwise_and(self.image[:, :, 3], otsu)

        mask = (cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255).astype(
            np.uint8
        )

        fill_img[:, :, 3] = mask
        self.image = fill_img

    def _add_text_texture(self, severity: List[int], background: Optional[np.ndarray] = None) -> None:
        # NOTE: severity not in use but given for consistency
        if background is None:
            random_texture_path = self.text_textures[np.random.randint(0, len(self.text_textures))]
            background = cv2.cvtColor(cv2.imread(str(random_texture_path), flags=cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        height, width = self.image.shape[0:2]
        # check size of texture image
        if background.shape[0] <= height:
            background = cv2.resize(background, (background.shape[1], height + 1))
        if background.shape[1] <= width:
            background = cv2.resize(background, (width + 1, background.shape[0]))

        ny, nx = np.random.randint(0, background.shape[0] - height), np.random.randint(0, background.shape[1] - width)
        new_image = background[ny : ny + height, nx : nx + width]
        new_image = background[ny : ny + height, nx : nx + width]

        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2RGBA)
        new_image[:, :, 3] = self.image[:, :, 3]

        self.image = new_image
        # cv2.imwrite("after_texture.png", self.image)

    def _add_rotation(self, severity: int) -> None:
        """Adds a warpped rotation (shear) augmentation to the given Charactor or Word image.

        Args:
            severity (float): Intensity of the shear between 0 and 1 (%).

        """
        # image as np.array
        self.image = imutils.rotate_bound(self.image, severity)

    def _add_scale_x(self, severity: float) -> None:
        """Stretches or squishes the x-axis of the Character or Word image

        Args:
            severity (float): Scale factor (0.5 will squish 50% and 2 will strech the image 100%) in the x direction only.
                              Note this does change the image size.

        """
        # image as np.array
        self.image = cv2.resize(self.image, None, fx=severity, fy=1)

    def _add_scale_y(self, severity: float) -> None:
        """Stretches or squishes the y-axis of the Character or Word image

        Args:
            severity (float): Scale factor (0.5 will squish 50% and 2 will strech the image 100%) in the y direction only.
                              Note this does change the image size.

        """
        # image as np.array
        self.image = cv2.resize(self.image, None, fx=1, fy=severity)

    """
    word level augments
    """

    def _add_xblur(self, severity: int) -> None:
        """Blurs the Character or Word image on the x-axis. Keeping the image shape but using a kernal to smoothen the image.

        Args:
            severity (int): x-directional kernal size.
        """
        # image as np.array
        self.image = cv2.blur(self.image, ksize=(1, severity), borderType=cv2.BORDER_ISOLATED)

    def _add_yblur(self, severity: int) -> None:
        """Blurs the Character or Word image on the y-axis. Keeping the image shape but using a kernal to smoothen the image.

        Args:
            severity (int): y-directional kernal size.
        """
        # image as np.array
        self.image = cv2.blur(self.image, ksize=(severity, 1), borderType=cv2.BORDER_ISOLATED)

    def _add_random_pixel_noise(self, severity: float) -> None:
        """Adds random pixels throughout the Character or Word image. These can be of any color.

        Args:
            severity (float): Percentage of pixels to change. Between 0 and 1.
        """
        # image as np.array
        noise_img = random_noise(self.image, mode="s&p", amount=severity)
        self.image = np.array(255 * noise_img, dtype="uint8")

    # XXX: Can we use sythtext localisation of areas to add text to (and crop out background)

    def _add_bg_image(self, severity: float, background: Optional[np.ndarray] = None) -> None:
        """Adds a random background image to the Character or Word image.
        You can edit the folder it looks in by changing self.bg_image_paths.

        Args:
            severity (float): Not in use.
        """
        # NOTE: severity not in use but given for consistency
        # NOTE: could allow for transparent backgrounds as well: https://stackoverflow.com/a/59211216/15210788
        if background is None:
            random_bg_path = self.bg_image_paths[np.random.randint(0, len(self.bg_image_paths))]
            background = cv2.imread(str(random_bg_path), flags=cv2.IMREAD_COLOR)
        if background.shape[-1] == 3:
            background = cv2.cvtColor(background, cv2.COLOR_RGB2RGBA)

        height, width = self.image.shape[0:2]
        # check size of background image
        if background.shape[0] <= height:
            background = cv2.resize(background, (background.shape[1], height + 1))
        if background.shape[1] <= width:
            background = cv2.resize(background, (width + 1, background.shape[0]))
        alpha = self.image[:, :, 3] / 255.0

        ny, nx = np.random.randint(0, background.shape[0] - height), np.random.randint(0, background.shape[1] - width)
        new_image = background[ny : ny + height, nx : nx + width]
        new_image = background[ny : ny + height, nx : nx + width]

        for color in range(0, 3):
            new_image[:, :, color] = alpha * self.image[:, :, color] + new_image[:, :, color] * (1 - alpha)
        self.image = new_image

    def _add_bg_colour(self, severity: float) -> None:
        """Adds background colour to an image.

        Args:
            severity (int): Metric representing the strength of the augmentation (i.e. opacity of background colour)

        """
        overlay = np.array(self.image).copy()
        output = np.array(self.image).copy()
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )  # choosing random colour
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), color, -1)
        # severity is alpha/opacity
        cv2.addWeighted(overlay, severity, output, 1 - severity, 0, output)
        # cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        self.image = output

    """
    FIXME: Ensure has background first? Or build own functions to do this as
    These augments require non-transparent images.
    Disabled for now since the broader use case will require transparency
    """
    # def _add_contrast(self, severity: float) -> None:
    #     self.image = iaa.imgcorruptlike.Contrast(severity=severity)(image=self.image)

    # def _add_spatter(self, severity: float) -> None:
    #     self.image = iaa.imgcorruptlike.Spatter(severity=severity)(image=self.image)

    # def _add_jpeg_compression(self, severity: float) -> None:
    #     self.image = iaa.imgcorruptlike.JpegCompression(severity=severity)(image=self.image)

    # def _add_pixelate(self, severity: float) -> None:
    #     self.image = iaa.imgcorruptlike.Pixelate(severity=severity)(image=self.image)

    # def _add_elastic_transformation(self, severity: float) -> None:
    #     self.image = iaa.imgcorruptlike.apply_elastic_transform(image=self.image, severity=severity, seed=None)


def _load_config(augment_config: Union[str, PurePath, Dict[Any, Any]]) -> List[Dict[Any, Any]]:
    """Loads the augmentations config file to be used by Augmentor.

    Args:
        augment_config (Union[str, PurePath, Dict]): Where the config can be found.

    Raises:
        Exception: If unable to find config file.

    Returns:
        Dict: Augmentation configurations containing available Character and Word augmentations,
        min and max augmentations to perform per object,
        min and max severity of each augmentation and any extra information required.
    """
    if isinstance(augment_config, (str, PurePath)):
        with open(augment_config, "r") as f:
            return json.load(f)
    raise LoadingConfigInputException(
        f"Augment config file could not be loaded, expecting string, Path or dict but got {type(augment_config)}."
    )


def _check_word_and_char_config(augment_config: List[Dict[str, Any]]) -> Tuple[dict, dict]:
    """Takes in a augmentation config dict and breaks it up into Character and Word augments while
    also checking that min and max augments are set correctly and wont error the Augmentor.

    Args:
        augment_config (List[Dict[str, Any]]): augmentation dict. See augmentor._load_config.

    Returns:
        Tuple[Dict, Dict]: Augment configs for the Character class and Word class respectively.
    """
    for _object in augment_config:
        enabled_augments = len([aug["name"] for aug in _object["augments"] if aug["enabled"]])
        _object["min_augments"], _object["max_augments"] = (
            min(_object["min_augments"], _object["max_augments"], enabled_augments),
            min(enabled_augments, max(_object["min_augments"], _object["max_augments"])),
        )
        if _object.get("object") == "Character":
            _char_aug_config = _object
        elif _object.get("object") == "Word":
            _word_aug_config = _object
    return _char_aug_config, _word_aug_config


def _list_background_images_from_file(word_aug_config: Dict) -> List[Path]:
    """Searches for png images from the given path in word_aug_config["augments"] where ["name"] == "bg_image"
    and then in the "folder" key for the path. This can be edited via Augmentor._word_aug_config.

    Args:
        word_aug_config (Dict): Augmentation config for the Word class objects.

    Returns:
        List[str]: list of full paths to background images.
    """
    for aug in word_aug_config["augments"]:
        if aug["name"] == "bg_image":
            if (aug.get("folder") is not None) and aug["enabled"]:
                background_img_path = Path(aug["folder"])
            else:
                background_img_path = Path(__file__).parent / "data" / "background_images"

            bg_imgs = list(background_img_path.glob("*.png"))
            if ((not background_img_path.exists()) or (len(bg_imgs) == 0)) and aug["enabled"]:
                raise BackgroundImagesException(
                    "Unable to find background images folder, please either point to a folder containing images in the config or set bg_image 'enable' to false."
                )
                # could make this just flip background to false, but like the notice given to user
            logger.info(f"Found {len(bg_imgs)} background images for word augmentations.")
            return bg_imgs
    return []


def _drop_mutually_exclusive_augments(augmentations: List[List[Any]]) -> List[List[Any]]:
    augs = [aug[0] for aug in augmentations]

    if "text_fill" in augs and "text_texture" in augs:
        remove = np.random.choice(["text_fill", "text_texture"])
        augmentations.pop(augs.index(remove))

    elif "bg_image" in augs and "bg_colour" in augs:
        remove = np.random.choice(["bg_image", "bg_colour"])
        augmentations.pop(augs.index(remove))

    return augmentations


def _order_augments(augmentations: List[List[Any]]) -> List[List[Any]]:
    augs = [aug[0] for aug in augmentations]

    # put background images second only to rotation
    firsts = ["bg_image", "bg_colour"]  # 'text_fill', 'text_texture',
    for i in firsts:
        if i in augs:
            augmentations.insert(0, augmentations.pop(augs.index(i)))

    # put rotation before anything else
    augs = [aug[0] for aug in augmentations]
    if "rotation" in augs:
        augmentations.insert(0, augmentations.pop(augs.index("rotation")))

    return augmentations


class BackgroundImagesException(Exception):
    """No images found in background images folder. Expecting .png files."""

    ...


class LoadingConfigInputException(Exception):
    """No images found in background images folder. Expecting .png files."""

    ...
