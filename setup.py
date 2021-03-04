import os
import setuptools
from pathlib import Path
from typing import List


def package_files(directory: str) -> List[str]:
    """Gets a list of all the absolute file paths, including subdirectories, of the given directory relative
     to the current file. Returns them as a list of strings.

    Arguments:
        directory {str} -- The directory to look in.

    Returns:
        List[str] -- The list of absolute paths.
    """
    paths = []
    directory = str(Path(__file__).parent.absolute()) + str(Path(directory))
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_files = package_files("/src/inked")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="inked",
    version="0.1.0",
    author="Capgemini Invent IDE",
    description="inked creates a labelled dataset of handwritten and/or typefont words given a dataset of characters. It uses a range of image augmentations at both the character and word level to create a diverse set of images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src", exclude=["tests"]),
    install_requires=[
        "Pillow",
        "numpy",
        "opencv-python",
        "torch",
        "fontTools",
        "requests",
        "scikit-image",
        "scipy",
        "matplotlib",
        "lmdb",
        "imutils",
    ],
    url="https://github.com/CapgeminiInventIDE/inked",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: OS Independent",
    ],
    keywords=["image generation", "data", "handwriting", "typefont", "OCR"],
    zip_safe=True,
    python_requires=">=3.6",
    package_data={"inked": extra_files},
    extras_require={"demo": ["fastapi", "uvicorn", "python-multipart", "jinja2"]},
    entry_points={"console_scripts": ["inked-demo=inked.server:main"]},
)
