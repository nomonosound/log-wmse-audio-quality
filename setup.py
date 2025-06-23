import codecs
import os
import re

from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="log-wmse-audio-quality",
    version=find_version("log_wmse_audio_quality", "__init__.py"),
    author="Iver Jordal",
    description=(
        "logWMSE is an audio quality metric with support for digital silence target."
        " Useful for evaluating audio source separation systems, even when there are"
        " many audio tracks or stems."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nomonosound/log-wmse-audio-quality",
    packages=find_packages(exclude=["dev", "tests"]),
    package_data={"log_wmse_audio_quality": ["filter_ir.pkl"]},
    license="Apache 2.0",
    install_requires=[
        "numpy>=1.22,<3",
        "soxr>=0.3.2,<1",
        "scipy>=1.3,<2",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Issue Tracker": "https://github.com/nomonosound/log-wmse-audio-quality/issues",
    },
)
