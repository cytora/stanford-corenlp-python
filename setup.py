from setuptools import setup, find_packages

PACKAGE = "corenlp"
NAME = "stanford-corenlp-python"
DESCRIPTION = "A Stanford Core NLP wrapper (wordseer fork)"
AUTHOR = "Hiroyoshi Komatsu, Dustin Smith, Aditi Muralidharan"
AUTHOR_EMAIL = "aditi.shrikumar@gmail.com"
URL = "https://github.com/Wordseer/stanford-corenlp-python"
VERSION = "3.3.0-0"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(),
    package_data = {"": ["*.properties"],
        "corenlp": ["*.properties"]},
    install_requires=[
        "pexpect >= 2.4",
        "unidecode >= 0.04.12",
        "xmltodict >= 0.4.6",
        "functools32 == 3.2.3-1"
    ],
    classifiers=[
        ("License :: OSI Approved :: GNU General Public License v2 or later "
            "(GPLv2+)"),
        "Programming Language :: Python",
    ],
)

