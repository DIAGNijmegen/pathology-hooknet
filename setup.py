from setuptools import setup, find_packages

setup(
    name="hooknet",
    version="0.0.3",
    author="Mart van Rijthoven",
    author_email="mart.vanrijthoven@gmail.com",
    packages=find_packages(exclude=("tests", "notebooks", "scripts", "os-level-virtualization", "docs")),
    url="http://pypi.python.org/pypi/hooknet/",
    license="LICENSE.txt",
    install_requires=[
        "numpy>=1.18.5",
    ],
    long_description="HookNet: multi-resolution whole-slide image segmentation",
)