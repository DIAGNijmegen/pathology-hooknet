from setuptools import setup, find_packages

setup(
    name="hooknet",
    version="0.0.2",
    author="Mart van Rijthoven",
    author_email="mart.vanrijthoven@gmail.com",
    packages=find_packages(exclude=("tests", "notebooks", "scripts", "os-level-virtualization", "docs")),
    url="http://pypi.python.org/pypi/hooknet/",
    license="LICENSE.txt",
    install_requires=[
        "tensorflow_gpu>=2.3.0",
        "opencv_python-headless>=4.4.0",
        "scikit_image>=0.17.2",
        "numpy>=1.18.5",
        "PyYAML>=5.3.1",
    ],
    long_description="HookNet: multi-resolution whole-slide image segmentation",
)