from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("LICENSE", "r", encoding="utf-8") as f:
    LICENSE = f.read()

setup(
    name='torchutils',
    version='0.0.1',
    packages=setuptools.find_packages(),
    url='https://github.com/tchaye59/torchutils',
    license=LICENSE,
    author='Jude TCHAYE',
    author_email='tchaye59@gmail.com',
    description='My torch models training utilities',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
