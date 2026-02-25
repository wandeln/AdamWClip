from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="AdamWClip",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["torch"],
    author="Nils Wandel",
    author_email="wandeln@cs.uni-bonn.de",
    description="AdamWClip is an optimizer that extends AdamW with adaptive gradient clipping.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wandeln/AdamWClip",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
