"""
# Steps to build and upload to PyPI
# 0. Update version in setup.py
# 1. python setup.py sdist bdist_wheel
# 2. twine upload dist/*
"""

from setuptools import setup


def read_requirements(file):
    try:
        with open(file, "r") as f:
            return [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        return []


with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="lmunit",
    version="1.0.1",
    description="Language Model Unit Testing Framework",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="William Berrios",
    url="https://github.com/williamberrios/LMUnit",
    license="MIT",
    python_requires=">=3.10",
    packages=["lmunit"],
    install_requires=read_requirements("requirements/requirements.txt"),
    extras_require={"dev": read_requirements("requirements/dev.txt")},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
