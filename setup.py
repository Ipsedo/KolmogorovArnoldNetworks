# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="kan",
    version="1.0.0",
    author="Samuel Berrien",
    packages=find_packages(include=["kan", "kan.*"]),
)
