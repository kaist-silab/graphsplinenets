#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="GraphSplineNets",
    author="Chuanbo Hua, Federico Berto",
    author_email="",
    url="https://github.com/cbhua/model-graph-spline-nets",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)