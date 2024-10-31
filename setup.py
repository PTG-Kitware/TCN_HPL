#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="tcn_hpl",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/user/project",
    # install_requires=["lightning", "hydra-core", "hydra-colorlog"],
    packages=find_packages(include=["tcn_hpl", "tcn_hpl.*"]),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = tcn_hpl.train:main",
            "eval_command = tcn_hpl.eval:main",
            "bbn_create_truth_coco = tcn_hpl.data.utils.bbn:create_truth_coco",
            "kwcoco_guided_subset = tcn_hpl.data.utils.kwcoco_guided_subset:main",
        ]
    },
)
