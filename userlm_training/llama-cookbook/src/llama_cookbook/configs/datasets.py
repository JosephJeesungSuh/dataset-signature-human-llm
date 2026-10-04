# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

@dataclass
class userlm_dataset:
    dataset: str = "userlm_dataset"
    train_split: str = "train"
    test_split: str = "validation"
