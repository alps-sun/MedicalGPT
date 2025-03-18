#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Auth ： mashang
@Date ： 2025/3/18 23:20
@Version ：
@Description ：mygrpo_train.py from trl.trainer.grpo_trainer
"""

from transformers import Trainer, PreTrainedModel
from typing import Union, Callable, Optional, Any
from datasets import Dataset, IterableDataset

# 我们称之为奖励函数的是一个可调用函数，它接受一系列提示和完成，并返回一系列奖励。当它是字符串时，它是一个模型ID，因此它被加载为预训练模型
# 当是str时， 这是一个模型id, 此时奖励函数是一个模型
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOConfig:
    # TODO
    pass


class MyGPPOTrainer(Trainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,

    ):
        pass
