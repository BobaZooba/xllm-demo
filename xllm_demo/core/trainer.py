# Copyright 2023 Boris Zubarev. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, Dict, Tuple, Optional

from peft import PeftModel
from torch import Tensor
from transformers import PreTrainedModel, TrainingArguments

from xllm.trainers import LMTrainer
from xllm.datasets import BaseDataset
from xllm.collators import BaseCollator

from xllm_demo.core.config import DemoXLLMConfig


class MyLMTrainer(LMTrainer):

    def __init__(
            self,
            config: DemoXLLMConfig,
            model: Union[PreTrainedModel, PeftModel],
            args: TrainingArguments,
            data_collator: BaseCollator,
            train_dataset: BaseDataset,
            ignore_index: int,
            eval_dataset: Optional[BaseDataset] = None,
    ):
        super().__init__(config, model, args, data_collator, train_dataset, ignore_index, eval_dataset)

        self.my_steps = 0

    def compute_loss(
            self,
            model: Union[PreTrainedModel, PeftModel],
            inputs: Dict[str, Tensor],
            return_outputs: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        self.my_steps += 1
        return super().compute_loss(model=model, inputs=inputs, return_outputs=return_outputs)
