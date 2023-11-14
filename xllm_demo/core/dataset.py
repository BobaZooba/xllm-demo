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

from typing import Tuple, Dict, List, Optional

from tqdm import tqdm
import datasets

from xllm.datasets import BaseDataset
from xllm.types import RawSample
from xllm import enums

from xllm_demo.core.config import DemoXLLMConfig


class AntropicDataset(BaseDataset):
    _HF_DATASET_ID = "Anthropic/hh-rlhf"

    @classmethod
    def get_data(cls, config: DemoXLLMConfig) -> Tuple[List[RawSample], Optional[List[RawSample]]]:
        rlhf_dataset = datasets.load_dataset(cls._HF_DATASET_ID)

        parsed_data: Dict[str, List[RawSample]] = dict()

        for split in ["train", "test"]:

            parsed_data[split] = list()

            for sample in tqdm(rlhf_dataset[split], desc=f"Parsing {split}"):
                text_parts = sample[config.text_field].split("\n\n")[1:]

                parsed_data[split].append(text_parts)

        train = parsed_data["train"]
        evaluation = parsed_data["test"]

        return train, evaluation

    def get_sample(self, index: int) -> RawSample:
        sample = {
            enums.General.text_parts: self.data[index]
        }
        return sample
