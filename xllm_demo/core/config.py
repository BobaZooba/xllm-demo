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

from dataclasses import dataclass, field

from xllm import Config

from xllm_demo.core.constants import DATASET_KEY


@dataclass
class DemoXLLMConfig(Config):
    text_field: str = field(default="chosen", metadata={
        "help": "Field for Antropic RLHF dataset",
    })
    dataset_key: str = field(default=DATASET_KEY, metadata={
        "help": "Dataset key",
    })
