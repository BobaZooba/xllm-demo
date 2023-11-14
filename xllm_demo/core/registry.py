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

from xllm.datasets import datasets_registry
from xllm.collators import collators_registry
from xllm.trainers import trainers_registry
from xllm.experiments import experiments_registry

from xllm_demo.core.constants import DATASET_KEY, COLLATOR_KEY, TRAINER_KEY, EXPERIMENT_KEY
from xllm_demo.core.dataset import AntropicDataset
from xllm_demo.core.experiment import MyExperiment
from xllm_demo.core.collator import LastPartCollator
from xllm_demo.core.trainer import MyLMTrainer


def components_registry():
    datasets_registry.add(key=DATASET_KEY, item=AntropicDataset)
    collators_registry.add(key=COLLATOR_KEY, item=LastPartCollator)
    trainers_registry.add(key=TRAINER_KEY, item=MyLMTrainer)
    experiments_registry.add(key=EXPERIMENT_KEY, item=MyExperiment)
