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

from xllm.experiments import Experiment
from xllm import dist_logger


class MyExperiment(Experiment):

    def before_model_build(self) -> None:
        assert self.model is None
        dist_logger.info("Model is not None", local_rank=self.config.local_rank)

    def after_model_build(self) -> None:
        assert self.model is not None
        dist_logger.info("Model is not None", local_rank=self.config.local_rank)

    def after_train(self) -> None:
        if hasattr(self.model, "my_steps"):
            num_steps = self.model.my_steps
            dist_logger.info(f"Steps: {num_steps}", local_rank=self.config.local_rank)
