# 🦖 Demo project using XLLM

This project is made in order to demonstrate the possibilities of customization of the `xllm` library. In this project,
the simplest extensions will be implemented so that the user can understand as much as possible how to make his project
using `xllm`.

`xllm` enables not only to prototype models, but also facilitates the development of production-ready solutions through
built-in capabilities and customization.

Using `xllm` to train a model is simple and involves these few steps:

1. `Prepare` — Get the data and the model ready by downloading and preparing them. Saves data locally
   to `config.train_local_path_to_data` and `config.eval_local_path_to_data` if you are using eval dataset
2. `Train` — Use the data prepared in the previous step to train the model
3. `Fuse` — If you used LoRA during the training, fuse LoRA
4. `Quantize` — Make your model take less memory by quantizing it

Remember, these tasks in `xllm` start from the command line. In this project, we will cover all aspects of
customizing `xllm`.

[<img src="https://github.com/BobaZooba/xllm/blob/main/static/images/xllm-badge.png" alt="Powered by X—LLM" width="175" height="32"/>](https://github.com/BobaZooba/xllm)

## Useful materials

- [X—LLM Repo](https://github.com/BobaZooba/xllm): main repo of the `xllm` library
- [Quickstart](https://github.com/KompleteAI/xllm/tree/docs-v1#quickstart-): basics of `xllm`
- [Examples](https://github.com/BobaZooba/xllm/examples): minimal examples of using `xllm`
- [Guide](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md): here, we go into detail about everything the library
  can
  do
- [Demo project](https://github.com/BobaZooba/xllm-demo): here's a minimal step-by-step example of how to use X—LLM and
  fit it
  into your own project
- [WeatherGPT](https://github.com/BobaZooba/wgpt): this repository features an example of how to utilize the xllm
  library. Included is a solution for a common type of assessment given to LLM engineers, who typically earn between
  $120,000 to $140,000 annually
- [Shurale](https://github.com/BobaZooba/shurale): project with the finetuned 7B Mistal model

# Installation

Create python virtual environment

```bash
python3 -m venv venv
```

Activate venv

```bash
source venv/bin/activate
```

Install project (editable) and requirements

```bash
pip install -e .
```

## Environment variables

Make `.env` file and fill with your values. Please take a look at `.env.template` file.

W&B values needs only for `train` step.

# Dataset

For demo purposes we will use [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)

This dataset is widely used in the field of language models. It is made up of pairs of dialogues, labeled as `chosen`
and `rejected`, with the distinction being only in the final line of dialogue. Our approach to utilizing this dataset
will be unconventional; rather than adhering to the authors' original design, we will select the `chosen` dialogues and
use them to train a conventional language model to predict the following token.

To achieve this, we will need to implement a new `xllm` dataset.

Implementing a new dataset is very simple. You need to create a new class that inherits from `BaseDataset` and implement
two methods: `get_data` (classmethod) and `get_sample`.

```python
from typing import Tuple, Dict, List, Optional

from xllm import Config
from xllm.datasets import BaseDataset
from xllm.types import RawSample


class AntropicDataset(BaseDataset):

  @classmethod
  def get_data(cls, config: Config) -> Tuple[List[RawSample], Optional[List[RawSample]]]:
    ...

  def get_sample(self, index: int) -> RawSample:
    ...
```

Please read this information to understand what you need to do at each
method: [How to implement dataset](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-implement-dataset)

In brief, the `get_data` method should return a list of training samples (and optionally evaluation samples) with
arbitrary structure. In the `get_sample` method, you need to transform each sample you created in `get_data` into a
specific structure, an example of which is provided below.

Example of `get_sample` output:

```python
{
  "text_parts": [
    "Hello!",
    "My name is Boris"
  ]
}
```

# Registry

We need to register new components such as dataset, collator, trainer, and experiment to make them accessible through
the `xllm` command-line tools.

To do this, let's implement the `components_registry` function. We need to import `datasets_registry` and
add `AntropicDataset` to it with the key `antropic`. We can place the key `antropic` in the constants of our project.

`xllm_demo/core/registry.py`

```python
from xllm.datasets import datasets_registry

from xllm_demo.core.constants import DATASET_KEY
from xllm_demo.core.dataset import AntropicDataset


def components_registry():
  datasets_registry.add(key=DATASET_KEY, item=AntropicDataset)
```

# CLI

At this stage, we are now ready to initiate the first step in our pipeline, which is `prepare`. This step is responsible
for data preparation, downloading the tokenizer, and the model.

Let's make a file `prepare.py`

`xllm_demo/cli/prepare.py`

```python
from xllm.cli import cli_run_prepare

from xllm_demo.core.registry import components_registry

if __name__ == "__main__":
  components_registry()
  cli_run_prepare()
```

Now we can run the `prepare` step, specifying the key of the new dataset.

```sh
python xllm_demo/cli/prepare.py --dataset_key antropic
```

More details you could find
here: [How to add CLI tools to your project](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-add-cli-tools-to-your-project)

### Extend CLI

Let's add the remaining CLI functions: `train`, `fuse`, `quantize`. In each function, we will call the component
registration and pass the new config.

`xllm_demo/cli/train.py`

```python
from xllm.cli import cli_run_train

from xllm_demo.core.registry import components_registry

if __name__ == "__main__":
  components_registry()
  cli_run_train()
```

`xllm_demo/cli/fuse.py`

```python
from xllm.cli import cli_run_fuse

from xllm_demo.core.registry import components_registry

if __name__ == "__main__":
  components_registry()
  cli_run_fuse()
```

`xllm_demo/cli/quantize.py`

```python
from xllm.cli import cli_run_quantize

from xllm_demo.core.registry import components_registry

if __name__ == "__main__":
  components_registry()
  cli_run_quantize()
```

## Run

Now our project is ready and we can train the model.

### Prepare

`cli/prepare.sh`

<details>
  <summary>Bash script</summary>

  ```bash
  #!/bin/bash
  
  python3 xllm_demo/cli/prepare.py \
    --dataset_key antropic \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --path_to_env_file ./.env
  ```

</details>

### Train

`cli/train.sh`

<details>
  <summary>Bash script</summary>

```bash
#!/bin/bash

python3 xllm_demo/cli/train.py \
  --use_gradient_checkpointing True \
  --deepspeed_stage 0 \
  --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --use_flash_attention_2 False \
  --load_in_4bit True \
  --apply_lora True \
  --raw_lora_target_modules all \
  --per_device_train_batch_size 2 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/DemoXLLM7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --path_to_env_file ./.env
```

</details>

### Fuse

`cli/fuse.sh`

<details>
  <summary>Bash script</summary>

```bash
#!/bin/bash

python3 xllm_demo/cli/fuse.py \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --lora_hub_model_id BobaZooba/DemoXLLM7B-v1-LoRA \
  --hub_model_id DemoXLLM7B-v1 \
  --hub_private_repo True \
  --force_fp16 True \
  --fused_model_local_path ./fused_model/ \
  --path_to_env_file ./.env
```

</details>

### Quantize

`cli/quantize.sh`

<details>
  <summary>Bash script</summary>

```bash
#!/bin/bash

python3 xllm_demo/cli/quantize.py \
  --model_name_or_path ./fused_model/ \
  --apply_lora False --stabilize False \
  --quantized_model_path ./quantized_model/ \
  --prepare_model_for_kbit_training False \
  --quantized_hub_model_id DemoXLLM7B-v1-GPTQ \
  --quantized_hub_private_repo True \
  --path_to_env_file ./.env
```

</details>

### DeepSpeed (Multiple GPUs)

We can train our model on multiple GPUs using DeepSpeed.

`cli/train-deepspeed.sh`

<details>
  <summary>Bash script</summary>

```sh
#!/bin/bash

deepspeed --num_gpus=8 xllm_demo/cli/train.py \
  --use_gradient_checkpointing True \
  --deepspeed_stage 2 \
  --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --use_flash_attention_2 False \
  --load_in_4bit True \
  --apply_lora True \
  --raw_lora_target_modules all \
  --per_device_train_batch_size 8 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/DemoXLLM7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --logging_steps 1 \
  --num_train_epochs 3 \
  --save_steps 1000 \
  --max_steps 10050 \
  --path_to_env_file ./.env
```

</details>

### Run scripts

Don't forget to install additional libraries (deepspeed) for this case, which are specified in `requirements-train.txt`.

```sh
pip install -r requirements-train.txt
```

Now we need to grant executable permission to our bash scripts, and then we can start our training.

```sh
chmod -R 755 ./scripts/
```

Run all steps

```sh
./scripts/prepare.sh
./scripts/train.sh
./scripts/fuse.sh
./scripts/quantize.sh
```

**IMPORTANT:** Actually, we can already run our entire project by implementing only the new dataset. All other steps are
optional.

# Collator

The main task of the collator is to convert a list of samples into a batch that can be input to the model. That is, to
turn texts into a tokenized batch and create targets.

Every collator must be inherited from `BaseCollator` and the method `parse_batch` must be implemented.

```python
from typing import List

from xllm.types import RawSample, Batch
from xllm.collators import BaseCollator


class LastPartCollator(BaseCollator):

  def parse_batch(self, raw_batch: List[RawSample]) -> Batch:
    ...
```

Your task is to write the logic of how to process the list in order to eventually obtain a `Batch`. A `Batch` is a
dictionary where the key is a string, and the value is a `PyTorch Tensor`.

More details you could find
here: [How to implement collator](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-implement-collator)

### Registry

Let's expand the registration of new components by adding a collator.

`xllm_demo/core/registry.py`

```python
from xllm.datasets import datasets_registry
from xllm.collators import collators_registry

from xllm_demo.core.constants import DATASET_KEY, COLLATOR_KEY
from xllm_demo.core.dataset import AntropicDataset
from xllm_demo.core.collator import LastPartCollator


def components_registry():
  datasets_registry.add(key=DATASET_KEY, item=AntropicDataset)
  collators_registry.add(key=COLLATOR_KEY, item=LastPartCollator)
```

# Config

We can extend Config. This may be necessary in order to pass the values of new fields via the command line.

Let's add a new field to the config.

`xllm_demo/core/config.py`

```python
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
```

At the moment of calling `get_data` of the newly implemented `Dataset`, we can choose not the `chosen` dialog but, for
example, the `rejected` one. Although this logic is made for demonstration purposes, it can be useful in real projects.
Now we will be able to pass through the command line which field needs to be selected.

Now by default, the `dataset_key` in the project is the one that we have recently implemented. Therefore, there is no
need for us to pass it every time we call the CLI functions. This is an optional step for demonstration.

Now we need to pass the new config class to every CLI method that we implement.

`xllm_demo/cli/prepare.py`

```python
from xllm.cli import cli_run_prepare

from xllm_demo.core.config import DemoXLLMConfig
from xllm_demo.core.registry import components_registry

if __name__ == "__main__":
  components_registry()
  cli_run_prepare(config_cls=DemoXLLMConfig)
```

Run CLI

```sh
python xllm_demo/cli/prepare.py --text_field rejected
```

Please note that we do not need to register a new config. We need to explicitly specify it in the calls to the `xllm`
CLI functions.

More details you could find
here: [How to extend config](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-extend-config-1)

In this example, we do not need to update the config for the other steps (`train`, `fuse`, `quantize`), because all that
we added to the config only concerns the preparation of the dataset, meaning only the `prepare` step. However, sometimes
it may be necessary. We can do this similarly to the `prepare` step.

`xllm_demo/cli/train.py`

```python
from xllm.cli import cli_run_train

from xllm_demo.core.config import DemoXLLMConfig
from xllm_demo.core.registry import components_registry

if __name__ == "__main__":
  components_registry()
  cli_run_train(config_cls=DemoXLLMConfig)
```

# Trainer

We can also implement a new `trainer`. It should inherit from the standard trainer of `transformers`.

`xllm_demo/core/trainer.py`

```python
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
```

Above, we simply added a step counter.

We also need to register the new trainer.

`xllm_demo/core/registry.py`

```python
from xllm.datasets import datasets_registry
from xllm.collators import collators_registry
from xllm.trainers import trainers_registry

from xllm_demo.core.constants import DATASET_KEY, COLLATOR_KEY, TRAINER_KEY
from xllm_demo.core.dataset import AntropicDataset
from xllm_demo.core.collator import LastPartCollator
from xllm_demo.core.trainer import MyLMTrainer


def components_registry():
  datasets_registry.add(key=DATASET_KEY, item=AntropicDataset)
  collators_registry.add(key=COLLATOR_KEY, item=LastPartCollator)
  trainers_registry.add(key=TRAINER_KEY, item=MyLMTrainer)
```

More details you could find
here: [How to implement trainer](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-implement-trainer)

# Experiment

The experiment acts as an aggregator for training, where the necessary components are loaded. We can customize
the `Experiment` by overriding hook methods.

Let's add a few checks and slightly expand the logging.

`xllm_demo/core/experiment.py`

```python
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
```

Let's register the new experiment.

`xllm_demo/core/registry.py`

```python
from xllm.datasets import datasets_registry
from xllm.collators import collators_registry
from xllm.trainers import trainers_registry
from xllm.experiments import experiments_registry

from xllm_demo.core.constants import DATASET_KEY, COLLATOR_KEY, TRAINER_KEY, EXPERIMENT_KEY
from xllm_demo.core.dataset import AntropicDataset
from xllm_demo.core.collator import LastPartCollator
from xllm_demo.core.trainer import MyLMTrainer
from xllm_demo.core.experiment import MyExperiment


def components_registry():
  datasets_registry.add(key=DATASET_KEY, item=AntropicDataset)
  collators_registry.add(key=COLLATOR_KEY, item=LastPartCollator)
  trainers_registry.add(key=TRAINER_KEY, item=MyLMTrainer)
  experiments_registry.add(key=EXPERIMENT_KEY, item=MyExperiment)
```

More details you could find
here: [How to implement experiment](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md#how-to-implement-experiment)

# Run

Having implemented and registered the new components, we can make some adjustments to the training bash scripts, as only
in this step might the new components be utilized.

We add three new lines:

```bash
  --collator_key last_part \
  --trainer_key steps \
  --experiment_key check_model
```

### Train

`cli/train.sh`

<details>
  <summary>Bash script</summary>

```bash
#!/bin/bash

python3 xllm_demo/cli/train.py \
  --use_gradient_checkpointing True \
  --deepspeed_stage 0 \
  --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --use_flash_attention_2 False \
  --load_in_4bit True \
  --apply_lora True \
  --raw_lora_target_modules all \
  --per_device_train_batch_size 2 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/DemoXLLM7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --path_to_env_file ./.env \
  --collator_key last_part \
  --trainer_key steps \
  --experiment_key check_model
```

</details>

### DeepSpeed (Multiple GPUs)

`cli/train-deepspeed.sh`

<details>
  <summary>Bash script</summary>

```sh
#!/bin/bash

deepspeed --num_gpus=8 xllm_demo/cli/train.py \
  --use_gradient_checkpointing True \
  --deepspeed_stage 2 \
  --stabilize True \
  --model_name_or_path mistralai/Mistral-7B-v0.1 \
  --use_flash_attention_2 False \
  --load_in_4bit True \
  --apply_lora True \
  --raw_lora_target_modules all \
  --per_device_train_batch_size 8 \
  --warmup_steps 1000 \
  --save_total_limit 0 \
  --push_to_hub True \
  --hub_model_id BobaZooba/DemoXLLM7B-v1-LoRA \
  --hub_private_repo True \
  --report_to_wandb True \
  --logging_steps 1 \
  --num_train_epochs 3 \
  --save_steps 1000 \
  --max_steps 10050 \
  --path_to_env_file ./.env \
  --collator_key last_part \
  --trainer_key steps \
  --experiment_key check_model
```

</details>

## Run scripts

Afterward, we can run all the scripts just as we did before.

```sh
./scripts/prepare.sh
./scripts/train.sh
./scripts/fuse.sh
./scripts/quantize.sh
```

# Conclusions

`xllm` library has a plethora of customization options. Not all of them are necessary, and most often you will only
utilize the implementation of a new dataset, which is by design.

# 🎉 Done! You are awesome!

## Now you know how to create projects using `xllm`

## Useful materials

- [X—LLM Repo](https://github.com/BobaZooba/xllm): main repo of the `xllm` library
- [Quickstart](https://github.com/KompleteAI/xllm/tree/docs-v1#quickstart-): basics of `xllm`
- [Examples](https://github.com/BobaZooba/xllm/examples): minimal examples of using `xllm`
- [Guide](https://github.com/BobaZooba/xllm/blob/main/GUIDE.md): here, we go into detail about everything the library
  can
  do
- [Demo project](https://github.com/BobaZooba/xllm-demo): here's a minimal step-by-step example of how to use X—LLM and
  fit it
  into your own project
- [WeatherGPT](https://github.com/BobaZooba/wgpt): this repository features an example of how to utilize the xllm
  library. Included is a solution for a common type of assessment given to LLM engineers, who typically earn between
  $120,000 to $140,000 annually
- [Shurale](https://github.com/BobaZooba/shurale): project with the finetuned 7B Mistal model

## Tale Quest

`Tale Quest` is my personal project which was built using `xllm` and `Shurale`. It's an interactive text-based game
in `Telegram` with dynamic AI characters, offering infinite scenarios

You will get into exciting journeys and complete fascinating quests. Chat
with `George Orwell`, `Tech Entrepreneur`, `Young Wizard`, `Noir Detective`, `Femme Fatale` and many more

Try it now: [https://t.me/talequestbot](https://t.me/TaleQuestBot?start=Z2g)
