.DEFAULT_GOAL:=help

.EXPORT_ALL_VARIABLES:

ifndef VERBOSE
.SILENT:
endif

#* Variables
PYTHON := python3
PYTHON_RUN := $(PYTHON) -m

help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

#* Installation
.PHONY: install
install:  ## Installation
	$(PYTHON_RUN) pip install -r requirements.txt

.PHONY: install-train
install-train:  ## Installation including deepspeed, flash-attn and auto-gptq
	$(PYTHON_RUN) pip install -r requirements-train.txt

#* Prepare
.PHONY: scripts-access
scripts-access:  ## Give access to bash scripts
	chmod -R 755 ./scripts/

#* Download
.PHONY: download
prepare:  ## Download data and model, prepare data for training
	./scripts/prepare.sh

#* Train
.PHONY: train
train:  ## Train on single GPU
	./scripts/train.sh

.PHONY: deepspeed-train
deepspeed-train:  ## Train on multiple GPUs using DeepSpeed
	./scripts/train-deepspeed.sh

#* Fuse
.PHONY: fuse
fuse:  ## Fuse LoRA to the model
	./scripts/fuse.sh

#* Quantization
.PHONY: gptq-quantize
quantize:  ## GPTQ quantization
	./scripts/quantize.sh
