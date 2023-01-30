<div align="center">

# GraphSplineNets

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

GraphSplineNets code based on the [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template)

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/cbhua/model-graph-spline-nets
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Quick Start

Run the example notebook:
```bash
python run.py experiment=example
```

## Examples

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)


Train model with default configuration

```bash
# train on CPU
python run.py trainer=cpu

# train on GPU
python run.py trainer=gpu
```

You can override any parameter from command line like this

```bash
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```
