<div align="center">

# GraphSplineNets

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

> Repository has work in progress - Arxiv paper, OpenReview, and final touches coming soon!


## Description

GraphSplineNets code based on the [Lightning Hydra Template](https://github.com/ashleve/lightning-hydra-template)

## How to run

### Clone repository
First, download the [repository on Anonymous Github](https://anonymous.4open.science/r/graphsplinenets) by running this on a terminal:
```bash
curl -sSL https://anonymous.4open.science/r/graphsplinenets/src/utils/download_anonymous_github.py | python3 -
```
or use the [downloader script](https://anonymous.4open.science/r/graphsplinenets/src/utils/download_anonymous_github.py)  and run it with your favorite Python interpreter. Note that we use the above since Anonymous Github is currently not providing a way to download the repository as a zip file.


### Install dependencies
```bash
# Automatically install dependencies with light the torch
pip install light-the-torch && python3 -m light_the_torch install --upgrade -r requirements.txt
```

The above script will [automatically install](https://github.com/pmeier/light-the-torch) PyTorch with the right GPU version for your system. Alternatively, you can use `pip install -r requirements.txt`

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
