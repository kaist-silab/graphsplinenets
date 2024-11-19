<div align="center">

# GraphSplineNets

[![arXiv](https://img.shields.io/badge/arXiv-2310.16397-b31b1b.svg)](https://arxiv.org/abs/2310.16397) [![OpenReview](https://img.shields.io/badge/⚖️-OpenReview-8b1b16)](https://openreview.net/forum?id=MGPST5I9DO) 
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

</div>

![graphsplinenets](https://github.com/user-attachments/assets/1d3a2018-c588-4b19-9b82-553e47f1b1eb)


## Abstract

While complex simulations of physical systems have been widely used in engineering and scientific computing, lowering their often prohibitive computational requirements has only recently been tackled by deep learning approaches. In this paper, we present GraphSplineNets, a novel deep-learning method to speed up the forecasting of physical systems by reducing the grid size and number of iteration steps of deep surrogate models. Our method uses two differentiable orthogonal spline collocation methods to efficiently predict response at any location in time and space. Additionally, we introduce an adaptive collocation strategy in space to prioritize sampling from the most important regions. GraphSplineNets improve the accuracy-speedup tradeoff in forecasting various dynamical systems with increasing complexity, including the heat equation, damped wave propagation, Navier-Stokes equations, and real-world ocean currents in both regular and irregular domains.


## How to run

### Installation

`pip install -r requirements.txt`

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

## Citation

If you find our work useful, please consider citing us:

```text
@article{hua2024learning_graphsplinenets,
  title={Learning Efficient Surrogate Dynamic Models with Graph Spline Networks},
  author={Hua, Chuanbo and Berto, Federico and Poli, Michael and Massaroli, Stefano and Park, Jinkyoo},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```

