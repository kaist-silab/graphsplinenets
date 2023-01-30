from typing import Callable

# Export the PROJECT_ROOT enviroment as the directory where this script is called from.
# You may want to modify these depending on your import structure.
import os

os.environ["PROJECT_ROOT"] = "."
os.environ["PYTHONPATH"] = "."

# Trick for avoiding problems
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

import dotenv
import hydra
from omegaconf import OmegaConf, DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)

# Hack
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def dictconfig_filter_key(d: DictConfig, fn: Callable) -> DictConfig:
    """Only keep keys where fn(key) is True. Support nested DictConfig."""
    return DictConfig(
        {
            k: dictconfig_filter_key(v, fn) if isinstance(v, DictConfig) else v
            for k, v in d.items()
            if fn(k)
        }
    )


@hydra.main(version_base="1.2", config_path="configs/", config_name="train")
def main(config: DictConfig):
    # fix for: _tkinter.TclError: no display name and no $DISPLAY environment variable
    import matplotlib

    matplotlib.use("Agg")

    # Remove config keys that start with '__'. These are meant to be used only in computing
    # other entries in the config.-----
    config = dictconfig_filter_key(config, lambda k: not k.startswith("__"))

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import main as train
    from src.eval import main as evaluate
    from src.utils import utils

    # A couple of optional utilities:
    # - disabling python warnings
    # - forcing debug-friendly configuration
    # - verifying experiment name is set when running in experiment mode
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    mode = config.get("mode", "train")
    if mode not in ["train", "eval"]:
        raise NotImplementedError(f"mode {mode} not supported")
    if mode == "train":
        return train(config)
    elif mode == "eval":
        return evaluate(config)


if __name__ == "__main__":
    main()
