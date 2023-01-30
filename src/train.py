from pathlib import Path
from typing import List, Sequence

import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils import utils, instantiate_callbacks, instantiate_loggers, pylogger

import pytorch_lightning as pl
from pytorch_lightning import (
    Trainer,
    seed_everything,
    LightningModule,
    LightningDataModule,
    Callback,
)
from pytorch_lightning.loggers import LightningLoggerBase

log = pylogger.get_pylogger(__name__)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float:

    # We want to add fields to config so need to call OmegaConf.set_struct
    OmegaConf.set_struct(cfg, False)

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.seed is not None:
        seed_everything(cfg.seed, workers=True)

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # _recursive_ set to False does not instantiate nested objects, so we can dynamically modify the model and do it inside the LightningModule
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model, cfg=cfg, _recursive_=False
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    # TODO: give a look at this part, may not be totally correct
    # https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py
    ckpt_cfg = {}
    if cfg.get("resume"):
        try:
            checkpoint_path = Path(cfg.callbacks.model_checkpoint.dirpath)
            if checkpoint_path.is_dir():
                checkpoint_path /= "last.ckpt"
            # DeepSpeed's checkpoint is a directory, not a file
            if checkpoint_path.is_file() or checkpoint_path.is_dir():
                ckpt_cfg = {"ckpt_path": str(checkpoint_path)}
            else:
                log.info(
                    f"Checkpoint file {str(checkpoint_path)} not found. Will start training from scratch"
                )
        except KeyError:
            pass

    # Configure ddp automatically
    n_devices = cfg.trainer.get("devices", 1)
    if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
        n_devices = len(n_devices)
    if n_devices > 1 and cfg.trainer.get("strategy", None) is None:
        cfg.trainer.strategy = dict(
            _target_="pytorch_lightning.strategies.DDPStrategy",
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
            # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
        )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(model, datamodule=datamodule)

    log.info("Training finished! Testing...")
    if cfg.get("logger", None) is not None:
        trainer_kwargs = {"ckpt_path": cfg.trainer.get("testing_ckpt_path", "best")}
    else:
        trainer_kwargs = {}
    trainer.test(model, datamodule=datamodule, **trainer_kwargs)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.close_loggers()


if __name__ == "__main__":
    main()
