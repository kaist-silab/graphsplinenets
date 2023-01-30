import subprocess
from pathlib import Path

import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.utilities import rank_zero_only


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class WatchModel(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, log: str = "gradients", log_freq: int = 100):
        self.log = log
        self.log_freq = log_freq

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        logger.watch(model=trainer.model, log=self.log, log_freq=self.log_freq)


class UploadCodeAsArtifact(Callback):
    """Upload all code files to wandb as an artifact, at the beginning of the run."""

    def __init__(self, code_dir: str, use_git: bool = True):
        """
        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir
        self.use_git = use_git

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_wandb_logger(trainer=trainer)
        experiment = logger.experiment

        code = wandb.Artifact("project-source", type="code")

        if self.use_git:
            # get .git folder
            # https://alexwlchan.net/2020/11/a-python-function-to-ignore-a-path-with-git-info-exclude/
            git_dir_path = Path(
                subprocess.check_output(["git", "rev-parse", "--git-dir"]).strip().decode("utf8")
            ).resolve()

            for path in Path(self.code_dir).resolve().rglob("*"):
                if (
                    path.is_file()
                    # ignore files in .git
                    and not str(path).startswith(str(git_dir_path))  # noqa: W503
                    # ignore files ignored by git
                    and (  # noqa: W503
                        subprocess.run(["git", "check-ignore", "-q", str(path)]).returncode == 1
                    )
                ):
                    code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        else:
            for path in Path(self.code_dir).resolve().rglob("*.py"):
                code.add_file(str(path), name=str(path.relative_to(self.code_dir)))

        experiment.log_artifact(code)


class UploadCheckpointsAsArtifact(Callback):
    """Upload checkpoints to wandb as an artifact, at the end of run."""

    def __init__(self, ckpt_dir: str = "checkpoints/", upload_best_only: bool = False):
        self.ckpt_dir = ckpt_dir
        self.upload_best_only = upload_best_only

    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        self.on_train_end(trainer, pl_module)

    @rank_zero_only
    def on_train_end(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger):
            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment

            ckpts = wandb.Artifact("experiment-ckpts", type="checkpoints")

            if self.upload_best_only:
                ckpts.add_file(trainer.checkpoint_callback.best_model_path)
            else:
                for path in Path(self.ckpt_dir).rglob("*.ckpt"):
                    ckpts.add_file(str(path))

            experiment.log_artifact(ckpts)
        



#--------------------------------------------


class _WandbArtifactCallback(Callback):
    # From Junyoung; we shouldnt need this code anymore, leaving for legacy
    
    def __init__(self, wandb_run_path, save_top_K: int, mode: str = 'min', **kwargs):
        self.run_path = wandb_run_path
        self.run = wandb.Api().run(self.run_path)
        self.save_top_K = save_top_K
        self.mode = mode

    def on_validation_end(self, trainer, pl_module):
        artifacts = self.run.logged_artifacts()
        if len(artifacts) > self.save_top_K:
            scores = [artf.metadata['score'] for artf in artifacts]

            if self.mode == 'min':
                threshold = artifacts[np.argsort(scores)[:self.save_top_K][-1]].metadata['score']
            else:
                threshold = artifacts[np.argsort(scores)[::-1][:self.save_top_K][-1]].metadata['score']

            for artifact in artifacts:
                if self.mode == 'min':
                    delete_cond = artifact.metadata['score'] > threshold
                else:
                    delete_cond = artifact.metadata['score'] < threshold

                if delete_cond:
                    artifact.delete()
