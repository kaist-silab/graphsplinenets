# Sequence graph training
import sys; sys.path.append('.')
import pytorch_lightning as pl
from src.datamodule import Gaussian2DSequence
from src.model import TimeSplineNets
from src.utils.logger import setup_logger
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Setup
datamodule = Gaussian2DSequence('data/gaussian.bin', 13, 1, 1)
model = TimeSplineNets(
    sequence_length=13,
    dim=2
)
logger = setup_logger(
    name='sgn-sq13-dim2',
    project_name='GraphSplineNets',
    use_wandb=False,
)
trainer = pl.Trainer(
    max_epochs=10,
    logger=logger,
    devices=1,
    accelerator='gpu',
    detect_anomaly=True
)

# Run
trainer.fit(model, datamodule)
