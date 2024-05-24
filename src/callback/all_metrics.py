from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

class AllMetricsLogger(Callback):
    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule, **kwargs
    ):
        if trainer.current_epoch % 10 == 0:
            pl_module.calculate_all_metrics()