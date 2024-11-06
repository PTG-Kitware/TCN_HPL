from typing import Any, Dict, Optional, Tuple, Union, List

from numpy.lib.utils import source
from pytorch_lightning import LightningModule
import torch
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import nn
import torch.nn.functional as F
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import F1Score, Recall, Precision

import einops


class PTGLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        smoothing_loss: float,
        use_smoothing_loss: bool,
        num_classes: int,
        compile: bool,
    ) -> None:
        """Initialize a `PTGLitModule`.

        :param net: The model to train.
        :param criterion: Loss Computation
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss functions
        self.criterion = criterion
        self.mse = nn.MSELoss(reduction="none")

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(
            task="multiclass", average="weighted", num_classes=num_classes
        )
        self.val_acc = Accuracy(
            task="multiclass", average="weighted", num_classes=num_classes
        )
        self.test_acc = Accuracy(
            task="multiclass", average="weighted", num_classes=num_classes
        )

        self.train_f1 = F1Score(
            num_classes=num_classes, average="weighted", task="multiclass"
        )
        self.val_f1 = F1Score(
            num_classes=num_classes, average="weighted", task="multiclass"
        )
        self.test_f1 = F1Score(
            num_classes=num_classes, average="weighted", task="multiclass"
        )

        self.train_recall = Recall(
            num_classes=num_classes, average="weighted", task="multiclass"
        )
        self.val_recall = Recall(
            num_classes=num_classes, average="weighted", task="multiclass"
        )
        self.test_recall = Recall(
            num_classes=num_classes, average="weighted", task="multiclass"
        )

        self.train_precision = Precision(
            num_classes=num_classes, average="weighted", task="multiclass"
        )
        self.val_precision = Precision(
            num_classes=num_classes, average="weighted", task="multiclass"
        )
        self.test_precision = Precision(
            num_classes=num_classes, average="weighted", task="multiclass"
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.train_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of features for frames.
        :param m: A tensor of mask of the valid frames.
        :return: A tensor of logits.
        """
        return self.net(x, m)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def compute_loss(self, p, y, mask):
        """Compute the total loss for a batch

        :param p: The prediction
        :param batch_target: The target labels
        :param mask: Marks valid input data

        :return: The loss
        """

        probs = torch.softmax(p, dim=1)  # shape (batch size, self.hparams.num_classes)
        preds = torch.argmax(probs, dim=1).float()  # shape: batch size

        loss = torch.zeros((1)).to(p[0])

        # TODO: Use only last frame per window

        loss += self.criterion(
            p.transpose(2, 1).contiguous().view(-1, self.hparams.num_classes),
            y.view(-1),
        )

        # loss += self.criterion(
        #     p[:,:,-1],
        #     y[:,-1],
        # )

        # need to penalize high volatility of predictions within a window
        mode, _ = torch.mode(y, dim=-1)
        mode = einops.repeat(mode, "b -> b c", c=preds.shape[-1])

        variation_coef = torch.abs(preds - mode)
        variation_coef = torch.sum(variation_coef, dim=-1)
        gt_variation_coef = torch.zeros_like(variation_coef)

        if self.hparams.use_smoothing_loss:
            loss += self.hparams.smoothing_loss * torch.mean(
                self.mse(
                    variation_coef,
                    gt_variation_coef,
                ),
            )

        loss += self.hparams.smoothing_loss * torch.mean(
            torch.clamp(
                self.mse(
                    F.log_softmax(p[:, :, 1:], dim=1),
                    F.log_softmax(p.detach()[:, :, :-1], dim=1),
                ),
                min=0,
                max=16,
            )
            * mask[:, None, 1:]
        )

        return loss

    def model_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        compute_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of features, target labels, and mask.
        :param compute_loss: Flag to enable or not the computation of the loss
            value. If this is False, the value of the loss term is undefined.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of probabilities per class
            - A tensor of predictions.
            - A tensor of target labels.
            - A tensor of [video name, frame idx]
        """
        x, y, m, source_vid, source_frame = batch
        # x shape: (batch size, window, feat dim)
        # y shape: (batch size, window)
        # m shape: (batch size, window)
        # source_vid shape: (batch size, window)
        # source_frame shape: (batch size, window)
        x = x.transpose(2, 1)  # shape (batch size, feat dim, window)
        logits = self.forward(
            x, m
        )  # shape (4, batch size, num_classes, window))
        loss = torch.zeros((1)).to(x)
        if compute_loss:
            for p in logits:
                loss += self.compute_loss(p, y, m)

        probs = torch.softmax(
            logits[-1, :, :, -1], dim=1
        )  # shape (batch size, self.hparams.num_classes)
        preds = torch.argmax(logits[-1, :, :, -1], dim=1)  # shape: batch size

        return loss, probs, preds, y, source_vid, source_frame

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> STEP_OUTPUT:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, probs, preds, targets, source_vid, source_frame = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets[:, -1])

        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        # return loss or backpropagation will fail
        return {
            "loss": loss,
            "preds": preds,
            "probs": probs,
            "targets": targets[:, -1],
            "source_vid": source_vid[:, -1],
            "source_frame": source_frame[:, -1],
        }

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        all_preds = torch.cat([o["preds"] for o in outputs])
        all_targets = torch.cat([o['targets'] for o in outputs])

        self.train_f1(all_preds, all_targets)
        self.train_recall(all_preds, all_targets)
        self.train_precision(all_preds, all_targets)
        self.log("train/f1", self.train_f1, prog_bar=True, on_epoch=True)
        self.log("train/recall", self.train_recall, prog_bar=True, on_epoch=True)
        self.log("train/precision", self.train_precision, prog_bar=True, on_epoch=True)

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> Optional[STEP_OUTPUT]:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, probs, preds, targets, source_vid, source_frame = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets[:, -1])

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        # Only retain the truth and source vid/frame IDs for the final window
        # frame as this is the ultimately relevant result.
        return {
            "loss": loss,
            "preds": preds,
            "probs": probs,
            "targets": targets[:, -1],
            "source_vid": source_vid[:, -1],
            "source_frame": source_frame[:, -1],
        }

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        all_preds = torch.cat([o['preds'] for o in outputs])
        all_targets = torch.cat([o['targets'] for o in outputs])

        acc = self.val_acc.compute()
        # log `val_acc_best` as a value through `.compute()` return, instead of
        # as a metric object otherwise metric would be reset by lightning after
        # each epoch.
        best_val_acc = self.val_acc_best(acc)  # update best so far val acc
        self.log("val/acc_best", best_val_acc, sync_dist=True, prog_bar=True)

        self.val_f1(all_preds, all_targets)
        self.val_recall(all_preds, all_targets)
        self.val_precision(all_preds, all_targets)
        self.log("val/f1", self.val_f1, prog_bar=True, on_epoch=True)
        self.log("val/recall", self.val_recall, prog_bar=True, on_epoch=True)
        self.log("val/precision", self.val_precision, prog_bar=True, on_epoch=True)

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> Optional[STEP_OUTPUT]:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, probs, preds, targets, source_vid, source_frame = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets[:, -1])
        self.test_f1(preds, targets[:, -1])
        self.test_recall(preds, targets[:, -1])
        self.test_precision(preds, targets[:, -1])
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)

        # Only retain the truth and source vid/frame IDs for the final window
        # frame as this is the ultimately relevant result.
        return {
            "loss": loss,
            "preds": preds,
            "probs": probs,
            "targets": targets[:, -1],
            "source_vid": source_vid[:, -1],
            "source_frame": source_frame[:, -1],
        }

    def setup(self, stage: Optional[str] = None) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = PTGLitModule(None, None, None, None)
