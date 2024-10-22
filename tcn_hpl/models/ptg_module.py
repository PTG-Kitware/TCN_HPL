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
        # topic: str,  # medical or cooking
        # data_dir: str,
        # mapping_file_name: str = "mapping.txt",
        # output_dir: str = None,
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

        self.val_f1 = F1Score(
            num_classes=num_classes, average="none", task="multiclass"
        )
        self.test_f1 = F1Score(
            num_classes=num_classes, average="none", task="multiclass"
        )

        self.val_recall = Recall(
            num_classes=num_classes, average="none", task="multiclass"
        )
        self.test_recall = Recall(
            num_classes=num_classes, average="none", task="multiclass"
        )

        self.val_precision = Precision(
            num_classes=num_classes, average="none", task="multiclass"
        )
        self.test_precision = Precision(
            num_classes=num_classes, average="none", task="multiclass"
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.train_acc_best = MaxMetric()

        self.validation_step_outputs_prob = []
        self.validation_step_outputs_pred = []
        self.validation_step_outputs_target = []
        self.validation_step_outputs_source_vid = []
        self.validation_step_outputs_source_frame = []

        self.training_step_outputs_target = []
        self.training_step_outputs_source_vid = []
        self.training_step_outputs_source_frame = []
        self.training_step_outputs_pred = []
        self.training_step_outputs_prob = []

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
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of features, target labels, and mask.

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
        # print(f"logits: {logits.shape}")
        loss = torch.zeros((1)).to(x)
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

        # Why is this stage specifically only using these special index
        # conditions?
        # # print(f"preds: {preds.shape}, targets: {targets.shape}")
        # # print(f"mask: {mask.shape}, {mask[0,:]}")
        # ys = targets[:, -1]
        # # print(f"y: {ys.shape}")
        # # print(f"y: {ys}")
        # windowed_preds, windowed_ys = [], []
        # window_size = 15
        # center = 7
        # inds = []
        # for i in range(preds.shape[0] - window_size + 1):
        #     y = ys[i : i + window_size].tolist()
        #     # print(f"y: {y}")
        #     # print(f"len of set: {len(list(set(y)))}")
        #     if len(list(set(y))) == 1:
        #         inds.append(i + center - 1)
        #         windowed_preds.append(preds[i + center - 1])
        #         windowed_ys.append(ys[i + center - 1])
        #
        # windowed_preds = torch.tensor(windowed_preds).to(targets)
        # windowed_ys = torch.tensor(windowed_ys).to(targets)
        #
        # self.validation_step_outputs_target.append(targets[inds, -1])
        # self.validation_step_outputs_source_vid.append(source_vid[inds, -1])
        # self.validation_step_outputs_source_frame.append(source_frame[inds, -1])
        # self.validation_step_outputs_pred.append(preds[inds])
        # self.validation_step_outputs_prob.append(probs[inds])

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        all_preds = torch.cat([o['preds'] for o in outputs])
        all_targets = torch.cat([o['targets'] for o in outputs])

        acc = self.val_acc.compute()
        current_best_val_acc = self.val_acc_best.value
        # log `val_acc_best` as a value through `.compute()` return, instead of
        # as a metric object otherwise metric would be reset by lightning after
        # each epoch.
        best_val_acc = self.val_acc_best(acc)  # update best so far val acc

        if best_val_acc > current_best_val_acc:
            val_f1_score = self.val_f1(all_preds, all_targets)
            val_recall_score = self.val_recall(all_preds, all_targets)
            val_precision_score = self.val_precision(all_preds, all_targets)

            # print(f"preds: {all_preds}")
            # print(f"all_targets: {all_targets}")
            print(f"validation f1 score: {val_f1_score}")
            print(f"validation recall score: {val_recall_score}")
            print(f"validation precision score: {val_precision_score}")

        self.log("val/acc_best", best_val_acc, sync_dist=True, prog_bar=True)

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
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs_target.append(targets[:, -1])
        self.validation_step_outputs_source_vid.append(source_vid[:, -1])
        self.validation_step_outputs_source_frame.append(source_frame[:, -1])
        self.validation_step_outputs_pred.append(preds)
        self.validation_step_outputs_prob.append(probs)

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
