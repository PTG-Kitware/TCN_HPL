from typing import Any, Dict, Tuple
import os
import torch
from torch import nn
import torch.nn.functional as F

# from lightning import LightningModule
from pytorch_lightning import LightningModule


from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import F1Score, Recall, Precision
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import kwcoco
from statistics import mode

import einops

from angel_system.data.common.load_data import time_from_name

try:
    from aim import Image
except ImportError:
    Image = None


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
        topic: str,  # medical or cooking
        data_dir: str,
        num_classes: int,
        compile: bool,
        mapping_file_name: str = "mapping.txt",
        output_dir: str = None,
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

        # Don't popup figures
        plt.ioff()

        self.topic = topic

        # Get Action Names
        mapping_file = f"{self.hparams.data_dir}/{mapping_file_name}"
        actions_dict = dict()
        with open(mapping_file, "r") as file_ptr:
            actions = file_ptr.readlines()
            actions = [a.strip() for a in actions]  # drop leading/trailing whitespace
            for a in actions:
                parts = a.split()  # split on any number of whitespace
                actions_dict[parts[1]] = int(parts[0])

        self.class_ids = list(actions_dict.values())
        self.classes = list(actions_dict.keys())

        self.action_id_to_str = dict(zip(self.class_ids, self.classes))

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
        # self.val_acc_best = 0
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

        self.train_frames = None
        self.val_frames = None
        self.test_frames = None

    def plot_gt_vs_preds(self, per_video_frame_gt_preds, split="train", max_items=30):
        # fig = plt.figure(figsize=(10,15))

        for index, video in enumerate(per_video_frame_gt_preds.keys()):

            if index >= max_items:
                return

            fig, ax = plt.subplots(figsize=(15, 8))
            video_gt_preds = per_video_frame_gt_preds[video]
            frame_inds = sorted(list(video_gt_preds.keys()))
            preds, gt, inds = [], [], []
            for ind in frame_inds:
                inds.append(int(ind))
                gt.append(int(video_gt_preds[ind][0]))
                preds.append(int(video_gt_preds[ind][1]))

            sns.lineplot(
                x=inds,
                y=preds,
                linestyle="dotted",
                color="blue",
                linewidth=1,
                label="Pred",
                ax=ax,
            ).set(
                title=f"{split} Step Prediction Per Frame",
                xlabel="Index",
                ylabel="Step",
            )
            sns.lineplot(x=inds, y=gt, color="magenta", label="GT", ax=ax, linewidth=3)

            ax.legend()
            root_dir = f"{self.hparams.output_dir}/steps_vs_preds"

            if not os.path.exists(root_dir):
                os.makedirs(root_dir)

            fig.savefig(f"{root_dir}/{split}_{video}.png", pad_inches=5)
            plt.close()

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
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        x = x.transpose(2, 1)  # shape (batch size, feat dim, window)
        logits = self.forward(
            x, m
        )  # shape (4, batch size, self.hparams.num_classes, window))
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
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
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

        # self.log(
        #     "train/lr", self.lr_schedulers().state_dict()[[0], on_step=False, on_epoch=True, prog_bar=True
        # )

        self.training_step_outputs_target.append(targets[:, -1])
        self.training_step_outputs_source_vid.append(source_vid[:, -1])
        self.training_step_outputs_source_frame.append(source_frame[:, -1])
        self.training_step_outputs_pred.append(preds)
        self.training_step_outputs_prob.append(probs)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        # acc = self.train_acc.compute()  # get current val acc
        # self.train_acc_best(acc)  # update best so far val acc

        all_targets = torch.cat(self.training_step_outputs_target)  # shape: #frames
        all_preds = torch.cat(self.training_step_outputs_pred)  # shape: #frames
        all_probs = torch.cat(
            self.training_step_outputs_prob
        )  # shape (#frames, #act labels)
        all_source_vids = torch.cat(self.training_step_outputs_source_vid)
        all_source_frames = torch.cat(self.training_step_outputs_source_frame)

        # print(f"Training Per class occurences in GT: {torch.unique(all_targets, return_counts=True)}")

        if self.train_frames is None:
            self.train_frames = {}
            vid_list_file_train = f"{self.hparams.data_dir}/splits/train.split1.bundle"
            with open(vid_list_file_train, "r") as train_f:
                self.train_videos = train_f.read().split("\n")[:-1]

            for video in self.train_videos:
                # Load frame filenames for the video
                frame_list_file_train = f"{self.hparams.data_dir}/frames/{video}"
                with open(frame_list_file_train, "r") as train_f:
                    train_fns = train_f.read().split("\n")[:-1]

                self.train_frames[video[:-4]] = train_fns

        per_video_frame_gt_preds = {}

        for (gt, pred, source_vid, source_frame) in zip(
            all_targets, all_preds, all_source_vids, all_source_frames
        ):
            video_name = self.train_videos[int(source_vid)][:-4]

            if video_name not in per_video_frame_gt_preds.keys():
                per_video_frame_gt_preds[video_name] = {}

            frame = self.train_frames[video_name][int(source_frame)]
            frame_idx, time = time_from_name(frame, self.topic)

            per_video_frame_gt_preds[video_name][frame_idx] = (int(gt), int(pred))

            # print(f"video name: {video_name}, frame index: {frame_idx}, gt: {gt}, pred: {pred}")
            # dset.add_image(
            #     file_name=frame,
            #     video_id=vid,
            #     frame_index=frame_idx,
            #     activity_gt=int(gt),
            #     activity_pred=int(pred),
            #     activity_conf=prob.tolist()
            # )

        # print(f"per_video_frame_gt_preds: {per_video_frame_gt_preds}")
        # exit()

        self.plot_gt_vs_preds(per_video_frame_gt_preds, split="train")

        cm = confusion_matrix(
            all_targets.cpu().numpy(),
            all_preds.cpu().numpy(),
            labels=self.class_ids,
            normalize="true",
        )

        num_act_classes = len(self.class_ids)
        fig, ax = plt.subplots(figsize=(num_act_classes, num_act_classes))

        sns.heatmap(cm, annot=True, ax=ax, fmt=".2f", linewidth=0.5, vmin=0, vmax=1)

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"CM Training Epoch {self.current_epoch}")
        ax.xaxis.set_ticklabels(self.classes, rotation=25)
        ax.yaxis.set_ticklabels(self.classes, rotation=0)

        self.logger.experiment.track(Image(fig), name=f"CM Training Epoch")

        fig.savefig(f"{self.hparams.output_dir}/confusion_mat_train.png", pad_inches=5)

        plt.close(fig)

        self.training_step_outputs_target.clear()
        self.training_step_outputs_source_vid.clear()
        self.training_step_outputs_source_frame.clear()
        self.training_step_outputs_pred.clear()
        self.training_step_outputs_prob.clear()

        # pass

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        _, _, mask, _, _ = batch
        loss, probs, preds, targets, source_vid, source_frame = self.model_step(batch)
        # update and log metrics
        self.val_loss(loss)

        # print(f"preds: {preds.shape}, targets: {targets.shape}")
        # print(f"mask: {mask.shape}, {mask[0,:]}")
        ys = targets[:, -1]
        # print(f"y: {ys.shape}")
        # print(f"y: {ys}")
        windowed_preds, windowed_ys = [], []
        window_size = 15
        center = 7
        inds = []
        for i in range(preds.shape[0] - window_size + 1):
            y = ys[i : i + window_size].tolist()
            # print(f"y: {y}")
            # print(f"len of set: {len(list(set(y)))}")
            if len(list(set(y))) == 1:
                inds.append(i + center - 1)
                windowed_preds.append(preds[i + center - 1])
                windowed_ys.append(ys[i + center - 1])

        windowed_preds = torch.tensor(windowed_preds).to(targets)
        windowed_ys = torch.tensor(windowed_ys).to(targets)

        self.val_acc(preds, targets[:, -1])

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.validation_step_outputs_target.append(targets[inds, -1])
        self.validation_step_outputs_source_vid.append(source_vid[inds, -1])
        self.validation_step_outputs_source_frame.append(source_frame[inds, -1])
        self.validation_step_outputs_pred.append(preds[inds])
        self.validation_step_outputs_prob.append(probs[inds])

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc

        all_targets = torch.cat(self.validation_step_outputs_target)  # shape: #frames
        all_preds = torch.cat(self.validation_step_outputs_pred)  # shape: #frames
        all_probs = torch.cat(
            self.validation_step_outputs_prob
        )  # shape (#frames, #act labels)
        all_source_vids = torch.cat(self.validation_step_outputs_source_vid)
        all_source_frames = torch.cat(self.validation_step_outputs_source_frame)

        if self.current_epoch >= 15:
            current_best_val_acc = self.val_acc_best.compute()
            self.val_acc_best(acc)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            best_val_acc = self.val_acc_best.compute()

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

        # Load val vidoes
        if self.val_frames is None:
            self.val_frames = {}
            vid_list_file_val = f"{self.hparams.data_dir}/splits/val.split1.bundle"
            with open(vid_list_file_val, "r") as val_f:
                self.val_videos = val_f.read().split("\n")[:-1]

            for video in self.val_videos:
                # Load frame filenames for the video
                frame_list_file_val = f"{self.hparams.data_dir}/frames/{video}"
                with open(frame_list_file_val, "r") as val_f:
                    val_fns = val_f.read().split("\n")[:-1]

                self.val_frames[video[:-4]] = val_fns

        # Save results
        dset = kwcoco.CocoDataset()
        dset.fpath = f"{self.hparams.output_dir}/val_activity_preds_epoch{self.current_epoch}.mscoco.json"
        dset.dataset["info"].append({"activity_labels": self.action_id_to_str})

        # print(f"video_lookup: {video_lookup}")
        per_video_frame_gt_preds = {}

        for (gt, pred, prob, source_vid, source_frame) in zip(
            all_targets, all_preds, all_probs, all_source_vids, all_source_frames
        ):
            video_name = self.val_videos[int(source_vid)][:-4]

            if video_name not in per_video_frame_gt_preds.keys():
                per_video_frame_gt_preds[video_name] = {}

            video_lookup = dset.index.name_to_video
            vid = (
                video_lookup[video_name]["id"]
                if video_name in video_lookup
                else dset.add_video(name=video_name)
            )

            frame = self.val_frames[video_name][int(source_frame)]
            frame_idx, time = time_from_name(frame, self.topic)

            per_video_frame_gt_preds[video_name][frame_idx] = (int(gt), int(pred))

            # print(f"video name: {video_name}, frame index: {frame_idx}, gt: {gt}, pred: {pred}")

            dset.add_image(
                file_name=frame,
                video_id=vid,
                frame_index=frame_idx,
                activity_gt=int(gt),
                activity_pred=int(pred),
                activity_conf=prob.tolist(),
            )
        # dset.dump(dset.fpath, newlines=True)
        # print(f"Saved dset to {dset.fpath}")

        # print(f"per_video_frame_gt_preds: {per_video_frame_gt_preds}")
        self.plot_gt_vs_preds(per_video_frame_gt_preds, split="validation")
        # Create confusion matrix
        cm = confusion_matrix(
            all_targets.cpu().numpy(),
            all_preds.cpu().numpy(),
            labels=self.class_ids,
            normalize="true",
        )

        num_act_classes = len(self.class_ids)
        fig, ax = plt.subplots(figsize=(num_act_classes, num_act_classes))

        sns.heatmap(cm, annot=True, ax=ax, fmt=".2f", linewidth=0.5, vmin=0, vmax=1)

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"CM Validation Epoch {self.current_epoch}, Accuracy: {acc:.4f}")
        ax.xaxis.set_ticklabels(self.classes, rotation=25)
        ax.yaxis.set_ticklabels(self.classes, rotation=0)

        self.logger.experiment.track(Image(fig), name=f"CM Validation Epoch")

        if self.current_epoch >= 15:
            if acc >= best_val_acc:
                fig.savefig(
                    f"{self.hparams.output_dir}/confusion_mat_val_acc_{acc:.4f}.png",
                    pad_inches=5,
                )

        plt.close(fig)

        self.validation_step_outputs_target.clear()
        self.validation_step_outputs_source_vid.clear()
        self.validation_step_outputs_source_frame.clear()
        self.validation_step_outputs_pred.clear()
        self.validation_step_outputs_prob.clear()

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
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

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # update and log metrics

        all_targets = torch.cat(self.validation_step_outputs_target)  # shape: #frames
        all_preds = torch.cat(self.validation_step_outputs_pred)  # shape: #frames
        all_probs = torch.cat(
            self.validation_step_outputs_prob
        )  # shape (#frames, #act labels)
        all_source_vids = torch.cat(self.validation_step_outputs_source_vid)
        all_source_frames = torch.cat(self.validation_step_outputs_source_frame)

        # Load test vidoes
        if self.test_frames is None:
            self.test_frames = {}
            vid_list_file_tst = f"{self.hparams.data_dir}/splits/test.split1.bundle"
            with open(vid_list_file_tst, "r") as test_f:
                self.test_videos = test_f.read().split("\n")[:-1]

            for video in self.test_videos:
                # Load frame filenames for the video
                frame_list_file_tst = f"{self.hparams.data_dir}/frames/{video}"
                with open(frame_list_file_tst, "r") as test_f:
                    test_fns = test_f.read().split("\n")[:-1]

                self.test_frames[video[:-4]] = test_fns

        # Save results
        dset = kwcoco.CocoDataset()
        dset.fpath = f"{self.hparams.output_dir}/test_activity_preds.mscoco.json"
        dset.dataset["info"].append({"activity_labels": self.action_id_to_str})

        per_video_frame_gt_preds = {}
        for (gt, pred, prob, source_vid, source_frame) in zip(
            all_targets, all_preds, all_probs, all_source_vids, all_source_frames
        ):
            video_name = self.test_videos[int(source_vid)][:-4]

            if video_name not in per_video_frame_gt_preds.keys():
                per_video_frame_gt_preds[video_name] = {}

            video_lookup = dset.index.name_to_video
            vid = (
                video_lookup[video_name]["id"]
                if video_name in video_lookup
                else dset.add_video(name=video_name)
            )

            frame = self.test_frames[video_name][int(source_frame)]
            frame_idx, time = time_from_name(frame, self.topic)

            per_video_frame_gt_preds[video_name][frame_idx] = (int(gt), int(pred))

            dset.add_image(
                file_name=frame,
                video_id=vid,
                frame_index=frame_idx,
                activity_gt=int(gt),
                activity_pred=int(pred),
                activity_conf=prob.tolist(),
            )
        dset.dump(dset.fpath, newlines=True)
        print(f"Saved dset to {dset.fpath}")

        self.plot_gt_vs_preds(per_video_frame_gt_preds, split="test")
        # Create confusion matrix
        cm = confusion_matrix(
            all_targets.cpu().numpy(),
            all_preds.cpu().numpy(),
            labels=self.class_ids,
            normalize="true",
        )

        num_act_classes = len(self.class_ids)
        fig, ax = plt.subplots(figsize=(num_act_classes, num_act_classes))

        sns.heatmap(cm, annot=True, ax=ax, fmt=".2f", vmin=0, vmax=1)

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"CM Test Epoch {self.current_epoch}")
        ax.xaxis.set_ticklabels(self.classes, rotation=25)
        ax.yaxis.set_ticklabels(self.classes, rotation=0)

        fig.savefig(
            f"{self.hparams.output_dir}/confusion_mat_test_acc_{self.test_acc.compute():0.2f}.png",
            pad_inches=5,
        )

        self.logger.experiment.track(Image(fig), name=f"CM Test Epoch")

        plt.close(fig)

        self.validation_step_outputs_target.clear()
        self.validation_step_outputs_pred.clear()
        self.validation_step_outputs_prob.clear()
        self.validation_step_outputs_source_vid.clear()
        self.validation_step_outputs_source_frame.clear()

    def setup(self, stage: str) -> None:
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
