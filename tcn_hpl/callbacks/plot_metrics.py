from collections import defaultdict
from pathlib import Path
import typing as ty
from typing import Any, Dict, Optional, Tuple

import kwcoco
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from debugpy.common.timestamp import current
from pytorch_lightning.callbacks import Callback
import seaborn as sns
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import confusion_matrix
import torch

try:
    from aim import Image
except ImportError:
    Image = None


def plot_gt_vs_preds(
    output_dir: Path,
    per_video_frame_gt_preds: ty.Dict[ty.Any, ty.Dict[int, ty.Tuple[int, int]]],
    split="train",
    max_items=30,
) -> None:
    """
    Plot activity classification truth and predictions through the course of a
    video's frames as lines.

    :param output_dir: Base directory into which to save plots.
    :param per_video_frame_gt_preds: Mapping of video-to-frames-to-tuple, where
        the tuple is the (gt, pred) pair of class IDs for the respective frame.
    :param split: Which train/val/test split the input is for. This will
        influence the names of files generated.
    :param max_items: Only consider the first N videos in the given
        `per_video_frame_gt_preds` structure.
    """
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
        root_dir = output_dir / "steps_vs_preds"
        root_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(root_dir / f"{split}_vid{video:03d}.jpg", pad_inches=5)
        plt.close()


class PlotMetrics(Callback):
    """
    Various on-stage-end plotting functionalities.

    This will currently only work with training a PTGLitModule due to metric
    access.

    Args:
        output_dir:
            Directory into which to output plots.
    """

    def __init__(
        self,
        output_dir: str,
    ):
        self.output_dir = Path(output_dir)

        # Don't popup figures
        plt.ioff()

        # Lists to cache train/val/test outputs across batches so that
        # on-epoch-end logic can make use of all computed values.
        # Contents should be cast to the CPU.
        self._train_all_preds = []
        self._train_all_probs = []
        self._train_all_targets = []
        self._train_all_source_vids = []
        self._train_all_source_frames = []

        self._val_all_preds = []
        self._val_all_probs = []
        self._val_all_targets = []
        self._val_all_source_vids = []
        self._val_all_source_frames = []

        # A flag to prevent validation related logic from running if no
        # training as occurred yet. It is known that lightning can perform a
        # "sanity check" which incurs a validation pass, for which we don't
        # care for plotting outputs for.
        self._has_begun_training = False

        # self.topic = topic
        #
        # # Get Action Names
        # mapping_file = f"{self.hparams.data_dir}/{mapping_file_name}"
        # actions_dict = dict()
        # with open(mapping_file, "r") as file_ptr:
        #     actions = file_ptr.readlines()
        #     actions = [a.strip() for a in actions]  # drop leading/trailing whitespace
        #     for a in actions:
        #         parts = a.split()  # split on any number of whitespace
        #         actions_dict[parts[1]] = int(parts[0])
        #
        # self.class_ids = list(actions_dict.values())
        # self.classes = list(actions_dict.keys())
        #
        # self.action_id_to_str = dict(zip(self.class_ids, self.classes))

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        self._train_all_preds.append(outputs["preds"].cpu())
        self._train_all_probs.append(outputs["probs"].cpu())
        self._train_all_targets.append(outputs["targets"].cpu())
        self._train_all_source_vids.append(outputs["source_vid"].cpu())
        self._train_all_source_frames.append(outputs["source_frame"].cpu())

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the train epoch ends."""
        self._has_begun_training = True
        all_preds = torch.cat(self._train_all_preds)  # shape: #frames
        all_targets = torch.cat(self._train_all_targets)  # shape: #frames
        all_probs = torch.cat(self._train_all_probs)  # shape (#frames, #act labels)
        all_source_vids = torch.cat(self._train_all_source_vids)  # shape: #frames
        all_source_frames = torch.cat(self._train_all_source_frames)  # shape: #frames

        current_epoch = pl_module.current_epoch
        curr_acc = pl_module.train_acc.compute()
        curr_f1 = pl_module.train_f1.compute()

        class_ids = np.arange(all_probs.shape[-1])
        num_classes = len(class_ids)

        #
        # Plot per-video class predictions vs. GT across progressive frames in
        # that video.
        #
        # Build up mapping of truth to preds for each video
        per_video_frame_gt_preds = defaultdict(dict)
        for (gt, pred, prob, source_vid, source_frame) in zip(
            all_targets, all_preds, all_probs, all_source_vids, all_source_frames
        ):
            per_video_frame_gt_preds[source_vid][source_frame] = (int(gt), int(pred))

        plot_gt_vs_preds(self.output_dir, per_video_frame_gt_preds, split="train")

        #
        # Create confusion matrix
        #
        cm = confusion_matrix(
            all_targets.cpu().numpy(),
            all_preds.cpu().numpy(),
            labels=class_ids,
            normalize="true",
        )

        fig, ax = plt.subplots(figsize=(num_classes, num_classes))

        sns.heatmap(cm, annot=True, ax=ax, fmt=".2f", linewidth=0.5, vmin=0, vmax=1)

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"CM Training Epoch {current_epoch}, Accuracy: {curr_acc:.4f}, F1: {curr_f1:.4f}")
        ax.xaxis.set_ticklabels(class_ids, rotation=25)
        ax.yaxis.set_ticklabels(class_ids, rotation=0)

        if Image is not None:
            pl_module.logger.experiment.track(Image(fig), name=f"CM Training Epoch")

        fig.savefig(
            self.output_dir / f"confusion_mat_train_epoch{current_epoch:04d}_acc_{curr_acc:.4f}_f1_{curr_f1:.4f}.jpg",
            pad_inches=5,
        )

        plt.close(fig)

        #
        # Clear local accumulators.
        #
        self._train_all_preds.clear()
        self._train_all_probs.clear()
        self._train_all_targets.clear()
        self._train_all_source_vids.clear()
        self._train_all_source_frames.clear()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Accumulate outputs for on-val-epoch-end usage."""
        self._val_all_preds.append(outputs["preds"].cpu())
        self._val_all_probs.append(outputs["probs"].cpu())
        self._val_all_targets.append(outputs["targets"].cpu())
        self._val_all_source_vids.append(outputs["source_vid"].cpu())
        self._val_all_source_frames.append(outputs["source_frame"].cpu())

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the val epoch ends."""
        if not self._has_begun_training:
            return

        all_preds = torch.cat(self._val_all_preds)  # shape: #frames
        all_targets = torch.cat(self._val_all_targets)  # shape: #frames
        all_probs = torch.cat(self._val_all_probs)  # shape (#frames, #act labels)
        all_source_vids = torch.cat(self._val_all_source_vids)  # shape: #frames
        all_source_frames = torch.cat(self._val_all_source_frames)  # shape: #frames

        current_epoch = pl_module.current_epoch
        curr_acc = pl_module.val_acc.compute()
        best_acc = pl_module.val_acc_best.compute()
        curr_f1 = pl_module.val_f1.compute()

        class_ids = np.arange(all_probs.shape[-1])
        num_classes = len(class_ids)

        #
        # Plot per-video class predictions vs. GT across progressive frames in
        # that video.
        #
        # Build up mapping of truth to preds for each video
        per_video_frame_gt_preds = defaultdict(dict)
        for (gt, pred, prob, source_vid, source_frame) in zip(
            all_targets, all_preds, all_probs, all_source_vids, all_source_frames
        ):
            per_video_frame_gt_preds[source_vid][source_frame] = (int(gt), int(pred))

        plot_gt_vs_preds(self.output_dir, per_video_frame_gt_preds, split="validation")

        #
        # Create confusion matrix
        #
        cm = confusion_matrix(
            all_targets.numpy(),
            all_preds.numpy(),
            labels=class_ids,
            normalize="true",
        )

        fig, ax = plt.subplots(figsize=(num_classes, num_classes))

        sns.heatmap(cm, annot=True, ax=ax, fmt=".2f", linewidth=0.5, vmin=0, vmax=1)

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"CM Validation Epoch {current_epoch}, Accuracy: {curr_acc:.4f}, F1: {curr_f1:.4f}")
        ax.xaxis.set_ticklabels(class_ids, rotation=25)
        ax.yaxis.set_ticklabels(class_ids, rotation=0)

        if Image is not None:
            pl_module.logger.experiment.track(Image(fig), name=f"CM Validation Epoch")

        if curr_acc >= best_acc:
            fig.savefig(
                self.output_dir
                / f"confusion_mat_val_epoch{current_epoch:04d}_acc_{curr_acc:.4f}_f1_{curr_f1:.4f}.jpg",
                pad_inches=5,
            )

        plt.close(fig)

        #
        # Clear local accumulators.
        #
        self._val_all_preds.clear()
        self._val_all_probs.clear()
        self._val_all_targets.clear()
        self._val_all_source_vids.clear()
        self._val_all_source_frames.clear()

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Called when the test batch ends."""
        # Re-using validation lists since test phase does not collide with
        # validation passes.
        self._val_all_preds.append(outputs["preds"].cpu())
        self._val_all_probs.append(outputs["probs"].cpu())
        self._val_all_targets.append(outputs["targets"].cpu())
        self._val_all_source_vids.append(outputs["source_vid"].cpu())
        self._val_all_source_frames.append(outputs["source_frame"].cpu())

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Called when the test epoch ends."""
        # Re-using validation lists since test phase does not collide with
        # validation passes.
        all_preds = torch.cat(self._val_all_preds)  # shape: #frames
        all_targets = torch.cat(self._val_all_targets)  # shape: #frames
        all_probs = torch.cat(self._val_all_probs)  # shape (#frames, #act labels)
        all_source_vids = torch.cat(self._val_all_source_vids)  # shape: #frames
        all_source_frames = torch.cat(self._val_all_source_frames)  # shape: #frames

        current_epoch = pl_module.current_epoch
        test_acc = pl_module.test_acc.compute()
        test_f1 = pl_module.test_f1.compute()

        class_ids = np.arange(all_probs.shape[-1])
        num_classes = len(class_ids)

        #
        # Plot per-video class predictions vs. GT across progressive frames in
        # that video.
        #
        # Build up mapping of truth to preds for each video
        per_video_frame_gt_preds = defaultdict(dict)
        for (gt, pred, prob, source_vid, source_frame) in zip(
            all_targets, all_preds, all_probs, all_source_vids, all_source_frames
        ):
            per_video_frame_gt_preds[source_vid][source_frame] = (int(gt), int(pred))

        plot_gt_vs_preds(self.output_dir, per_video_frame_gt_preds, split="test")

        # Built a COCO dataset of test results to output.
        # TODO: Configure activity test COCO file as input.
        #       Align outputs to videos/frames based on vid/frame_index combos

        #
        # Create confusion matrix
        #
        cm = confusion_matrix(
            all_targets.cpu().numpy(),
            all_preds.cpu().numpy(),
            labels=class_ids,
            normalize="true",
        )

        fig, ax = plt.subplots(figsize=(num_classes, num_classes))

        sns.heatmap(cm, annot=True, ax=ax, fmt=".2f", vmin=0, vmax=1)

        # labels, title and ticks
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        ax.set_title(f"CM Test Epoch {current_epoch}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        ax.xaxis.set_ticklabels(class_ids, rotation=25)
        ax.yaxis.set_ticklabels(class_ids, rotation=0)

        fig.savefig(
            self.output_dir / f"confusion_mat_test_acc_{test_acc:0.2f}_f1_{test_f1:.4f}.jpg",
            pad_inches=5,
        )

        if Image is not None:
            self.logger.experiment.track(Image(fig), name=f"CM Test Epoch")

        plt.close(fig)

        #
        # Clear local accumulators.
        #
        self._val_all_preds.clear()
        self._val_all_probs.clear()
        self._val_all_targets.clear()
        self._val_all_source_vids.clear()
        self._val_all_source_frames.clear()
