from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import seaborn as sns
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.metrics import confusion_matrix
import torch
import kwcoco

try:
    from aim import Image
except ImportError:
    Image = None


def create_video_frame_gt_preds(
    all_targets: torch.Tensor,
    all_preds: torch.Tensor,
    all_source_vids: torch.Tensor,
    all_source_frames: torch.Tensor,
) -> Dict[int, Dict[int, Tuple[int, int]]]:
    """
    Create a two-layer mapping from video ID to frame ID to pair of (gt, pred)
    class IDs.

    :param all_targets: Tensor of all target window class IDs.
    :param all_preds: Tensor of all predicted window class IDs.
    :param all_source_vids: Tensor of video IDs for the window.
    :param all_source_frames: Tensor of video frame number for the final
        frame of windows.

    :return: New mapping.
    """
    per_video_frame_gt_preds = defaultdict(dict)
    for (gt, pred, source_vid, source_frame) in zip(
        all_targets, all_preds, all_source_vids, all_source_frames
    ):
        per_video_frame_gt_preds[source_vid.item()][source_frame.item()] = (gt.item(), pred.item())
    return per_video_frame_gt_preds


def plot_gt_vs_preds(
    output_dir: Path,
    per_video_frame_gt_preds: Dict[int, Dict[int, Tuple[int, int]]],
    epoch: int,
    split="train",
    max_items=np.inf,
) -> None:
    """
    Plot activity classification truth and predictions through the course of a
    video's frames as lines.

    Successive calls to this function will overwrite any images in the given
    output directory for the given split.

    :param output_dir: Base directory into which to save plots.
    :param per_video_frame_gt_preds: Mapping of video-to-frames-to-tuple, where
        the tuple is the (gt, pred) pair of class IDs for the respective frame.
    :param epoch: The current epoch number for the plot title.
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
            title=f"{split} Step Prediction Per Frame (Epoch {epoch})",
            xlabel="Index",
            ylabel="Step",
        )
        sns.lineplot(x=inds, y=gt, color="magenta", label="GT", ax=ax, linewidth=3)

        ax.legend()
        root_dir = output_dir / "steps_vs_preds"
        root_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(root_dir / f"{split}_vid{video:03d}.jpg", pad_inches=5)
        plt.close(fig)


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

        #
        # Plot per-video class predictions vs. GT across progressive frames in
        # that video.
        #
        plot_gt_vs_preds(
            self.output_dir,
            create_video_frame_gt_preds(
                all_targets,
                all_preds,
                all_source_vids,
                all_source_frames
            ),
            epoch=current_epoch,
            split="train",
        )

        #
        # Create confusion matrix
        #
        class_ids = np.arange(all_probs.shape[-1])
        cm = confusion_matrix(
            all_targets.cpu().numpy(),
            all_preds.cpu().numpy(),
            labels=class_ids,
            normalize="true",
        )

        num_classes = len(class_ids)
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
        curr_f1 = pl_module.val_f1.compute()
        best_f1 = pl_module.val_f1_best.compute()

        #
        # Create confusion matrix
        #
        class_ids = np.arange(all_probs.shape[-1])
        cm = confusion_matrix(
            all_targets.numpy(),
            all_preds.numpy(),
            labels=class_ids,
            normalize="true",
        )

        num_classes = len(class_ids)
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

        if curr_f1 >= best_f1:
            fig.savefig(
                self.output_dir
                / f"confusion_mat_val_epoch{current_epoch:04d}_acc_{curr_acc:.4f}_f1_{curr_f1:.4f}.jpg",
                pad_inches=5,
            )
            #
            # Plot per-video class predictions vs. GT across progressive frames in
            # that video.
            #
            plot_gt_vs_preds(
                self.output_dir,
                create_video_frame_gt_preds(
                    all_targets,
                    all_preds,
                    all_source_vids,
                    all_source_frames
                ),
                epoch=current_epoch,
                split="validation",
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
        self._preds_dset_output_fpath = self.output_dir / "tcn_activity_predictions.kwcoco.json"

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

        # Create activity predictions KWCOCO JSON
        truth_dset_fpath = trainer.datamodule.hparams["coco_test_activities"]
        truth_dset = kwcoco.CocoDataset(truth_dset_fpath)
        acts_dset = kwcoco.CocoDataset()
        acts_dset.fpath = self._preds_dset_output_fpath
        acts_dset.dataset['videos'] = truth_dset.dataset['videos']
        acts_dset.dataset['images'] = truth_dset.dataset['images']
        acts_dset.dataset['categories'] = truth_dset.dataset['categories']
        acts_dset.index.build(acts_dset)
        # Create numpy lookup tables
        for i in range(len(all_preds)):
            frame_index = all_source_frames[i].item()
            video_id = all_source_vids[i].item()
            # Now get the image_id that matches the frame_index and video_id.
            sorted_img_ids_for_one_video = acts_dset.index.vidid_to_gids[int(video_id)]
            image_id = sorted_img_ids_for_one_video[frame_index]
            # Sanity check: this image_id corresponds to the frame_index and video_id
            assert acts_dset.index.imgs[image_id]['frame_index'] == frame_index
            assert acts_dset.index.imgs[image_id]['video_id'] == video_id

            acts_dset.add_annotation(
                image_id=image_id,
                category_id=all_preds[i].item(),
                score=all_probs[i][all_preds[i]].item(),
                prob=all_probs[i].numpy().tolist(),
            )  
        print(f"Dumping activities file to {acts_dset.fpath}")
        acts_dset.dump(acts_dset.fpath, newlines=True)  


        #
        # Plot per-video class predictions vs. GT across progressive frames in
        # that video.
        #
        plot_gt_vs_preds(
            self.output_dir,
            create_video_frame_gt_preds(
                all_targets,
                all_preds,
                all_source_vids,
                all_source_frames
            ),
            epoch=current_epoch,
            split="test",
        )

        # Built a COCO dataset of test results to output.
        # TODO: Configure activity test COCO file as input.
        #       Align outputs to videos/frames based on vid/frame_index combos

        #
        # Create confusion matrix
        #
        class_ids = np.arange(all_probs.shape[-1])
        cm = confusion_matrix(
            all_targets.cpu().numpy(),
            all_preds.cpu().numpy(),
            labels=class_ids,
            normalize="true",
        )

        num_classes = len(class_ids)
        fig, ax = plt.subplots(figsize=(num_classes, num_classes))

        sns.heatmap(cm, annot=True, ax=ax, fmt=".2f", linewidth=0.5, vmin=0, vmax=1)

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
