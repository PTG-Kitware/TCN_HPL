#!/usr/bin/env python3
"""
Generate bounding box detections, then generate poses for patients.
"""

from pathlib import Path
import warnings
from typing import Callable
from typing import List
from typing import Sequence
from typing import Set
from typing import Optional
from typing import Tuple

import click
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import kwcoco
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo
import numpy.typing as npt
import torch
from tqdm import tqdm

from tcn_hpl.data.utils.pose_generation.predictor import VisualizationDemo


warnings.filterwarnings("ignore")


# Expected classes in the input detection model in ascending index order.
# Yes this is hard-coded, not I don't like it.
DETECTION_CLASSES = [
    "patient",
    "user",
]


# Expected keypoints per detection class.
# A class not represented is not expected to have keypoints predicted.
# Also note that the keypoints are detailed in `self.pose_dataset_info.keypoint_info`.
DETECTION_CLASS_KEYPOINTS = {
    "patient": [
        "nose",
        "mouth",
        "throat",
        "chest",
        "stomach",
        "left_upper_arm",
        "right_upper_arm",
        "left_lower_arm",
        "right_lower_arm",
        "left_wrist",
        "right_wrist",
        "left_hand",
        "right_hand",
        "left_upper_leg",
        "right_upper_leg",
        "left_knee",
        "right_knee",
        "left_lower_leg",
        "right_lower_leg",
        "left_foot",
        "right_foot",
        "back",
    ]
}


def setup_detectron_cfg(
    config_filepath: str,
    config_opts: Sequence[str] = (),
    confidence_threshold: float = 0.8,
    model_checkpoint_filepath: str = None,
    device: str = "cuda",
):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_filepath)
    cfg.merge_from_list(list(config_opts))
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        confidence_threshold
    )
    if model_checkpoint_filepath is not None:
        cfg.MODEL.WEIGHTS = model_checkpoint_filepath
    if device is not None:
        cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


class PosesGenerator(object):
    """
    Controller to handle pose bbox and keypoint estimation.

    Args:
        det_config_file:
            Base configuration for the bbox detection model.
            E.g. `python-tpl/TCN_HPL/tcn_hpl/data/utils/pose_generation/configs/medic_pose.yaml`
        pose_config_file:
            Base configuration for the pose model.
        confidence_threshold:
            Optional confidence threshold to apply to detections.
        det_model_ckpt:
            Optional path to model weights to use.
            If not provided, uses the checkpoint specified in the input config.
        det_model_device:
            The device to utilize for inference processing of the detection
            model.
        pose_model_ckpt:
            Optional path to model weights to use.
            If not provided, uses the checkpoint specified in the input config.
        pose_model_device:
            The device to utilize for inference processing of the pose model.
        config_overrides:
            Optional sequence of strings providing alternating keys and values
            of configuration properties to override, as it related to the
            detection model configuration (detectron2).
    """

    def __init__(
        self,
        det_config_file: str,
        pose_config_file: str,
        confidence_threshold: float = 0.8,
        det_model_ckpt: Optional[str] = None,
        det_model_device: str = "cuda:0",
        pose_model_ckpt: Optional[str] = None,
        pose_model_device: str = "cuda:0",
        config_overrides: Optional[Sequence[str]] = (),
    ):
        # Only some classes should have keypoints predicted
        self.pose_pred_classes: Set[int] = {
            DETECTION_CLASSES.index(k) for k in DETECTION_CLASS_KEYPOINTS
        }

        detectron_cfg = setup_detectron_cfg(
            det_config_file,
            config_overrides,
            confidence_threshold,
            model_checkpoint_filepath=det_model_ckpt,
            device=det_model_device,
        )
        self.det_conf_thresh = confidence_threshold
        self.predictor = VisualizationDemo(detectron_cfg)

        self.pose_model = init_pose_model(
            pose_config_file,
            pose_model_ckpt,
            device=pose_model_device,
        )

        self.pose_dataset = self.pose_model.cfg.data["test"]["type"]
        self.pose_dataset_info = self.pose_model.cfg.data["test"].get(
            "dataset_info", None
        )
        if self.pose_dataset_info is None:
            warnings.warn(
                "Please set `dataset_info` in the config."
                "Check https://github.com/open-mmlab/mmpose/pull/663 for details.",
                DeprecationWarning,
            )
        else:
            self.pose_dataset_info = DatasetInfo(self.pose_dataset_info)

    def predict_single(
        self,
        image: npt.NDArray,
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, List[Optional[npt.NDArray]]]:
        """
        Predict boxes and keypoints for the given image.

        Args:
             image:
                BGR Image matrix to predict over.

        Returns:
            Detected bounding boxes, their respective scores, their predicted
            class IDs (integers) and optional 2D keypoint locations in pixel
            coordinates with confidences.
            There may be ``None`` values in the keypoint list for detections
            that are of a class for which keypoints are not applicable.
            Boxes returned are in xyxy (left, top, right, bottom) format.
            Key-point matrices, when output, should be of shape n_joints x 3,
            where each row is [x, y, score].
        """
        predictions, _ = self.predictor.run_on_image(image)
        instances = predictions["instances"].to("cpu")
        boxes: npt.NDArray = (
            instances.pred_boxes.tensor.numpy()
            if instances.has("pred_boxes")
            else torch.Tensor(0, 4).numpy()
        )
        scores: npt.NDArray = (
            instances.scores.numpy()
            if instances.has("scores")
            else torch.Tensor(0).numpy()
        )
        classes: npt.NDArray = (
            instances.pred_classes.numpy()
            if instances.has("pred_classes")
            else torch.Tensor(0).numpy()
        )

        # List for storing keypoints matrices for detections that satisfy the
        # requirement to have them (i.e. be patients). A None should be
        # inserted for boxes which are not the appropriate class.
        keypoints_list: List[Optional[npt.NDArray]] = []

        if boxes is not None:
            for bbox, score, cls_idx in zip(boxes, scores, classes):
                # Only predict poses for those classes that support it.
                if cls_idx in self.pose_pred_classes:
                    person_results = [{"bbox": bbox}]
                    pose_results, returned_outputs = inference_top_down_pose_model(
                        model=self.pose_model,
                        img_or_path=image,
                        person_results=person_results,
                        bbox_thr=None,
                        format="xyxy",
                        dataset=self.pose_dataset,
                        dataset_info=self.pose_dataset_info,
                        return_heatmap=False,
                        outputs=["backbone"],
                    )
                    keypoints_list.append(pose_results[0]['keypoints'])
                else:
                    keypoints_list.append(None)

        return boxes, scores, classes, keypoints_list

    def predict_coco(
        self,
        dset: kwcoco.CocoDataset,
        on_image_done_callback: Optional[Callable[[kwcoco.CocoDataset], None]] = None,
    ) -> kwcoco.CocoDataset:
        """
        Generates a CocoDataset with bounding box (bbs) and pose annotations generated from the dataset's images.
        This method processes each image, detects bounding boxes and classifies them into 'patient' or 'user' categories,
        and performs pose estimation on 'patient' detections. Annotations are added to the dataset, including bounding
        box coordinates, category IDs, and, for patients, pose keypoints.

        Arguments:
            dset:
                The dataset specifying videos/images to generate poses for,
                which must be an instance of `kwcoco.CocoDataset`.
            on_image_done_callback:
                An optional function that when provided is called after
                finishing prediction for an image, and given the current state
                of the output COCO dataset instance.
                This function should not alter the given dataset and is
                expected to write it out somewhere user-specific.
                This is useful for long-running jobs to prevent data loss and
                to track progress.

        Returns:
            The input dataset, now added with additional annotations for
            bounding boxes and pose keypoints where applicable.

        Notes:
            - The bounding box and pose estimation models are assumed to be
              accessible via `self.predictor` and `self.pose_model`,
              respectively. These models must be properly configured before
              calling this method.
            - The method uses a progress bar to indicate processing progress
              through the dataset's images.
            - This function automatically handles the categorization of
              detections into 'patient' and 'user' based on the model's
              predictions and performs pose estimation only on 'patient'
              detections.
            - Save intervals for the intermediate dataset dumps can be adjusted
              based on the dataset size and processing time per image to
              balance between progress tracking and performance.
            - The `kwcoco.CocoDataset` class is part of the `kwcoco` package,
              offering structured management of COCO-format datasets, including
              easy addition of annotations and categories, and saving/loading
              datasets.
        """
        # Output dataset to populate
        out_dset = kwcoco.CocoDataset()
        # Carry forward video/image data from input dataset that will be
        # predicted over.
        out_dset.dataset['videos'] = dset.dataset['videos']
        out_dset.dataset['images'] = dset.dataset['images']
        out_dset.index.build(out_dset)
        # Equality can later be tested with:
        #   guiding_dset.index.videos == dset.index.videos
        #   guiding_dset.index.imgs == dset.index.imgs
        # Remove reference to dset so we don't accidentally do things to it
        # here.
        del dset

        # Populate the categories.
        class_to_id = {}
        for cls_id, cls_label in enumerate(DETECTION_CLASSES):
            kw = {}
            if cls_label in DETECTION_CLASS_KEYPOINTS:
                kw["keypoints"] = DETECTION_CLASS_KEYPOINTS[cls_label]
            class_to_id[cls_label] = out_dset.ensure_category(
                name=cls_label,
                id=cls_id,
                **kw,
            )

        for img_id in tqdm(
            out_dset.images(),
            desc="Processing images",
            unit="images",
        ):
            img_path = Path(out_dset.get_image_fpath(img_id)).as_posix()
            img = read_image(img_path, format="BGR")
            boxes, scores, classes, keypoints_list = self.predict_single(img)

            # We will need non-numpy data types to insert into the structure to
            # follow JSON compliance.
            boxes_list = boxes.tolist()
            scores_list = scores.tolist()
            classes_list = classes.tolist()

            # Construct annotations for predictions.
            for box, score, cls_idx, kp_mat in zip(
                boxes_list, scores_list, classes_list, keypoints_list
            ):
                # Convert keypoints from scored XY coordinates to the COCO
                # notation with visibility.
                kp_kw = {}
                if kp_mat is not None:
                    kp_vals = []
                    # According to spec (https://cocodataset.org/#format-data):
                    # visibility flag v defined as v=0: not labeled (in
                    # which case x=y=0), v=1: labeled but not visible, and
                    # v=2: labeled and visible.
                    for kp in kp_mat.tolist():
                        # TODO: Filter keypoints if present by a threshold?
                        #       if not above threshold, fill in triple-0.
                        kp_vals.extend([*kp[:2], 2])
                    kp_kw["keypoints"] = kp_vals
                    kp_kw["keypoint_scores"] = kp_mat[:, 2].ravel().tolist()

                out_dset.add_annotation(
                    image_id=img_id,
                    category_id=cls_idx,
                    bbox=box,
                    score=score,
                    **kp_kw,
                )

            # Checkpoint output the COCO Dataset if we are at some interval
            if on_image_done_callback is not None:
                on_image_done_callback(out_dset)

        return out_dset


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-i", "--input-coco", "input_coco_filepath",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The input COCO dataset file containing the videos/images to be processed.",
)
@click.option(
    "-o", "--output-coco", "output_coco_filepath",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="The output COCO dataset file where results will be saved.",
)
@click.option(
    "--det-config", "detector_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The config file for the detector to use.",
)
@click.option(
    "--det-weights", "detector_weights_filepath",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The weights file for the detector to use.",
)
@click.option(
    "--pose-config", "pose_config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The config file for the pose estimator to use.",
)
@click.option(
    "--pose-weights", "pose_weights_filepath",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The weights file for the pose estimator to use.",
)
@click.option(
    "--det-device", "detector_device",
    default="cuda:0",
    show_default=True,
    help="The device on which to run the detector (e.g., 'cpu', 'cuda:0')."
)
@click.option(
    "--pose-device", "pose_device",
    default="cuda:0",
    show_default=True,
    help="The device on which to run the pose estimator (e.g., 'cpu', 'cuda:0')."
)
@click.option(
    "--conf-thresh", "confidence_threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="The confidence threshold to use for the bounding-box detector."
)
@click.option(
    "--checkpoint-interval", "checkpoint_interval",
    type=int,
    show_default=True,
    help=(
        "How often (in frames) to save a checkpoint of the pose estimator's "
        "output."
    ),
)
# TODO: Keypoint confidence threshold.
def main(
    input_coco_filepath: Path,
    output_coco_filepath: Path,
    detector_config: Path,
    detector_weights_filepath: Path,
    pose_config: Path,
    pose_weights_filepath: Path,
    detector_device: str,
    pose_device: str,
    confidence_threshold: float,
    checkpoint_interval: Optional[int],
):
    """
    Predict poses for objects in videos/images specified by the input COCO
    dataset.

    Expected use-case: generate object detections for video frames (images)
    that we have activity classification truth for.

    Non-background model classes will be assigned IDs starting with 0.

    \b
    Example:
        python-tpl/TCN_HPL/tcn_hpl/data/utils/pose_generation/generate_pose_data.py \\
            -i ~/data/darpa-ptg/tcn_training_example/activity_truth.coco.json \\
            -o ./test_pose_preds.coco.json \\
            --det-config ./python-tpl/TCN_HPL/tcn_hpl/data/utils/pose_generation/configs/medic_pose.yaml \\
            --det-weights ./model_files/pose_estimation/pose_det_model.pth \\
            --pose-config python-tpl/TCN_HPL/tcn_hpl/data/utils/pose_generation/configs/ViTPose_base_medic_casualty_256x192.py \\
            --pose-weights ./model_files/pose_estimation/pose_model.pth
    """
    input_dset = kwcoco.CocoDataset(input_coco_filepath)


    img_done_cb = None
    if checkpoint_interval is not None:
        imgs_processed = 0

        def img_done_cb(out_dset: kwcoco.CocoDataset) -> None:
            nonlocal imgs_processed
            imgs_processed += 1
            if (imgs_processed % checkpoint_interval) == 0:
                intermediate_out_path = (
                    output_coco_filepath.parent / output_coco_filepath.with_suffix(f".{imgs_processed}{output_coco_filepath.suffix}")
                )
                intermediate_out_path.parent.mkdir(parents=True, exist_ok=True)
                out_dset.dump(intermediate_out_path, newlines=True)
                print(
                    f"Saved intermediate dataset at index {imgs_processed} to: "
                    f"{intermediate_out_path}"
                )

    pg = PosesGenerator(
        det_config_file=detector_config.as_posix(),
        pose_config_file=pose_config.as_posix(),
        confidence_threshold=confidence_threshold,
        det_model_ckpt=detector_weights_filepath.as_posix(),
        det_model_device=detector_device,
        pose_model_ckpt=pose_weights_filepath.as_posix(),
        pose_model_device=pose_device,
    )
    output_dset = pg.predict_coco(input_dset, img_done_cb)
    output_dset.dump(
        output_coco_filepath,
        newlines=True,
    )


if __name__ == "__main__":
    main()
