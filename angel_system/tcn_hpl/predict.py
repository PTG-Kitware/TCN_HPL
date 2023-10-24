from dataclasses import dataclass
import os
from typing import List
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import lightning.fabric.utilities.seed
import numpy as np
import numpy.typing as npt
import torch

from tcn_hpl.data.components.augmentations import NormalizePixelPts
from tcn_hpl.models.ptg_module import PTGLitModule

from angel_system.impls.detect_activities.detections_to_activities.utils import (
    tlbr_to_xywh,
    obj_det2d_set_to_feature,
)


def load_module(checkpoint_file, label_mapping_file, torch_device) -> PTGLitModule:
    """

    :param checkpoint_file:
    :param label_mapping_file:
    :param torch_device:
    :return:
    """
    # # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # lightning.fabric.utilities.seed.seed_everything(12345)

    mapping_file_dir = os.path.abspath(os.path.dirname(label_mapping_file))
    mapping_file_name = os.path.basename(label_mapping_file)
    model_device = torch.device(torch_device)
    model = PTGLitModule.load_from_checkpoint(
        checkpoint_file,
        map_location=model_device,
        # HParam overrides
        data_dir=mapping_file_dir,
        mapping_file_name=mapping_file_name,
    )

    return model


@dataclass
class ObjectDetectionsLTRB:
    """
    Expected object detections format for a single frame from the ROS2
    ecosystem.
    """

    left: Tuple[float]
    top: Tuple[float]
    right: Tuple[float]
    bottom: Tuple[float]
    labels: Tuple[str]
    confidences: Tuple[float]


def normalize_detection_features(
    det_feats: npt.ArrayLike,
    feat_version: int,
    img_width: int,
    img_height: int,
    num_det_classes: int,
) -> None:
    """
    Normalize input object detection descriptor vectors, outputting new vectors
    of the same shape.

    Expecting input `det_feats` to be in the shape `[window_size, num_feats]'.

    NOTE: This method normalizes in-place, so be sure to clone the input array
    if that is not desired.

    :param det_feats: Object Detection features to be normalized.

    :return: Normalized object detection features.
    """
    # This method is known to normalize in-place.
    # Shape [window_size, n_feats]
    NormalizePixelPts(img_width, img_height, num_det_classes, feat_version)(det_feats)


def objects_to_feats(
    frame_object_detections: Sequence[Optional[ObjectDetectionsLTRB]],
    det_label_to_idx: Dict[str, int],
    feat_version: int,
    image_width: int,
    image_height: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert some object detections for some window of frames into a feature
    vector of version requested.

    :param frame_object_detections: Sequence of object detections for some
        window of frames. The window size is dictated by this length of this
        sequence. Some frame "slots" may be None to indicate there were no
        object detections for that frame.
    :param det_label_to_idx: Mapping of object detector classes to the
        activity-classifier input index expectation.
    :param feat_version: Integer version of the feature vector to generate.
        See the `obj_det2d_set_to_feature` function for details.
    :param image_width: Integer pixel width of the image that object detections
        were generated on.
    :param image_height: Integer pixel height of the image that object
        detections were generated on.

    :raises ValueError: No non-None object detections in the given input
        window.

    :return: Window of normalized feature vectors for the given object
        detections (shape=[window_size, n_feats]), and an appropriate mask
        vector for use with activity classification (shape=[window_size]).
    """
    if all([d is None for d in frame_object_detections]):
        raise ValueError("No frames with detections in input.")

    window_size = len(frame_object_detections)
    # Shape [window_size, None|n_feats]
    feature_list: List[Optional[npt.NDArray]] = [None] * window_size
    feature_ndim = None
    feature_dtype = None
    for i, frame_dets in enumerate(frame_object_detections):
        frame_dets: ObjectDetectionsLTRB
        if frame_dets is not None:
            # the input message has tlbr, but obj_det2d_set_to_feature
            # requires xywh.
            xs, ys, ws, hs = tlbr_to_xywh(
                frame_dets.top,
                frame_dets.left,
                frame_dets.bottom,
                frame_dets.right,
            )
            feature_list[i] = (
                obj_det2d_set_to_feature(
                    frame_dets.labels,
                    xs,
                    ys,
                    ws,
                    hs,
                    frame_dets.confidences,
                    None,
                    None,
                    None,
                    None,
                    None,
                    det_label_to_idx,
                    version=feat_version,
                )
                .ravel()
                .astype(np.float32)
            )
            feature_ndim = feature_list[i].shape
            feature_dtype = feature_list[i].dtype
    # Already checked that we should have non-zero frames with detections above
    # so feature_ndim/_dtype should not be None at this stage
    assert feature_ndim is not None
    assert feature_dtype is not None

    # Create mask vector, which should indicate which window indices should not
    # be considered.
    # NOTE: The expected network is not yet trained to do this, so the mask is
    #       always 1's right now.
    # Shape [window_size]
    mask = torch.ones(window_size)

    # Fill in the canonical "empty" feature vector for those frames that had no
    # detections.
    empty_vec = np.zeros(shape=feature_ndim, dtype=feature_dtype)
    for i in range(window_size):
        if feature_list[i] is None:
            feature_list[i] = empty_vec

    # Shape [window_size, n_feats]
    feature_vec = torch.tensor(feature_list)

    # Normalize features
    # Shape [window_size, n_feats]
    normalize_detection_features(
        feature_vec, feat_version, image_width, image_height, len(det_label_to_idx)
    )

    return feature_vec, mask


def predict(
    model: PTGLitModule,
    window_feats: torch.Tensor,
    mask: torch.Tensor,
):
    """
    Compute model activity classifications, returning a tensor of softmax
    probabilities.

    We assume the input model and tensors are already on the appropriate
    device.

    We assume that input features normalized before being provided to this
    function. See :ref:`normalize_detection_features`.

    The "prediction" of this result can be determined via the `argmax`
    function::

        proba = predict(model, window_feats, mask)
        pred = torch.argmax(proba)

    :param model: PTGLitModule instance to use.
    :param window_feats: Window (sequence) of *normalized* object detection
        features. Shape: [window_size, feat_dim].
    :param mask: Boolean array indicating which frames of the input window for
        the network to consider. Shape: [window_size].

    :return: Probabilities (softmax) of the activity classes.
    """
    with torch.no_grad():
        logits = model(window_feats.T, mask[None, :])
    # Logits access mirrors model step function argmax access here:
    #   tcn_hpl.models.ptg_module --> PTGLitModule.model_step
    # ¯\_(ツ)_/¯
    return torch.softmax(logits[-1, :, :, -1], dim=1)[0]


###############################################################################
# Functions for debugging things in an interpreter
#
def windows_from_all_feature(
    all_features: npt.ArrayLike, window_size: int
) -> npt.ArrayLike:
    """
    Iterate over overlapping windows in the frame detections features given.

    :param all_features: All object detection feature vectors for all frames to
        consider. Shape: [n_frames, n_feats]
    :param window_size: Size of the window to slide.

    :return: Generator yielding different windows of feature vectors.
    """
    i = 0
    stride = 1
    while (i + window_size) < np.shape(all_features)[0]:
        yield all_features[i : (i + window_size), :]
        i += stride


def debug_from_array_file() -> None:
    import functools
    import re
    import numpy as np
    import torch
    from tqdm import tqdm
    from tcn_hpl.data.components.augmentations import NormalizePixelPts
    from angel_system.tcn_hpl.predict import (
        load_module,
        predict,
        windows_from_all_feature,
    )

    # Pre-computed, un-normalized features per-frame extracted from the
    # training harness, in temporally ascending order.
    # Shape = [n_frames, n_feats]
    all_features = torch.tensor(
        np.load("./model_files/all_activities_20.npy").astype(np.float32).T
    ).to("cuda")

    model = load_module(
        "./model_files/activity_tcn-coffee-checkpoint.ckpt",
        "./model_files/activity_tcn-coffee-mapping.txt",
        "cuda",
    ).eval()

    # Above model window size = 30
    mask = torch.ones(30).to("cuda")

    # Normalize features
    # The `objects_to_feats` above includes normalization along with the
    # bounding box conversion, so this needs to be applied explicitly outside
    # using `objects_to_feats` (though, using the same normalize func).
    norm_func = functools.partial(
        normalize_detection_features,
        feat_version=5,
        img_width=1280,
        img_height=720,
        num_det_classes=42,
    )

    # Shape [n_windows, window_size, n_feats]
    all_windows = list(windows_from_all_feature(all_features, 30))

    all_proba = list(
        tqdm(
            (predict(model, norm_func(w.clone()), mask) for w in all_windows),
            total=len(all_windows),
        )
    )

    all_preds_idx = np.asarray([int(torch.argmax(p)) for p in all_proba])
    all_preds_lbl = [model.classes[p] for p in all_preds_idx]

    # Load Hannah preds
    comparison_preds_file = "./model_files/all_activities_20_preds.txt"
    re_window_pred = re.compile(r"^gt: (\d+), pred: (\d+)$")
    comparison_gt = []
    comparison_preds_idx = []
    with open(comparison_preds_file) as infile:
        for l in infile.readlines():
            m = re_window_pred.match(l.strip())
            comparison_gt.append(int(m.groups()[0]))
            comparison_preds_idx.append(int(m.groups()[1]))
    comparison_preds_idx = np.asarray(comparison_preds_idx)

    ne_mask = all_preds_idx != comparison_preds_idx
    all_preds_idx[ne_mask], comparison_preds_idx[ne_mask]