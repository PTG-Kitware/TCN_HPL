"""
Logic and utilities to perform the vectorization of input data into an
embedding space used for TCN training and prediction.
"""
import functools
from dataclasses import dataclass
import typing as tg

import numpy as np
import numpy.typing as npt
from jedi.plugins.stdlib import functools_partial

from tcn_hpl.data.vectorize_classic import (
    obj_det2d_set_to_feature,
    zero_joint_offset,
)


@dataclass
class FrameObjectDetections:
    """
    Representation of object detection predictions for a single image.
    All sequences should be the same length.
    """

    # Detection 2D bounding boxes in xywh format for the left, width, height and
    # bottom coordinates respectively. Shape: (num_detections, 4)
    boxes: npt.NDArray[float]
    # Object category ID of the most confident class. Shape: (num_detections,)
    labels: npt.NDArray[int]
    # Vectorized detection confidence value of the most confidence class.
    # Shape: (num_detections,)
    scores: npt.NDArray[float]

    def __post_init__(self):
        assert self.boxes.ndim == 2
        assert self.labels.ndim == 1
        assert self.scores.ndim == 1
        assert self.boxes.shape[0] == self.labels.shape[0]
        assert self.boxes.shape[0] == self.scores.shape[0]
        assert self.boxes.shape[1] == 4

    def __bool__(self):
        return bool(self.boxes.size)


@dataclass
class FramePoses:
    """
    Represents pose estimations for a single image.

    We currently assume that all poses will be composed of the same number of
    keypoints.
    """

    # Array of scores for each pose. Ostensibly the bbox score. Shape: (num_poses,)
    scores: npt.NDArray[float]
    # Pose join 2D positions in ascending joint ID order. If the joint is not
    # present, 0s are used. Shape: (num_poses, num_joints, 2)
    joint_positions: npt.NDArray[float]
    # Poise joint scores. Shape: (num_poses, num_joints)
    joint_scores: npt.NDArray[float]

    def __post_init__(self):
        assert self.scores.ndim == 1
        assert self.joint_positions.ndim == 3
        assert self.joint_scores.ndim == 2
        assert self.scores.shape[0] == self.joint_positions.shape[0]
        assert self.scores.shape[0] == self.joint_scores.shape[0]
        assert self.joint_positions.shape[1] == self.joint_scores.shape[1]

    def __bool__(self):
        return bool(self.scores.size)


@dataclass
class FrameData:
    object_detections: tg.Optional[FrameObjectDetections]
    poses: tg.Optional[FramePoses]

    def __bool__(self):
        return bool(self.object_detections) or bool(self.poses)


@functools.lru_cache()
def _class_labels_to_map(
    class_labels: tg.Sequence[tg.Optional[str]]
) -> tg.Dict[str, int]:
    """
    Transform a sequence of class label strings into a mapping from label name
    to index.

    The output mapping will map labels to 0-based indices based on the order of
    the labels provided.
    """

    lbl_to_idx = {lbl: i for i, lbl in enumerate(class_labels) if lbl is not None}
    # Determine min index to subtract.
    min_cat = min(lbl_to_idx.values())
    for k in lbl_to_idx:
        lbl_to_idx[k] -= min_cat
    assert (
        set(lbl_to_idx.values()) == set(range(len(lbl_to_idx)))
    ), "Resulting category indices must start at 0 and be contiguous."
    return lbl_to_idx


def vectorize_window(
    frame_data: tg.Sequence[FrameData],
    det_class_labels: tg.Sequence[tg.Optional[str]],
    feat_version: int = 6,
    top_k_objects: int = 1,
) -> npt.NDArray[float]:
    """
    Construct an embedding vector for some window of data to be used for
    training and predicting against the TCN model.

    The length of the input data list is interpreted as the window size.

    Args:
        frame_data:
            Component data to use for constructing the embedding.
        det_class_labels:
            Sequence of string labels corresponding to object detection
            classes, in index order. Some indices may be None which means there
            is no class associated with that index. Any possible
            `FrameObjectDetections.labels` value must map to a label in this
            input. Other unused intermediate indices may be set to None.
        feat_version:
            Integer version number of the feature to compute.
            (historical concept)
        top_k_objects:
            Use this many most confident instances of every object type in the
            feature vector.

    Returns:
        Embedding vector matrix for the input window of data.
        Shape: (len(frame_data), embedding_dim)
    """
    # Discover feature dimension on first successful call to the per-frame
    # feature generation function.
    feat_dim: tg.Optional[int] = None
    feat_dtype = np.float32

    # Inverse mapping to the input index-to-label sequence for the classic
    # vectorizer.
    obj_label_to_ind = _class_labels_to_map(det_class_labels)

    f_vecs: tg.List[tg.Optional[npt.NDArray[float]]] = [None] * len(frame_data)
    for i, f_data in enumerate(frame_data):
        if f_data.object_detections is None:
            # Cannot proceed with classic vector computation without object
            # detections.
            continue

        f_dets = f_data.object_detections
        # extract object detection xywh as 4 component vectors.
        det_xs = f_dets.boxes.T[0]
        det_ys = f_dets.boxes.T[1]
        det_ws = f_dets.boxes.T[2]
        det_hs = f_dets.boxes.T[3]

        # There may be zero or multiple poses predicted on a frame.
        # If multiple poses, select the most confident "patient" pose.
        # If there was no pose on this frame, provide a list of 0's equal in
        # length to the number of joints.
        if f_data.poses.scores.size:
            f_poses = f_data.poses
            best_pose_idx = np.argmax(f_poses.scores)
            pose_kps = [
                {"xy": joint_pt}
                for joint_pt in f_poses.joint_positions[best_pose_idx]
            ]
        else:
            # special value for the classic method to indicate no pose joints.
            pose_kps = zero_joint_offset

        frame_feat = obj_det2d_set_to_feature(
            label_vec=[det_class_labels[lbl] for lbl in f_dets.labels],
            xs=det_xs,
            ys=det_ys,
            ws=det_ws,
            hs=det_hs,
            label_confidences=f_dets.scores,
            pose_keypoints=pose_kps,
            obj_label_to_ind=obj_label_to_ind,
            version=feat_version,
            top_k_objects=top_k_objects,
        ).ravel().astype(feat_dtype)
        feat_dim = frame_feat.size
        f_vecs[i] = frame_feat

    # If a caller is getting this, we could start to throw a more specific
    # error, and the caller could safely catch it to consider this window as
    # whatever the "background" class is.
    assert (
        feat_dim is not None
    ), "No features computed for any frame this window?"

    # If a feature fails to be generated for a frame:
    # * insert zero-vector matching dimensionality.
    empty_vec = np.zeros(shape=(feat_dim,), dtype=feat_dtype)
    for i in range(len(f_vecs)):
        if f_vecs[i] is None:
            f_vecs[i] = empty_vec

    return np.asarray(f_vecs)
