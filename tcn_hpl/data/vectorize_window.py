"""
Logic and utilities to perform the vectorization of input data into an
embedding space used for TCN training and prediction.
"""
import typing as tg

import numpy as np
import numpy.typing as npt

from tcn_hpl.data.vectorize import (
    FrameObjectDetections,
    FramePoses,
    FrameData,
)
from tcn_hpl.data.vectorize.classic import _class_labels_to_map  # noqa
from tcn_hpl.data.vectorize_classic import (
    obj_det2d_set_to_feature,
    zero_joint_offset,
)


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
        f_dets = f_data.object_detections
        if f_dets is None:
            # Cannot proceed with classic vector computation without object
            # detections.
            continue

        # extract object detection xywh as 4 component vectors.
        det_xs = f_dets.boxes.T[0]
        det_ys = f_dets.boxes.T[1]
        det_ws = f_dets.boxes.T[2]
        det_hs = f_dets.boxes.T[3]

        # There may be zero or multiple poses predicted on a frame.
        # If multiple poses, select the most confident "patient" pose.
        # If there was no pose on this frame, provide a list of 0's equal in
        # length to the number of joints.
        f_poses = f_data.poses
        if f_poses is not None and f_poses.scores.size:
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
