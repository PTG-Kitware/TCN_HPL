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
    default_bbox
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
            best_pose_score = np.max(f_poses.scores)
            pose_kps = [
                {"xy": joint_pt, "score": joint_score}
                for (joint_pt, joint_score) in zip(f_poses.joint_positions[best_pose_idx], f_poses.joint_scores[best_pose_idx])
            ]
        else:
            # special value for the classic method to indicate no pose joints.
            pose_kps = zero_joint_offset
        casualty = {"score": , "xywh": }

        frame_feat = obj_det2d_set_to_feature_by_method_new(
            label_vec=[det_class_labels[lbl] for lbl in f_dets.labels],
            xs=det_xs,
            ys=det_ys,
            ws=det_ws,
            hs=det_hs,
            label_confidences=f_dets.scores,
            pose_confidence=best_pose_score,
            pose_keypoints=pose_kps,
            obj_label_to_ind=obj_label_to_ind,
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

def obj_det2d_set_to_feature_by_method_new(
    label_vec: List[str],
    xs: List[float],
    ys: List[float],
    ws: List[float],
    hs: List[float],
    label_confidences: List[float],
    pose_confidence: float,
    pose_keypoints: List[Dict],
    obj_label_to_ind: Dict[str, int],
    top_k_objects: int = 1,
    ):
    """
    :param label_vec: List of object labels for each detection (length: # detections)
    :param xs: List of x values for each detection (length: # detections)
    :param ys: List of y values for each detection (length: # detections)
    :param ws: List of width values for each detection (length: # detections)
    :param hs: List of height values for each detection (length: # detections)
    :param label_confidences: List of confidence values for each detection (length: # detections)
    :param pose_keypoints:
        List of joints, represented by a dictionary contining the x and y corrdinates of the points and the category id and string
    :param obj_label_to_ind:
        Dictionary mapping a label str and returns the index within the feature vector.
    :param top_k_objects: Number top confidence objects to use per label, defaults to 1
    :param use_activation: If True, add the confidence values of the detections to the feature vector, defaults to False
    :param use_hand_dist: If True, add the distance of the detection centers to both hand centers to the feature vector, defaults to False
    :param use_intersection: If True, add the intersection of the detection boxes with the hand boxes to the feature vector, defaults to False
    :param use_joint_hand_offset: If True, add the distance of the hand centers to the patient joints to the feature vector, defaults to False
    :param use_joint_object_offset: If True, add the distance of the object centers to the patient joints to the feature vector, defaults to False

    :return:
        resulting feature data
    """
    #########################
    # Data
    #########################
    # Number of object detection classes
    num_det_classes = len(obj_label_to_ind)

    # Maximum confidence observe per-class across input object detections.
    # If a class has not been observed, it is set to 0 confidence.
    det_class_max_conf = np.zeros((num_det_classes, top_k_objects))
    # The bounding box of the maximally confident detection
    det_class_bbox = np.zeros((top_k_objects, num_det_classes, 4), dtype=np.float64)
    det_class_bbox[:] = default_bbox

    # Binary mask indicates which detection classes are present on this frame.
    det_class_mask = np.zeros((top_k_objects, num_det_classes), dtype=np.bool_)

    # Record the most confident detection for each object class as recorded in
    # `obj_label_to_ind` (confidence & bbox)
    for i, label in enumerate(label_vec):
        assert label in obj_label_to_ind, f"Label {label} is unknown"

        conf = label_confidences[i]
        ind = obj_label_to_ind[label]

        conf_list = det_class_max_conf[ind, :]
        if conf > det_class_max_conf[ind].min():
            # Replace the lowest confidence object with our new higher confidence object
            min_conf_ind = np.where(conf_list == conf_list.min())[0][0]

            conf_list[min_conf_ind] = conf
            det_class_bbox[min_conf_ind, ind] = [xs[i], ys[i], ws[i], hs[i]]
            det_class_mask[min_conf_ind, ind] = True

            # Sort the confidences to determine the top_k order
            sorted_index = np.argsort(conf_list)[::-1]
            sorted_conf_list = np.array([conf_list[k] for k in sorted_index])

            # Reorder the values to match the confidence top_k order
            det_class_max_conf[ind] = sorted_conf_list

            bboxes = det_class_bbox.copy()
            mask = det_class_mask.copy()
            for idx, sorted_ind in enumerate(sorted_index):
                det_class_bbox[idx, ind] = bboxes[sorted_ind, ind]
                det_class_mask[idx, ind] = mask[sorted_ind, ind]

    #########################
    # util functions
    #########################
    def find_hand(hand_str):
        hand_idx = obj_label_to_ind[hand_str]
        hand_conf = det_class_max_conf[hand_idx][0]
        hand_bbox = kwimage.Boxes([det_class_bbox[0, hand_idx]], "xywh")

        return hand_idx, hand_bbox, hand_conf, hand_bbox.center

    #########################
    # Hands
    #########################
    # Find the right hand
    (right_hand_idx, right_hand_bbox, right_hand_conf, right_hand_center) = find_hand(
        "hand (right)"
    )

    # Find the left hand
    (left_hand_idx, left_hand_bbox, left_hand_conf, left_hand_center) = find_hand(
        "hand (left)"
    )

    #########################
    # Feature vector
    #########################
    feature_vec = np.zeros(97)
    i = 0

    # HANDS
    # Both hands' confidence, X, Y, W, H
    right_hand_bbox, right_hand_conf
    feature_vec[i] = right_hand_conf
    i += 1
    feature_vec[i:i+4] = right_hand_bbox
    i += 4
    feature_vec[i] = left_hand_conf
    i += 1
    feature_vec[i:i+4] = left_hand_bbox
    i += 4

    # OBJECTS
    # All top-confidence objects' Confidence, X, Y, W, H
    # To use "top_k_objects" to implement K > 1, The 0 index below should be an iterable.
    for obj_ind in range(num_det_classes):
        feature_vec.append([det_class_max_conf[obj_ind][0]])

        # Confidence
        feature_vec[i] = det_class_max_conf[obj_ind][0]
        i += 1
        # Coordinates
        feature_vec[i:i+4] = det_class_bbox[0][obj_ind]
        i += 4

    # CASUALTY
    # Casualty object Confidence (TODO: get X, Y, W, H)
    feature_vec[i] = pose_confidence
    i += 1

    # POSE JOINTS
    # All pose joints' conf, X, Y
    for joint in pose_keypoints:
        jscore = joint["score"]
        jx, jy = joint["xy"]
        feature_vec[i] = joint["score"]
        i += 1
        feature_vec[i:i+2] = joint["xy"]
        i += 2

    return feature_vec
