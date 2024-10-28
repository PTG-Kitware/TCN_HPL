import functools
import typing as tg

import numpy as np
from numpy import typing as npt

from tcn_hpl.data.vectorize._interface import Vectorize, FrameData
from tcn_hpl.data.vectorize_classic import (
    obj_det2d_set_to_feature,
    zero_joint_offset,
    HAND_STR_LEFT,
    HAND_STR_RIGHT,
)


class Classic(Vectorize):
    """
    Previous manual approach to vectorization.

    Arguments:
        feat_version: Version number of the feature to produce.
        top_k: The number of top per-class examples to use in vector
            construction.
    """

    def __init__(
        self,
        feat_version: int = 6,
        top_k: int = 1,
        num_classes: int = 7,
        background_idx: int = 0,
        hand_left_idx: int = 5,
        hand_right_idx: int = 6,
    ):
        super().__init__()

        self._feat_version = feat_version
        self._top_k = top_k
        # The classic means of vectorization required some inputs that involved
        # string forms of object class labels.
        # * The first vector of string labels is the sequence of predicted
        #   classes but as the string semantic labels. This is only ever used
        #   to index into a second mapping structure to get the numerical index
        #   of that class.
        # * The second structure is a mapping of string class labels to some
        #   zero-based index. The indices represented in this mapping must
        #   start at 0 and consecutively increase. There is expected to be some
        #   background class index in raw object predictions that will need to
        #   be appropriately excluded. This mapping is known to be checked for
        #   special left and right hand names (HAND_STR_LEFT & HAND_STR_RIGHT).
        #
        self._num_classes = num_classes
        self._bg_idx = background_idx
        self._h_l_idx = hand_left_idx
        self._h_r_idx = hand_right_idx
        # Construct a vector of "labels" that can be mapped to via object
        # detection preds, specifically injecting the special names for hands
        # at the specified indices.
        det_class_labels = list(map(str, range(num_classes)))
        det_class_labels[background_idx] = None
        det_class_labels[hand_left_idx] = HAND_STR_LEFT
        det_class_labels[hand_right_idx] = HAND_STR_RIGHT
        self._det_class_labels = det_class_labels = tuple(det_class_labels)
        self._det_class2idx_map = _class_labels_to_map(det_class_labels)

    def vectorize(self, data: FrameData) -> npt.NDArray[np.float32]:
        det_class_labels = self._det_class_labels
        obj_label_to_ind = self._det_class2idx_map

        f_dets = data.object_detections
        if f_dets is not None:
            # extract object detection xywh as 4 component vectors.
            det_xs = f_dets.boxes.T[0]
            det_ys = f_dets.boxes.T[1]
            det_ws = f_dets.boxes.T[2]
            det_hs = f_dets.boxes.T[3]
            # Other vectors
            det_lbls = [det_class_labels[lbl] for lbl in f_dets.labels]
            det_scores = f_dets.scores
        else:
            det_lbls = []
            det_xs = []
            det_ys = []
            det_ws = []
            det_hs = []
            det_scores = []

        # There may be zero or multiple poses predicted on a frame.
        # If multiple poses, select the most confident "patient" pose.
        # If there was no pose on this frame, provide a list of 0's equal in
        # length to the number of joints.
        f_poses = data.poses
        if f_poses is not None and f_poses.scores.size:
            best_pose_idx = np.argmax(f_poses.scores)
            pose_kps = [
                {"xy": joint_pt} for joint_pt in f_poses.joint_positions[best_pose_idx]
            ]
        else:
            # special value for the classic method to indicate no pose joints.
            pose_kps = zero_joint_offset

        frame_feat = (
            obj_det2d_set_to_feature(
                label_vec=det_lbls,
                xs=det_xs,
                ys=det_ys,
                ws=det_ws,
                hs=det_hs,
                label_confidences=det_scores,
                pose_keypoints=pose_kps,
                obj_label_to_ind=obj_label_to_ind,
                version=self._feat_version,
                top_k_objects=self._top_k,
            )
            .ravel()
            .astype(np.float32)
        )

        return frame_feat


@functools.lru_cache()
def _class_labels_to_map(
    class_labels: tg.Sequence[tg.Optional[str]],
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
    assert set(lbl_to_idx.values()) == set(
        range(len(lbl_to_idx))
    ), "Resulting category indices must start at 0 and be contiguous."
    return lbl_to_idx
