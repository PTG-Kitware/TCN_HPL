from dataclasses import dataclass
import typing as tg

import numpy.typing as npt


__all__ = [
    "FrameObjectDetections",
    "FramePoses",
    "FrameData",
]


@dataclass
class FrameObjectDetections:
    """
    Representation of object detection predictions for a single image.
    All sequences should be the same length.
    """

    # Detection 2D bounding boxes in xywh format for the left, top, width and
    # height measurements respectively. Shape: (num_detections, 4)
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
