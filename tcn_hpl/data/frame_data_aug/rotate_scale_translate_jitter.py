import typing as tg

import numpy as np
from skimage.transform import SimilarityTransform
import torch

from tcn_hpl.data.frame_data import FrameData, FrameObjectDetections, FramePoses


class FrameDataRotateScaleTranslateJitter(torch.nn.Module):
    """
    Randomly translate, scale and rotate object detections and pose keypoints
    consistently across frames in a window, within input ranges.

    A global rotation, scaling and translation will be applied consistently to
    all object detections and pose keypoints within a window. Window global
    rotation will occur about a center of rotation randomly selected within the
    lower central region of the frame space. We assume that all frames in the
    window are the same shape.
    The following ASCII roughly shows an inner region of a "frame" where the
    center of rotation will be randomly selected from.

        +--------+
        |        |
        |  +--+  |
        |  |  |  |
        +--------+

    A separate per-Object/Keypoint jitter translation will be randomly applied
    to individual [x,y] locations within a separate tolerance than the global
    translation.
    This jitter is to happen **after** the window-global transformation.

    Args:
        translate:
            Relative amount of the frame's width and height that we may
            translate a whole window's scene by. This should be in the range
            [0, 1]. This single value reflects a [-translate, translate] range
            that is randomly selected from twice to apply to the image height
            and width.
        scale:
            Range from which we will select the random scene scale
            transformation. The values should be in ascending order.
        rotate:
            Range in degrees for which we will randomly rotate the scene about
            a chosen center of rotation. The values given should be in
            ascending order.
        location_jitter:
            A relative amount to randomly adjust the position, height and width
            of frame data. For object detections, this is applied to all corner
            points uniformly per detection. For pose joint keypoints, jitter is
            applied independently per keypoint.
        dets_score_jitter:
            Randomly adjust the object detection confidence value within +/-
            the given value. The resulting value is clamped within the [0, 1]
            range.
        pose_score_jitter:
            Randomly adjust the pose keypoint confidence value within +/-
            the given value. The resulting value is clamped within the [0, 1]
            range.
    TODO: Some maximum tolerance for accepting boxes that are outside of the
          image space?
    """

    def __init__(
        self,
        translate: float = 0.1,
        scale: tg.Sequence[float] = (0.9, 1.1),
        rotate: tg.Sequence[float] = (-10, 10),
        location_jitter: float = 0.05,
        dets_score_jitter: float = 0.1,
        pose_score_jitter: float = 0.1,
    ):
        super().__init__()
        self.translate = translate
        self.scale = scale
        self.rotate = rotate
        self.location_jitter = location_jitter
        self.dets_score_jitter = dets_score_jitter
        self.pose_score_jitter = pose_score_jitter

    def _tform_dets(
        self,
        transform: SimilarityTransform,
        dets: FrameObjectDetections,
        frame_height: int,
        frame_width: int,
    ):
        boxes = dets.boxes  # Shape: [4, n_dets]
        n_boxes = len(boxes)
        # All box ltrb values. This is formatted this way to get the matrix we
        # want via a much less expensive np.reshape operation.
        corners = np.asarray([
            [boxes[:, 0], boxes[:, 0] + boxes[:, 2], boxes[:, 0] + boxes[:, 2], boxes[:, 0]],
            [boxes[:, 1], boxes[:, 1], boxes[:, 1] + boxes[:, 3], boxes[:, 1] + boxes[:, 3]],
        ]).T  # shape: [n_boxes, 4, 2]
        # Reshape into a flat list for applying the transform to.
        corners = corners.reshape(n_boxes * 4, 2)
        # corners shape now: [4 * n_boxes, 2]
        t_corners = transform(corners)  # shape: [4* n_boxes, 2]
        t_corners = t_corners.reshape(n_boxes, 4, 2)
        # t_corners shape now: [n_boxes, 4, 2]
        # Get min and max values of transformed boxes for each
        # detection to create new axis-aligned bounding boxes.
        x_min = t_corners[:, :, 0].min(axis=1)  # shape: [n_boxes]
        y_min = t_corners[:, :, 1].min(axis=1)  # shape: [n_boxes]
        x_max = t_corners[:, :, 0].max(axis=1)  # shape: [n_boxes]
        y_max = t_corners[:, :, 1].max(axis=1)  # shape: [n_boxes]
        # Generate and apply a little bownian jitter based on a
        # fraction of the transformed object bounding box size. This is
        # happening post scene transform on purpose.
        jitter_max = (
            self.location_jitter * np.array([x_max - x_min, y_max - y_min]).T
        )  # shape: [n_boxes, 2]
        jitter_offset = (2 * torch.rand(n_boxes, 2) - 1).numpy() * jitter_max
        x_min += jitter_offset[:, 0]
        y_min += jitter_offset[:, 1]
        x_max += jitter_offset[:, 0]
        y_max += jitter_offset[:, 1]
        # Jitter confidence scores
        new_scores = (
            (self.dets_score_jitter * 2 * torch.rand(dets.scores.shape).numpy())
            - self.dets_score_jitter
        ) + dets.scores
        new_scores[new_scores < 0] = 0
        new_scores[new_scores > 1] = 1

        # Create mask for boxes that are at least partially in-frame.
        mask = (
            (x_max > 0) & (y_max > 0) & (x_min < frame_width) & (y_min < frame_height)
        )
        new_boxes = np.asarray([x_min, y_min, x_max - x_min, y_max - y_min]).T
        return FrameObjectDetections(
            boxes=new_boxes[mask],
            labels=dets.labels[mask],
            scores=new_scores[mask],
        )

    def _tform_poses(
        self,
        transform: SimilarityTransform,
        poses: FramePoses,
        frame_height: int,
        frame_width: int,
    ):
        n_poses, n_kps = poses.joint_positions.shape[:2]
        joints = poses.joint_positions.reshape(n_poses * n_kps, -1)
        joints = transform(joints)
        joints = joints.reshape(n_poses, n_kps, -1)  # [n_poses, n_joints, 2]
        # Add random jitter to keypoint positions
        jitter_max = self.location_jitter * np.ones_like(joints)
        jitter_offset = (2 * torch.rand(jitter_max.shape).numpy() - 1) * jitter_max
        joints += jitter_offset
        # Add random jitter to keypoint scores
        new_joint_scores = (
            (self.pose_score_jitter * 2 * torch.rand(poses.joint_scores.shape).numpy())
            - self.pose_score_jitter
        ) + poses.joint_scores
        new_joint_scores[new_joint_scores < 0] = 0
        new_joint_scores[new_joint_scores > 1] = 1
        # Zero out the scores for any joints that are now out of the
        # frame.
        in_frame_mask = (
            (joints[:, :, 0] >= 0)
            & (joints[:, :, 1] >= 0)
            & (joints[:, :, 0] < frame_width)
            & (joints[:, :, 1] < frame_height)
        )
        new_joint_scores[~in_frame_mask] = 0
        return FramePoses(
            scores=poses.scores,
            joint_positions=joints,
            joint_scores=new_joint_scores,
        )

    def forward(self, window: tg.Sequence[FrameData]) -> tg.List[FrameData]:
        # Extract frame size from the first frame (assuming all frames have the same size)
        frame_width, frame_height = window[0].size

        # Create a pool of random numbers we will use below.
        # torch's random operators are utilized to align with training system's
        # setting seeds to torch and not numpy.
        r_pool = torch.rand(6).numpy()

        # Random center of rotation within the central lower quadrant of the
        # frame space. Based on [vvv] to produce range [r1, r2)
        # https://stackoverflow.com/questions/44328530/how-to-get-a-uniform-distribution-in-a-range-r1-r2-in-pytorch
        c_x_upper = 3 * frame_width / 4
        c_x_lower = frame_width / 4
        center_x = (c_x_upper - c_x_lower) * r_pool[0] + c_x_lower
        c_y_upper = frame_height / 2
        c_y_lower = frame_height
        center_y = (c_y_upper - c_y_lower) * r_pool[1] + c_y_lower

        # Apply consistent global transformations (translate, scale, rotate)
        # across all frames in the window
        angle = (self.rotate[1] - self.rotate[0]) * r_pool[2] + self.rotate[0]
        translate_x = (self.translate * 2 * r_pool[3] - self.translate) * frame_width
        translate_y = (self.translate * 2 * r_pool[4] - self.translate) * frame_height
        scale_factor = (self.scale[1] - self.scale[0]) * r_pool[5] + self.scale[0]

        t_to_origin = SimilarityTransform(translation=(-center_x, -center_y))
        t_rotate_scale = SimilarityTransform(
            scale=scale_factor,
            rotation=np.deg2rad(angle),
            translation=(translate_x, translate_y),
        )
        t_to_base = SimilarityTransform(translation=(center_x, center_y))
        transform = t_to_base + t_rotate_scale + t_to_origin

        modified_sequence: tg.List[FrameData] = []

        for fd in window:
            new_fd = FrameData(
                object_detections=None,
                poses=None,
                size=fd.size,
            )

            if fd.object_detections is not None:
                new_fd.object_detections = self._tform_dets(
                    transform=transform,
                    dets=fd.object_detections,
                    frame_height=frame_height,
                    frame_width=frame_width,
                )

            if fd.poses is not None:
                new_fd.poses = self._tform_poses(
                    transform=transform,
                    poses=fd.poses,
                    frame_height=frame_height,
                    frame_width=frame_width,
                )

            modified_sequence.append(new_fd)

        return modified_sequence


def test():
    from IPython.core.getipython import get_ipython
    import matplotlib.pyplot as plt
    from tcn_hpl.data.frame_data import FrameObjectDetections, FramePoses

    torch.manual_seed(0)

    rng = np.random.RandomState(0)
    n_poses = 3
    pose_scores = rng.uniform(0, 1, n_poses)
    pose_joint_locs = rng.randint(0, 500, (n_poses, 22, 2))
    pose_joint_scores = rng.uniform(0, 1, (n_poses, 22))

    frame1 = FrameData(
        # 3 detections
        object_detections=FrameObjectDetections(
            boxes=np.array([[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]),
            labels=np.array([1, 2, 3]),
            scores=np.array([0.9, 0.75, 0.11]),
        ),
        # 3 poses, 22 keypoints (nominal)
        poses=FramePoses(
            scores=pose_scores,
            joint_positions=pose_joint_locs,
            joint_scores=pose_joint_scores,
        ),
        size=(500, 500),
    )
    window = [frame1] * 25

    augment = FrameDataRotateScaleTranslateJitter()

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("timeit", "augment(window)")

    # Visualize detection boxes before augmentation
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].set_title("Before Augmentation")
    axes[0].set_xlim(-100, 600)
    axes[0].set_ylim(-100, 600)
    for box in window[0].object_detections.boxes:
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='magenta', facecolor='none')
        axes[0].add_patch(rect)
    for pose_kp in window[0].poses.joint_positions:
        axes[0].plot(pose_kp[:, 0], pose_kp[:, 1])

    # Sanity check performing a single window augmentation
    new_window = augment(window)

    axes[1].set_title("After Augmentation")
    axes[1].set_xlim(-100, 600)
    axes[1].set_ylim(-100, 600)
    for box in new_window[0].object_detections.boxes:
        x, y, w, h = box
        rect = plt.Rectangle((x, y), w, h, linewidth=1, edgecolor='magenta', facecolor='none')
        axes[1].add_patch(rect)
    for i, pose_kp in enumerate(new_window[0].poses.joint_positions):
        axes[1].plot(pose_kp[:, 0], pose_kp[:, 1])

    plt.savefig('FrameDataRotateScaleTranslateJitter_vis.png')


if __name__ == "__main__":
    test()
