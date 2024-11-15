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
        det_loc_jitter:
            A relative amount to randomly adjust the position, height and width
            of frame data. Locations jitter is relative to the height and width
            of the frame. Box width and height jitter is relative to the width
            and height of the affected box. This should be in the [0, 1] range.
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
        det_loc_jitter: float = 0.025,
        det_wh_jitter: float = 0.1,
        dets_score_jitter: float = 0.1,
        pose_score_jitter: float = 0.1,
        pose_kp_loc_jitter: float = 0.025,
        pose_kp_score_jitter: float = 0.1,
    ):
        super().__init__()
        self.translate = translate
        self.scale = scale
        self.rotate = rotate
        self.det_loc_jitter = det_loc_jitter
        self.det_wh_jitter = det_wh_jitter
        self.dets_score_jitter = dets_score_jitter
        self.pose_score_jitter = pose_score_jitter
        self.pose_kp_loc_jitter = pose_kp_loc_jitter
        self.pose_kp_score_jitter = pose_kp_score_jitter

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
        transform = t_to_origin + t_rotate_scale + t_to_base
        # While we could just do `transform(2d_coors)`, it's not specifically
        # the fastest. Instead, we'll just do the dot product ourselves.
        tform = lambda pts: (
            transform.params
            @ np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T
        )[:2].T

        # Collect all frame detection boxes and scores in order to batch
        # transform and jitter.
        all_box_coords = []
        all_box_labels = []
        all_box_scores = []
        frame_dets_indices = []
        all_pose_scores = []
        all_pose_kps = []
        all_pose_kp_scores = []
        frame_pose_indices = []
        for i, frame in enumerate(window):
            if frame.object_detections is not None:
                all_box_coords.append(frame.object_detections.boxes)
                all_box_labels.append(frame.object_detections.labels)
                all_box_scores.append(frame.object_detections.scores)
                frame_dets_indices.extend([i] * len(frame.object_detections.boxes))
            if frame.poses is not None:
                all_pose_scores.append(frame.poses.scores)
                all_pose_kps.append(frame.poses.joint_positions)
                all_pose_kp_scores.append(frame.poses.joint_scores)
                frame_pose_indices.extend([i] * len(frame.poses.scores))

        if all_box_coords:
            # Pull together all xywh boxes.
            all_box_coords = np.concatenate(all_box_coords)  # shape: [n_dets, 4]
            all_box_labels = np.concatenate(all_box_labels)  # shape: [n_dets]
            all_box_scores = np.concatenate(all_box_scores)  # shape: [n_dets]
            frame_dets_indices = np.asarray(frame_dets_indices)
            n_dets = len(frame_dets_indices)

            # All the random for box components + scores generated in one
            # pass.
            det_rand = torch.rand(n_dets, 5).numpy()

            # Jitter box locations and sizes.
            xy_jitter_max = all_box_coords[:, 2:] * self.det_loc_jitter
            xy_jitter = (xy_jitter_max * 2 * det_rand[:, :2]) - xy_jitter_max
            all_box_coords[:, :2] += xy_jitter
            wh_jitter_max = all_box_coords[:, 2:] * self.det_wh_jitter
            wh_jitter = (wh_jitter_max * 2 * det_rand[:, 2:4]) - wh_jitter_max
            all_box_coords[:, 2:] += wh_jitter
            # Jitter box scores and clamp
            score_jitter = (
                self.dets_score_jitter * 2 * det_rand[:, 4]
            ) - self.dets_score_jitter
            all_box_scores += score_jitter
            all_box_scores[all_box_scores < 0] = 0
            all_box_scores[all_box_scores > 1] = 1

            # All box ltrb values. This is formatted this way to get the matrix
            # we want via a much less expensive np.reshape operation as opposed
            # to leaning on einops.
            corners = np.asarray(
                [
                    [
                        all_box_coords[:, 0],
                        all_box_coords[:, 0] + all_box_coords[:, 2],
                        all_box_coords[:, 0] + all_box_coords[:, 2],
                        all_box_coords[:, 0],
                    ],
                    [
                        all_box_coords[:, 1],
                        all_box_coords[:, 1],
                        all_box_coords[:, 1] + all_box_coords[:, 3],
                        all_box_coords[:, 1] + all_box_coords[:, 3],
                    ],
                ]
            ).T  # shape: [n_dets, 4, 2]
            # Reshape into a flat list for applying the transform to.
            corners = corners.reshape(n_dets * 4, 2)
            corners = tform(corners)
            corners = corners.reshape(n_dets, 4, 2)

            # Get min and max values of transformed boxes for each
            # detection to create new axis-aligned bounding boxes.
            x_min = corners[:, :, 0].min(axis=1)  # shape: [n_dets]
            y_min = corners[:, :, 1].min(axis=1)  # shape: [n_dets]
            x_max = corners[:, :, 0].max(axis=1)  # shape: [n_dets]
            y_max = corners[:, :, 1].max(axis=1)  # shape: [n_dets]
            all_box_coords = np.asarray([x_min, y_min, x_max - x_min, y_max - y_min]).T
            # Create mask for dets that are at least partially in the frame.
            in_frame = (
                (x_max > 0)
                & (y_max > 0)
                & (x_min < frame_width)
                & (y_min < frame_height)
            )
            # Filter down detections to those in the frame
            all_box_coords = all_box_coords[in_frame]
            all_box_labels = all_box_labels[in_frame]
            all_box_scores = all_box_scores[in_frame]
            frame_dets_indices = frame_dets_indices[in_frame]

        if all_pose_kps:
            all_pose_scores = np.concatenate(all_pose_scores)
            all_pose_kps = np.concatenate(all_pose_kps)
            all_pose_kp_scores = np.concatenate(all_pose_kp_scores)
            frame_pose_indices = np.asarray(frame_pose_indices)
            n_poses, n_kps = all_pose_kps.shape[:2]

            # All the random for box components + scores generated in one
            # pass. Need random values for each keypoint location and score
            # for each pose.
            pose_kp_rand = torch.rand(n_poses, n_kps, 3).numpy()
            pose_score_rand = torch.rand(n_poses).numpy()

            # Jitter pose score & clamp
            score_jitter = (
                self.pose_score_jitter * 2 * pose_score_rand
            ) - self.pose_score_jitter
            all_pose_scores += score_jitter
            all_pose_scores[all_pose_scores < 0] = 0
            all_pose_scores[all_pose_scores > 1] = 1
            # Jitter pose keypoint locations
            xy_jitter_max = (
                np.asarray([frame_width, frame_height]) * self.pose_kp_loc_jitter
            )
            xy_jitter = (xy_jitter_max * 2 * pose_kp_rand[:, :, :2]) - xy_jitter_max
            all_pose_kps += xy_jitter
            # Jitter pose keypoint scores & clamp
            score_jitter = (
                self.pose_kp_score_jitter * 2 * pose_kp_rand[:, :, 2]
            ) - self.pose_kp_score_jitter
            all_pose_kp_scores += score_jitter
            all_pose_kp_scores[all_pose_kp_scores < 0] = 0
            all_pose_kp_scores[all_pose_kp_scores > 1] = 1

            # Transform keypoint locations
            all_pose_kps = all_pose_kps.reshape(n_poses * n_kps, 2)
            all_pose_kps = tform(all_pose_kps)
            all_pose_kps = all_pose_kps.reshape(n_poses, n_kps, 2)

            # Zero out the scores for any joints that are now out of the frame
            in_frame = (
                (all_pose_kps[:, :, 0] >= 0)
                & (all_pose_kps[:, :, 1] >= 0)
                & (all_pose_kps[:, :, 0] < frame_width)
                & (all_pose_kps[:, :, 1] < frame_height)
            )
            all_pose_kp_scores[~in_frame] = 0

        modified_sequence: tg.List[FrameData] = [
            FrameData(None, None, size=d.size) for d in window
        ]
        for i, new_frame in enumerate(modified_sequence):
            if window[i].object_detections is not None:
                # Make sure we emit an instance if there was an instance input
                # even if the arrays are empty.
                frame_i_mask = frame_dets_indices == i
                new_frame.object_detections = FrameObjectDetections(
                    boxes=all_box_coords[frame_i_mask],
                    labels=all_box_labels[frame_i_mask],
                    scores=all_box_scores[frame_i_mask],
                )
            if window[i].poses is not None:
                # Make sure we emit an instance if there was an instance input
                # even if the arrays are empty.
                frame_i_mask = frame_pose_indices == i
                new_frame.poses = FramePoses(
                    scores=all_pose_scores[frame_i_mask],
                    joint_positions=all_pose_kps[frame_i_mask],
                    joint_scores=all_pose_kp_scores[frame_i_mask],
                )

        return modified_sequence


def test():
    from IPython.core.getipython import get_ipython
    import matplotlib.pyplot as plt
    from tcn_hpl.data.frame_data import FrameObjectDetections, FramePoses

    torch.manual_seed(0)
    # Prime the pump
    torch.rand(1)

    rng = np.random.RandomState(0)
    n_poses = 3
    pose_scores = rng.uniform(0, 1, n_poses)
    pose_joint_locs = rng.randint(0, 500, (n_poses, 22, 2)).astype(float)
    pose_joint_scores = rng.uniform(0, 1, (n_poses, 22))

    target_wh = 500, 500
    frame1 = FrameData(
        # 3 detections
        object_detections=FrameObjectDetections(
            boxes=np.array([[10.0, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]),
            labels=np.array([1, 2, 3]),
            scores=np.array([0.9, 0.75, 0.11]),
        ),
        # 3 poses, 22 keypoints (nominal)
        poses=FramePoses(
            scores=pose_scores,
            joint_positions=pose_joint_locs,
            joint_scores=pose_joint_scores,
        ),
        size=target_wh,
    )
    window = [frame1] * 25

    augment = FrameDataRotateScaleTranslateJitter(
        translate=0.25,
        scale=[0.7, 1.3],
        rotate=[-25, 25],
        det_loc_jitter=0,  # 0.03,
        det_wh_jitter=0,  # 0.25,
        pose_kp_loc_jitter=0,  # 0.035,
        dets_score_jitter=0.4,
        pose_score_jitter=0.4,
        pose_kp_score_jitter=0.4,
    )

    # Idiot check.
    augment(window)

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("timeit", "augment(window)")

    # Visualize detection boxes before augmentation
    fig, axes = plt.subplots(5, 5, figsize=(25, 25))
    axes = axes.ravel()
    axes[0].set_title("Before Augmentation")
    axes[0].set_xlim(0, target_wh[0])
    axes[0].set_ylim(0, target_wh[1])
    for frame in window:
        for box in frame.object_detections.boxes:
            x, y, w, h = box
            rect = plt.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor="magenta", facecolor="none"
            )
            axes[0].add_patch(rect)
        for pose_kp in frame.poses.joint_positions:
            axes[0].plot(pose_kp[:, 0], pose_kp[:, 1])

    # For every other axes, show a different augmentation example
    for ax_i in range(1, len(axes)):
        aug_window = augment(window)
        axes[ax_i].set_title(f"Separate Augmentation [{ax_i}]")
        axes[ax_i].set_xlim(0, target_wh[0])
        axes[ax_i].set_ylim(0, target_wh[1])
        for frame in aug_window:
            for box in frame.object_detections.boxes:
                x, y, w, h = box
                rect = plt.Rectangle(
                    (x, y), w, h, linewidth=1, edgecolor="magenta", facecolor="none"
                )
                axes[ax_i].add_patch(rect)
            for pose_kp in frame.poses.joint_positions:
                axes[ax_i].plot(pose_kp[:, 0], pose_kp[:, 1])

    plt.savefig("FrameDataRotateScaleTranslateJitter_vis.png")
    plt.close(fig)

    # Visualize every frame in a window
    fig, axes = plt.subplots(5, 5, figsize=(25, 25))
    axes = axes.ravel()
    augmented_window = augment(window)
    for f_i, frame in enumerate(augmented_window):
        axes[f_i].set_title(f"Augmented Frame [{f_i}]")
        axes[f_i].set_xlim(0, target_wh[0])
        axes[f_i].set_ylim(0, target_wh[1])
        for box in frame.object_detections.boxes:
            x, y, w, h = box
            rect = plt.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor="magenta", facecolor="none"
            )
            axes[f_i].add_patch(rect)
        for pose_kp in frame.poses.joint_positions:
            axes[f_i].plot(pose_kp[:, 0], pose_kp[:, 1])

    plt.savefig("FrameDataRotateScaleTranslateJitter_allFrames.png")
    plt.close(fig)


if __name__ == "__main__":
    test()
