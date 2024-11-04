from typing import Sequence, List, Optional

import numpy as np
import torch

from tcn_hpl.data.frame_data import FrameData


class DropoutFrameDataTransform(torch.nn.Module):
    """
    Augmentation of a FrameData window that will drop out object detections or
    pose estimations for some frames as if they were never computed for those
    frames.

    This aims to simulate how a live system cannot keep up with predicting
    these estimations on all input data in a streaming system.

    Args:
        frame_rate:
            The frame rate in Hz of the input video or sequence from which a
            window of data is associated with.
        dets_throughput_mean:
            Rate in Hz at which object detection predictions should be
            represented in the window.
        pose_throughput_mean:
            Rate in Hz at which pose estimation predictions should be
            represented in the window.
        dets_latency:
            Optional separate latency in seconds for object detection
            predictions. If not provided, we will interpret latency to be the
            inverse of throughput. It many be useful to provide a specific
            latency value if processing conditions are specialized beyond the
            naive consideration that windows will end in the latest observable
            image frame.
        pose_latency:
            Optional separate latency in seconds for pose estimation
            predictions. If not provided, we will interpret latency to be the
            inverse of throughput. It many be useful to provide a specific
            latency value if processing conditions are specialized beyond the
            naive consideration that windows will end in the latest observable
            image frame.
    """

    def __init__(
        self,
        frame_rate: float,
        dets_throughput_mean: float,
        pose_throughput_mean: float,
        dets_latency: Optional[float] = None,
        pose_latency: Optional[float] = None,
    ):
        super().__init__()
        self.frame_rate = frame_rate
        self.dets_throughput_mean = dets_throughput_mean
        self.pose_throughput_mean = pose_throughput_mean
        # If no separate latency, then just assume inverse of throughput.
        self.dets_latency = dets_latency if dets_latency is not None else 1. / dets_throughput_mean
        self.pose_latency = pose_latency if pose_latency is not None else 1. / pose_throughput_mean

    def forward(self, window: Sequence[FrameData]) -> List[FrameData]:
        # Starting from some latency back from the end of the window, start
        # dropping out detections and poses as if they were not produced for
        # that frame. Do this separately for poses and detections as their
        # agents can operate at different rates.

        n_frames = len(window)
        one_frame_time = 1.0 / self.frame_rate

        # Vector of frame time offsets starting from the oldest frame.
        # Time progresses from the first frame (0 seconds) to the
        # last frame in the window (increasing by one_frame_time for each frame).
        frame_times = np.arange(n_frames) * one_frame_time
        max_frame_time = frame_times[-1]

        # Define processing intervals (how often a frame is processed)
        # TODO: Vectorize this, adding random variation by utilizing
        #       `torch.normal(mean, std)`.
        dets_interval = 1.0 / self.dets_throughput_mean
        pose_interval = 1.0 / self.pose_throughput_mean

        # Initialize end time trackers for processing detections and poses.
        # Simulate that agents may already be part-way through processing a
        # frame before the beginning of this window.
        dets_processing_end = np.full(n_frames + 1, torch.rand(1) * dets_interval)
        pose_processing_end = np.full(n_frames + 1, torch.rand(1) * pose_interval)

        # Boolean arrays to keep track of whether a frame can be processed
        dets_mask = np.zeros(n_frames, dtype=bool)
        pose_mask = np.zeros(n_frames, dtype=bool)

        # Simulate realistic processing behavior
        for idx in range(n_frames):
            # Processing can occur on this frame if the processing for the
            # previous frame finishes before the frame after this arrives
            # (represented by the `+ one_frame_time`), since the "current"
            # frame for the agent would still be this frame.

            # Object detection processing
            if frame_times[idx] + one_frame_time > dets_processing_end[idx]:
                # Agent finishes processing before the next frame would come it
                # so it processes this frame.
                dets_mask[idx] = True
                dets_processing_end[idx + 1:] = (
                    dets_processing_end[idx] + dets_interval
                    if dets_processing_end[idx] >= frame_times[idx]
                    else frame_times[idx] + dets_interval
                )

            # Pose processing
            if frame_times[idx] + one_frame_time > pose_processing_end[idx]:
                # Agent finishes processing before the next frame would come it
                # so it processes this frame.
                pose_mask[idx] = True
                pose_processing_end[idx + 1:] = (
                    pose_processing_end[idx] + pose_interval
                    if pose_processing_end[idx] >= frame_times[idx]
                    else frame_times[idx] + pose_interval
                )

        # Mask out the ends based on configured latencies. This ensures we do
        # not mark a window frame as "processed" when its completion would
        # occur *after* the final frame is received.
        dets_mask &= (frame_times + self.dets_latency) <= max_frame_time
        pose_mask &= (frame_times + self.pose_latency) <= max_frame_time

        # Create the modified sequence
        modified_sequence = [
            FrameData(
                object_detections=(
                    window[idx].object_detections if dets_mask[idx] else None
                ),
                poses=(window[idx].poses if pose_mask[idx] else None),
            )
            for idx in range(n_frames)
        ]

        return modified_sequence


def test():
    from tcn_hpl.data.frame_data import FrameObjectDetections, FramePoses

    frame1 = FrameData(
        object_detections=FrameObjectDetections(
            boxes=np.array([[10, 20, 30, 40], [50, 60, 70, 80]]),
            labels=np.array([1, 2]),
            scores=np.array([0.9, 0.75]),
        ),
        poses=FramePoses(
            scores=np.array([0.8]),
            joint_positions=np.array([[[10, 20], [30, 40], [50, 60]]]),
            joint_scores=np.array([[0.9, 0.85, 0.8]]),
        ),
    )
    sequence = [frame1] * 25
    transform = DropoutFrameDataTransform(
        frame_rate=1,
        dets_throughput_mean=15,
        pose_throughput_mean=0.66,
    )
    # transform = DropoutFrameDataTransform(
    #     frame_rate=15,
    #     dets_throughput=14.7771,
    #     dets_latency=0,
    #     pose_throughput=10,
    #     pose_latency=1/10,  # (1/10)-(1/14.7771),
    # )
    modified_sequence = transform(sequence)

    for idx, frame in enumerate(modified_sequence):
        print(
            f"Frame {idx}: Object Detections: {frame.object_detections is not None}, Poses: {frame.poses is not None}"
        )


if __name__ == "__main__":
    test()
