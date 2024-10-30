import logging
import os
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union

import kwcoco
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tcn_hpl.data.vectorize import (
    FrameObjectDetections,
    FramePoses,
    FrameData,
    Vectorize,
)


logger = logging.getLogger(__name__)


class TCNDataset(Dataset):
    """
    TCN Activity Classifier dataset.

    This dataset performs common initialization and indexing of windowed data.

    The mask return from the get-item method is being honored for backwards
    compatibility, but will only ever contain 1's as there should no longer any
    video overlap possible in returned windows.

    During "online" mode, which means data is populated into this dataset via
    the `load_data_online()` method, the truth, source video ID and source
    frame ID returns from get-item are undefined.

    When loading data in "offline" mode, i.e. given COCO datasets, we will
    pre-compute window vectors to save time during training. We will attempt to
    cache these vectors if a cache directory is provided to the
    `load_data_offline` method.

    Args:
        window_size:
            The size of the sliding window used to collect inputs from either a
            real-time or offline source.
        vectorizer:
            Vectorization functor to convert frame data into an embedding
            space.
        transform:
            Optional feature vector transformation/augmentation function.
    """

    def __init__(
        self,
        window_size: int,
        vectorizer: Vectorize,
        transform: Optional[Callable] = None,
    ):
        self.window_size = window_size
        self.vectorizer = vectorizer
        self.transform = transform

        # For offline mode, pre-cut videos into clips according to window
        # size for easy batching.
        # For online mode, expect only one window to be set at a time via the
        # `load_data_online` method.
        # Content to be indexed into during __getitem__.
        # This cannot be stored as a ndarray due to its variable nature.
        self._window_data: List[List[FrameData]] = []
        # The truth labels per-frame per-window.
        # Shape: (n_windows, window_size)
        self._window_truth: Optional[npt.NDArray[int]] = None
        # Per-window, which source video ID it is associated with. For online
        # mode, the value is undefined.
        # Shape: (n_windows,)
        self._window_vid: Optional[npt.NDArray[int]] = None
        # Video frame indices per-frame per-window.
        # Shape: (n_windows, window_size)
        self._window_frames: Optional[npt.NDArray[int]] = None
        # Optionally calculated weight to apply to a window. This is to support
        # weighted random sampling during training. This should only be
        # available when there is truth avaialble, i.e. during offline mode.
        self._window_weights: Optional[npt.NDArray[float]] = None
        # Optionally defined set of pre-computed window vectors.
        self._window_vectors: Optional[npt.NDArray[float]] = None

        # Constant 1's mask value to re-use during get-item.
        self._ones_mask: npt.NDArray[int] = np.ones(window_size, dtype=int)

    @property
    def window_weights(self) -> npt.NDArray[float]:
        """
        Get per-index weights to use with a weighted sampler.

        Returns:
            Array of per-index weight floats.
        """
        if self._window_weights is None:
            raise RuntimeError("No class weights calculated for this dataset.")
        return self._window_weights

    def _vectorize_window(
        self, window_data: Sequence[FrameData]
    ) -> npt.NDArray[np.float32]:
        """
        Vectorize a single window of data.

        Args:
            window_data: Window of data to vectorize. Must be window-size
                in length.

        Returns:
            Transformed vector.
        """
        assert len(window_data) == self.window_size
        v = self.vectorizer
        return np.asarray([v(d) for d in window_data])

    def load_data_offline(
        self,
        activity_coco: kwcoco.CocoDataset,
        dets_coco: kwcoco.CocoDataset,
        pose_coco: kwcoco.CocoDataset,
        target_framerate: float,  # probably 15
        framerate_round_decimals: int = 1,
        pre_vectorize: bool = True,
        pre_vectorize_cores: int = os.cpu_count(),
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Load data from filesystem resources for use during training.

        We will pre-compute window vectors to save time during training. We
        will attempt to cache these vectors if a cache directory is provided.

        Vector caching also requires that the input COCO datasets have an
        associated filepath that exists.

        Args:
            activity_coco:
                COCO dataset of per-frame activity classification ground truth.
                This dataset also serves as the authority for data processing
                alignment with co-input object detections and pose estimations.
            dets_coco:
                COCO Dataset of object detections inferenced on the video
                frames for which we have activity truth.
            pose_coco:
                COCO Dataset of pose data inferenced on the video frames for
                which we have activity truth.
            target_framerate:
                Target frame-rate to assert and normalize videos inputs to
                allow for the yielding of temporally consistent windows of
                data.
            framerate_round_decimals:
                Number of floating-point decimals to round to when considering
                frame-rates.
            pre_vectorize:
                If we should pre-compute window vectors, possibly caching the
                results, as part of this load.
            pre_vectorize_cores:
                Number of cores to utilize when pre-computing window vectors.
            cache_dir:
                Optional directory for cache file storage and retrieval. If
                this is not specified, no caching will occur.
        """
        # The data coverage for all the input datasets must be congruent.
        logger.info("Checking dataset video/image congruency")
        assert activity_coco.index.videos == dets_coco.index.videos  # noqa
        assert activity_coco.index.videos == pose_coco.index.videos  # noqa
        assert activity_coco.index.imgs == dets_coco.index.imgs
        assert activity_coco.index.imgs == pose_coco.index.imgs

        # Check that videos are, or are an integer multiple of, the target
        # framerate.
        logger.info("Checking video framerate multiples")
        misaligned_fr = False
        vid_id_to_fr_multiple: Dict[int, int] = {}
        for vid_id, vid_obj in activity_coco.index.videos.items():
            vid_fr: float = round(vid_obj["framerate"], framerate_round_decimals)
            vid_fr_multiple = round(vid_fr / target_framerate, framerate_round_decimals)
            if vid_fr_multiple.is_integer():
                vid_id_to_fr_multiple[vid_id] = int(vid_fr_multiple)
            else:
                logger.error(
                    f"Video ({vid_obj['name']}) framerate ({vid_fr}) is not a "
                    f"integer multiple of the target framerate "
                    f"({target_framerate}). Multiple found to be "
                    f"{vid_fr_multiple}."
                )
                misaligned_fr = True
        if misaligned_fr:
            raise ValueError(
                "Videos in the input dataset do not all align to the target "
                "framerate. See error logging."
            )

        # Introspect from COCO categories pose keypoint counts.
        # Later functionality assumes that all poses considered consist of the
        # same quantity of keypoints, i.e. we cannot support multiple pose
        # categories with **varying** keypoint quantities.
        pose_num_keypoints_set: Set[int] = {
            len(obj["keypoints"])
            for obj in pose_coco.categories().objs
            if "keypoints" in obj
        }
        if len(pose_num_keypoints_set) != 1:
            raise ValueError(
                "Input pose categories either did not contain any keypoints, "
                "or contained multiple keypoint-containing categories with "
                "varying quantities of keypoints. "
                f"Found sizes in the set: {pose_num_keypoints_set}"
            )
        num_pose_keypoints = list(pose_num_keypoints_set)[0]

        # For each video in dataset, collect "windows" of window_size entries.
        # Each entry needs to be a collection of:
        # * object detections on that frame
        # * pose predictions (including keypoints) for that frame
        #
        # Videos that are a multiple of the target framerate should be treated
        # as multiple videos, with sub-video is one frame offset from the
        # others. E.g. 15Hz target but 30Hz video, sub-video 1 starts at frame
        # 0 and taking every-other frame (30/15=2), sub-video 2 starts at frame
        # 1 and taking every-other frame, thus each video considers unique
        # frame products.

        # Default "empty" instances to share for frames with no detections or
        # poses. These are for storing empty, but shaped, arrays.
        empty_dets = FrameObjectDetections(
            np.ndarray(shape=(0, 4)),
            np.ndarray(shape=(0,)),
            np.ndarray(shape=(0,)),
        )
        empty_pose = FramePoses(
            np.ndarray(shape=(0,)),
            np.ndarray(shape=(0, num_pose_keypoints, 2)),
            np.ndarray(shape=(0, num_pose_keypoints)),
        )

        #
        # Collect per-frame data first per-video, then slice into windows.
        #

        # Windows of per-frame data that would go into producing a vector.
        window_data: List[List[FrameData]] = []

        # Activity classification truth labels per-frame per-window.
        window_truth: List[List[int]] = []

        # Video ID represented per window. Only one video should be represented
        # in any one window.
        window_vid: List[int] = []

        # Image ID per-frame per-window.
        window_frames: List[List[int]] = []

        # cache frequently called module functions
        np_asarray = np.asarray

        for vid_id in tqdm(activity_coco.videos()):
            vid_id: int
            vid_images = activity_coco.images(video_id=vid_id)
            vid_img_ids: List[int] = list(vid_images)
            vid_frames_all: List[int] = vid_images.lookup("frame_index")  # noqa
            # Iterate over sub-videos if applicable. See comment earlier in func.
            vid_fr_multiple = vid_id_to_fr_multiple[vid_id]
            for starting_idx in range(vid_fr_multiple):  # may just be a single [0]
                # video-local storage to keep things separate, will extend main
                # structures afterward.
                vid_frame_truth = []
                vid_frame_data = []
                vid_gid = vid_img_ids[starting_idx::vid_fr_multiple]
                vid_frames = vid_frames_all[starting_idx::vid_fr_multiple]
                for img_id in vid_gid:
                    img_id: int
                    img_act = activity_coco.annots(image_id=img_id)
                    img_dets = dets_coco.annots(image_id=img_id)
                    img_poses = pose_coco.annots(image_id=img_id)

                    # There should only be one activity truth annotation per
                    # image.
                    assert (
                        len(img_act) == 1
                    ), "Only a single activity label is allowed per image."
                    vid_frame_truth.append(img_act.cids[0])

                    # There may be no detections on this frame.
                    if img_dets:
                        frame_dets = FrameObjectDetections(
                            img_dets.boxes.data,
                            np_asarray(img_dets.cids, dtype=int),
                            np_asarray(img_dets.lookup("score"), dtype=float),
                        )
                    else:
                        frame_dets = empty_dets

                    # Only consider annotations that actually have keypoints.
                    # There may be no poses on this frame.
                    img_poses = img_poses.take(
                        [
                            idx
                            for idx, obj in enumerate(img_poses.objs)
                            if "keypoints" in obj
                        ]
                    )
                    if img_poses:
                        # Only keep the positions, drop visibility value.
                        # This will have a shape error if there are different
                        # pose classes with differing quantities of keypoints.
                        # !!!
                        # BASICALLY ASSUMING THERE IS ONLY ONE POSE CLASS WITH
                        # KEYPOINTS.
                        # !!!
                        kp_pos = np_asarray(img_poses.lookup("keypoints")).reshape(
                            -1, num_pose_keypoints, 3
                        )[:, :, :2]
                        frame_poses = FramePoses(
                            np_asarray(img_poses.lookup("score"), dtype=float),
                            kp_pos,
                            np_asarray(
                                img_poses.lookup("keypoint_scores"), dtype=float
                            ),
                        )
                    else:
                        frame_poses = empty_pose
                    vid_frame_data.append(FrameData(frame_dets, frame_poses))

                # Slide this video's worth of frame data into windows such that
                # each window is window_size long.
                # If this video has fewer frames than window_size, this video
                # effectively be skipped.
                vid_window_truth = []
                vid_window_data = []
                vid_window_vid = []  # just a single ID per window referencing video
                vid_window_frames = []  # Video frame numbers for frames of this window
                for i in range(len(vid_frame_data) - self.window_size):
                    vid_window_truth.append(vid_frame_truth[i : i + self.window_size])
                    vid_window_data.append(vid_frame_data[i : i + self.window_size])
                    vid_window_vid.append(vid_id)
                    vid_window_frames.append(vid_frames[i : i + self.window_size])

                window_truth.extend(vid_window_truth)
                window_data.extend(vid_window_data)
                window_vid.extend(vid_window_vid)
                window_frames.extend(vid_window_frames)

        self._window_data = window_data
        self._window_truth = np.asarray(window_truth)
        self._window_vid = np.asarray(window_vid)
        self._window_frames = np.asarray(window_frames)

        # Collect for weighting the truth labels for the final frames of
        # windows, which is the truth value for the window as a whole.
        window_final_class_ids = self._window_truth[:, -1]
        cls_ids, cls_counts = np.unique(window_final_class_ids, return_counts=True)
        # Some classes may not be represented in the truth, so initialize the
        # weights vector separately, and then assign weight values based on
        # which class IDs were actually represented.
        cls_weights = np.zeros(len(activity_coco.cats))
        cls_weights[cls_ids] = 1.0 / cls_counts
        self._window_weights = cls_weights[window_final_class_ids]

        # Check if there happens to be a cache file of pre-computed window
        # vectors available to load.
        #
        # Caching is even possible if:
        # * given a directory home for cache files
        # * input COCO dataset filepaths are real and can be checksum'ed.
        has_vector_cache = False
        cache_filepath = None
        activity_coco_fpath = Path(activity_coco.fpath)
        dets_coco_fpath = Path(dets_coco.fpath)
        pose_coco_fpath = Path(pose_coco.fpath)
        if (
            pre_vectorize
            and cache_dir is not None
            and activity_coco_fpath.is_file()
            and dets_coco_fpath.is_file()
            and pose_coco_fpath.is_file()
        ):
            csum = sha256()
            with open(activity_coco_fpath, "rb") as f:
                csum.update(f.read())
            with open(dets_coco_fpath, "rb") as f:
                csum.update(f.read())
            with open(pose_coco_fpath, "rb") as f:
                csum.update(f.read())
            csum.update(f"{target_framerate:0.{framerate_round_decimals}f}".encode())
            csum.update(f"{self.window_size:d}".encode())
            csum.update(json.dumps(self.vectorizer.hparams()).encode())
            # Include vectorization variables in the name of the file.
            # Note the "z" in the name, expecting to use savez_compressed.
            cache_filename = "{}.npz".format(csum.hexdigest())
            cache_filepath = Path(cache_dir) / cache_filename
            has_vector_cache = cache_filepath.is_file()

        if pre_vectorize:
            if has_vector_cache:
                logger.info("Loading window vectors from cache...")
                with np.load(cache_filepath) as data:
                    self._window_vectors = data["window_vectors"]
                logger.info("Loading window vectors from cache... Done")
            else:
                # Pre-vectorize data for iteration efficiency during training.
                # * Creating a mini Dataset/Dataloader situation to efficiently
                #   generate vectors.
                vectorize_window = self._vectorize_window
                window_vectors: List[npt.NDArray[np.float32]] = []

                class VecDset(Dataset):
                    def __getitem__(self, item):
                        return vectorize_window(window_data[item])

                    def __len__(self):
                        return len(window_data)

                # Using larger batch sizes than 1 did not show any particular
                # increase in throughput. This may require increasing
                # `ulimit -n`, though.
                dloader = DataLoader(
                    VecDset(),
                    batch_size=1,
                    num_workers=pre_vectorize_cores,
                )

                for batch in tqdm(
                    dloader,
                    desc="Windows vectorized",
                    unit="windows",
                ):
                    window_vectors.extend(batch.numpy())
                self._window_vectors = window_vectors

                if cache_filepath is not None:
                    logger.info("Saving window vectors to cache...")
                    cache_filepath.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache_filepath,
                        window_vectors=window_vectors,
                    )
                    logger.info("Saving window vectors to cache... Done")

    def load_data_online(
        self,
        window_data: Sequence[FrameData],
    ) -> None:
        """
        Receive data from a streaming runtime to yield from __getitem__.

        If any one frame has no object detections or poses estimated for it,
        `None` should be filled in the corresponding position(s).

        Args:
            window_data: Per-frame data to compose the solitary window.
        """
        # Just load one windows worth of stuff so only __getitem__(0) makes
        # sense.
        if len(window_data) != self.window_size:
            raise ValueError(
                f"Input sequences did not match the configured window size "
                f"({len(window_data)} != {self.window_size})."
            )

        # Assign a single window of frame data.
        self._window_data = [list(window_data)]
        # The following are undefined for online mode, so we're just filling in
        # 0's enough to match size/shape requirements.
        self._window_truth = np.zeros(shape=(1, self.window_size), dtype=int)
        self._window_vid = np.asarray([0])
        self._window_frames = np.asarray([list(range(self.window_size))])

    def __getitem__(
        self, index: int
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[int],
        npt.NDArray[int],
        npt.NDArray[int],
        npt.NDArray[int],
    ]:
        """
        Fetches the data point and defines its TCN vector.

        Args:
            index: The index of the data point.

        Returns:
            Series of 5 numpy arrays:
              * Embedding Vector, shape: (window_size, n_dims)
              * per-frame truth, shape: (window_size,)
              * per-frame applicability mask, shape: (window_size,)
              * per-frame video ID, shape: (window_size,)
              * per-frame image ID, shape: (window_size,)
        """
        window_data = self._window_data[index]
        window_truth = self._window_truth[index]
        window_vid = self._window_vid[index]
        window_frames = self._window_frames[index]

        window_vectors = self._window_vectors
        if window_vectors is not None:
            tcn_vector = window_vectors[index]
        else:
            tcn_vector = self._vectorize_window(window_data)

        # Augmentation has to happen on the fly and cannot be pre-computed due
        # to random aspects that augmentation can be configured to have during
        # training.
        if self.transform is not None:
            # TODO: Augment using a helper on the vectorizer? I'm imaging that
            #       augmentations might be specific to which vectorizer is
            #       used.
            tcn_vector = self.transform(tcn_vector)

        return (
            tcn_vector,
            window_truth,
            # Under the current operation of this dataset, the mask should always
            # consist of 1's. This may be removed in the future.
            self._ones_mask,
            np.repeat(window_vid, self.window_size),
            window_frames,
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            length: Length of the dataset.
        """
        return len(self._window_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage:
    activity_coco = kwcoco.CocoDataset(
        # "/home/local/KHQ/paul.tunison/data/darpa-ptg/train-TCN-M2_bbn_hololens/activity_truth.coco.json"
        "/home/local/KHQ/paul.tunison/data/darpa-ptg/train-TCN-M2_bbn_hololens/TEST-activity_truth.coco.json"
        # "/home/local/KHQ/paul.tunison/data/darpa-ptg/train-TCN-M2_bbn_hololens/TRAIN-activity_truth-vid_1.coco.json"
    )
    dets_coco = kwcoco.CocoDataset(
        # "/home/local/KHQ/paul.tunison/data/darpa-ptg/train-TCN-M2_bbn_hololens/all_object_detections.coco.json"
        "/home/local/KHQ/paul.tunison/data/darpa-ptg/train-TCN-M2_bbn_hololens/TEST-object_detections.coco.json"
        # "/home/local/KHQ/paul.tunison/data/darpa-ptg/train-TCN-M2_bbn_hololens/TRAIN-object_detections-vid_1.coco.json"
    )
    pose_coco = kwcoco.CocoDataset(
        # "/home/local/KHQ/paul.tunison/data/darpa-ptg/train-TCN-M2_bbn_hololens/all_poses.coco.json"
        "/home/local/KHQ/paul.tunison/data/darpa-ptg/train-TCN-M2_bbn_hololens/TEST-pose_estimates.coco.json"
        # "/home/local/KHQ/paul.tunison/data/darpa-ptg/train-TCN-M2_bbn_hololens/TRAIN-pose_estimates-vid_1.coco.json"
    )

    from tcn_hpl.data.vectorize.classic import Classic

    vectorizer = Classic(
        feat_version=6,
        top_k=1,
        num_classes=7,  # M2 object detection classes
        background_idx=0,
        hand_left_idx=5,
        hand_right_idx=6,
    )
    dataset = TCNDataset(window_size=25, vectorizer=vectorizer)
    dataset.load_data_offline(
        activity_coco, dets_coco, pose_coco, target_framerate=15, cache_dir=None
    )

    print(f"dataset: {len(dataset)}")
    batch_size = 512  # 16
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    count = 0
    s = time.time()
    for idx, batch in tqdm(
        enumerate(data_loader),
        desc="Iterating batches of features",
        unit="batches",
    ):
        count += 1
    duration = time.time() - s

    print(
        f"Total batches of size {batch_size}: {count} ({duration:.02f} seconds total)"
    )
