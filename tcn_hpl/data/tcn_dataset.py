import click
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
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from tcn_hpl.data.frame_data import (
    FrameObjectDetections,
    FramePoses,
    FrameData,
)
from tcn_hpl.data.vectorize import Vectorize


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

        # FrameData for the total set of frames.
        # This is not being stored as a ndarray due to its variable nature.
        self._frame_data: List[FrameData] = []
        # Content to be indexed into during __getitem__ that refers to which
        # indices of self._frame_data compose that window index.
        # Shape: (n_windows, window_size)
        self._window_data_idx: Optional[npt.NDArray[int]] = None
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
        # available when there is truth available, i.e. during offline mode.
        self._window_weights: Optional[npt.NDArray[float]] = None
        # Optionally defined set of pre-computed vectors for each frame.
        # Congruent index association with self._frame_data, so
        # self._window_data_idx values may be used here.
        # Shape: (n_frames, feat_dim)  # see self._frame_data
        self._frame_vectors: Optional[npt.NDArray[np.float32]] = None

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

        # FrameData instances for each frame of each video. Each entry here
        # would ostensibly be transformed into a vector.
        frame_data: List[FrameData] = []

        # Windows specifying which frames are a part of that window via index
        # reference into frame_data.
        # Shape: (n_windows, window_size)
        window_data_idx: List[List[int]] = []

        # Activity classification truth labels per-frame per-window.
        # Shape: (n_windows, window_size)
        window_truth: List[List[int]] = []

        # Video ID represented per window. Only one video should be represented
        # in any one window.
        # Shape: (n_windows,)
        window_vid: List[int] = []

        # Image ID per-frame per-window.
        # Shape: (n_windows, window_size)
        window_frames: List[List[int]] = []

        # cache frequently called module functions
        np_asarray = np.asarray

        for vid_id in tqdm(activity_coco.videos()):
            vid_id: int
            vid_images = activity_coco.images(video_id=vid_id)
            vid_img_ids: List[int] = list(vid_images)
            vid_frames_all: List[int] = vid_images.lookup("frame_index")  # noqa
            # Iterate over sub-videos if applicable. This should only turn out
            # to be some integer >= 1. See comment earlier in func.
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
                    
                    # Frame height and width should be available.
                    img_info = activity_coco.index.imgs[img_id]
                    assert "height" in img_info
                    assert "width" in img_info
                    frame_size = (img_info["width"], img_info["height"])

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
                    # import ipdb; ipdb.set_trace()
                    vid_frame_data.append(FrameData(frame_dets, frame_poses, frame_size))

                # Compose a list of indices into frame_data that this video's
                # worth of content resides.
                vid_frame_data_idx: List[int] = list(
                    range(
                        len(frame_data),
                        len(frame_data) + len(vid_frame_data),
                    )
                )
                frame_data.extend(vid_frame_data)

                # Slide this video's worth of frame data into windows such that
                # each window is window_size long.
                # If this video has fewer frames than window_size, this video
                # effectively be skipped.
                vid_window_truth: List[List[int]] = []
                vid_window_data_idx: List[List[int]] = []
                # just a single ID per window referencing the video that window
                # is pertaining to.
                vid_window_vid: List[int] = []
                # Video frame numbers for frames in windows.
                vid_window_frames = []
                for i in range(len(vid_frame_data) - self.window_size):
                    vid_window_truth.append(vid_frame_truth[i : i + self.window_size])
                    vid_window_data_idx.append(
                        vid_frame_data_idx[i : i + self.window_size]
                    )
                    vid_window_vid.append(vid_id)
                    vid_window_frames.append(vid_frames[i : i + self.window_size])

                window_truth.extend(vid_window_truth)
                window_data_idx.extend(vid_window_data_idx)
                window_vid.extend(vid_window_vid)
                window_frames.extend(vid_window_frames)

        self._frame_data = frame_data
        self._window_data_idx = np.asarray(window_data_idx)
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
            csum.update(self.vectorizer.__class__.__module__.encode())
            csum.update(self.vectorizer.__class__.__name__.encode())
            csum.update(json.dumps(self.vectorizer.hparams()).encode())
            # Include vectorization variables in the name of the file.
            # Note the "z" in the name, expecting to use savez_compressed.
            cache_filename = "{}.npz".format(csum.hexdigest())
            cache_filepath = Path(cache_dir) / cache_filename
            has_vector_cache = cache_filepath.is_file()

        if pre_vectorize:
            if has_vector_cache:
                logger.info("Loading frame vectors from cache...")
                with np.load(cache_filepath) as data:
                    self._frame_vectors = data["frame_vectors"]
                logger.info("Loading frame vectors from cache... Done")
            else:
                # Pre-vectorize data for iteration efficiency during training.
                # * Creating a mini Dataset/Dataloader situation to efficiently
                #   generate vectors.

                # Set the sharing strategy to filesystem for the duration of
                # this operation, and then restoring the existing strategy
                # after we're done.
                current_sharing_strategy = torch.multiprocessing.get_sharing_strategy()

                try:
                    # This iteration seems to go twice as fast when utilizing
                    # the file-system strategy.
                    torch.multiprocessing.set_sharing_strategy("file_system")

                    vec_dset = _VectorizationDataset(self.vectorizer, frame_data)

                    # Using larger batch sizes than 1 did not show any particular
                    # increase in throughput. This may require increasing
                    # `ulimit -n`, though.
                    dloader = DataLoader(
                        vec_dset,
                        batch_size=1,
                        num_workers=pre_vectorize_cores,
                        # Required, especially for large dataset sizes, so the
                        # dataloader multiprocessing iteration does not exhaust
                        # shared memory.
                        pin_memory=True,
                    )

                    frame_vectors: List[npt.NDArray[np.float32]] = []
                    for batch in tqdm(
                        dloader,
                        desc="Frame data vectorized",
                        unit="frames",
                    ):
                        frame_vectors.extend(batch.numpy())
                finally:
                    torch.multiprocessing.set_sharing_strategy(current_sharing_strategy)

                self._frame_vectors = np.asarray(frame_vectors)

                if cache_filepath is not None:
                    logger.info("Saving window vectors to cache...")
                    cache_filepath.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        cache_filepath,
                        frame_vectors=frame_vectors,
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
        window_size = self.window_size

        # Assign a single window of frame data.
        self._frame_data = list(window_data)
        # Make sure it has shape of (1, window_size) with the reshape.
        self._window_data_idx = np.arange(window_size, dtype=int).reshape(1, -1)
        # The following are undefined for online mode, so we're just filling in
        # 0's enough to match size/shape requirements.
        self._window_truth = np.zeros(shape=(1, window_size), dtype=int)
        self._window_vid = np.asarray([0])
        self._window_frames = self._window_data_idx

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
        frame_data = self._frame_data
        window_data_idx = self._window_data_idx[index]
        window_truth = self._window_truth[index]
        window_vid = self._window_vid[index]
        window_frames = self._window_frames[index]

        frame_vectors = self._frame_vectors
        if frame_vectors is not None:
            window_mat = frame_vectors[window_data_idx]
        else:
            vectorizer = self.vectorizer
            window_mat = np.asarray(
                [vectorizer(frame_data[idx]) for idx in window_data_idx]
            )

        # Augmentation has to happen on the fly and cannot be pre-computed due
        # to random aspects that augmentation can be configured to have during
        # training.
        if self.transform is not None:
            # TODO: Augment using a helper on the vectorizer? I'm imaging that
            #       augmentations might be specific to which vectorizer is
            #       used.
            window_mat = self.transform(window_mat)

        return (
            window_mat,
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
        return len(self._window_data_idx) if self._window_data_idx is not None else 0


class _VectorizationDataset(Dataset):
    """
    Helper dataset for iterating over individual frames of data and producing
    embedding vectors.
    """

    def __init__(self, vectorize: Vectorize, frame_data: Sequence[FrameData]):
        self.vectorize = vectorize
        self.frame_data = frame_data

    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, item):
        return self.vectorize(self.frame_data[item])


@click.command()
@click.help_option("-h", "--help")
@click.argument("activity_coco", type=click.Path(path_type=Path))
@click.argument("detections_coco", type=click.Path(path_type=Path))
@click.argument("pose_coco", type=click.Path(path_type=Path))
@click.option(
    "--window-size",
    type=int,
    default=25,
    show_default=True,
)
@click.option(
    "--target-framerate",
    type=float,
    default=15,
    show_default=True,
)
@click.option(
    "--pre-vectorize",
    is_flag=True,
    help="Run pre-vectorization or not.",
    show_default=True,
)
def test_dataset_for_input(
    activity_coco: Path,
    detections_coco: Path,
    pose_coco: Path,
    window_size: int,
    target_framerate: float,
    pre_vectorize: bool,
):
    """
    Test the TCN Dataset iteration over some test data.
    """
    logging.basicConfig(level=logging.INFO)

    activity_coco = kwcoco.CocoDataset(activity_coco)
    dets_coco = kwcoco.CocoDataset(detections_coco)
    pose_coco = kwcoco.CocoDataset(pose_coco)

    # TODO: Some method of configuring which vectorizer to use.
    from tcn_hpl.data.vectorize.locs_and_confs import LocsAndConfs
    vectorizer = LocsAndConfs(
        top_k = 1,
        num_classes = 7,
        use_joint_confs = True,
        use_pixel_norm = True,
        use_hand_obj_offsets = False,
        background_idx = 0
    )

    dataset = TCNDataset(window_size=window_size, vectorizer=vectorizer)
    dataset.load_data_offline(
        activity_coco,
        dets_coco,
        pose_coco,
        target_framerate=target_framerate,
        pre_vectorize=pre_vectorize,
    )

    logger.info(f"Number of windows: {len(dataset)}")

    # Get vector dimensionality
    window_vecs = dataset[0]
    logger.info(f"Feature vector dims: {window_vecs[0].shape[1]}")

    # Test that we can iterate over the dataset using a DataLoader with
    # shuffling.
    batch_size = 512  # 16
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    count = 0
    s = time.time()
    for batch in tqdm(
        data_loader,
        desc="Iterating batches of features",
        unit="batches",
    ):
        count += 1
    duration = time.time() - s
    logger.info(f"Iterated over the full TCN Dataset in {duration:.2f} s.")

    # Test creating online mode with subset of data from above.
    dset_online = TCNDataset(window_size=window_size, vectorizer=vectorizer)
    dset_online.load_data_online(dataset._frame_data[:window_size])  # noqa
    assert len(dset_online) == 1, "Online dataset should be size 1"
    _ = dset_online[0]
    failed_index_error = True
    try:
        # Should index error
        dset_online[1]
    except IndexError:
        failed_index_error = False
    assert not failed_index_error, "Should have had an index error at [1]"
    assert (
        (dataset[0][0] == dset_online[0][0]).all()  # noqa
    ), "Online should have produced same window matrix as offline version."


if __name__ == "__main__":
    test_dataset_for_input()
