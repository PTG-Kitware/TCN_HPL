#!/usr/bin/env python3

import dataclasses
import os
from pathlib import Path
import re
import typing
import warnings

import click
import cv2
import kwcoco
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import yaml


# Regex to match a BBN Truth file and parse out the "basename" for which we
# should find a matching `.mp4` file next to it.
RE_TRUTH_FILENAME = re.compile(r"^(?P<basename>.*)\.skill_labels_by_frame\.txt$")

# Parsing a BBN Truth file line into component parts.
# Assumes that surrounding whitespace has been stripped.
RE_BBN_TRUTH_LINE = re.compile(
    r"^(?P<start_frame>\d+)\s+(?P<end_frame>\d+)\s+(?P<task_name>[\w\d]+)\s+"
    r"(?:Error: (?P<error>.*) S\s+)?(?P<description>.*)$"
)


@dataclasses.dataclass
class VideoInfo:
    truth_file: Path
    mp4_file: Path
    frames_dir: Path = dataclasses.field(init=False)
    num_frames: int = dataclasses.field(init=False)
    fps: float = dataclasses.field(init=False)
    frame_size: typing.Tuple[int, int] = dataclasses.field(init=False)


def extract_bbn_video_frames(
    video_path: Path, output_directory: Path
) -> typing.Tuple[int, float, typing.Tuple[int, int]]:
    """
    Extract the frames of a BBN MP4 video into a target directory.

    Side effect: Frame files will be output to the given directory following
    the naming format "%05d.png" where %05d is an integer index starting at 0.
    If this directory already exists and contains a number of files equal to
    the number of frames in the given video, this will do nothing.

    :param video_path: Path to the MP4 video file.
    :param output_directory: Path to output video frames to.

    :returns: Integer number of frames in the input video, the fps of the
        video, and the pixel resolution in (height, width) format.
    """
    assert video_path.is_file()
    video = cv2.VideoCapture(video_path.as_posix())
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # If the directory exists and has a number of files in it matching the
    # quantity of frames in the video, we assume that this is already done.
    # Otherwise, progress for each frame, writing out the frame file if it does
    # not already exist in the directory.
    if (
        not output_directory.is_dir()
        or len(list(output_directory.iterdir())) != num_frames
    ):
        output_directory.mkdir(exist_ok=True)
        for i in tqdm(
            range(int(num_frames)),
            desc=f"Extracting frames from {video_path.name}",
            unit="frame",
        ):
            ret, frame = video.read()
            frame_filepath = output_directory / f"{i:05d}.png"
            if not frame_filepath.is_file():
                cv2.imwrite(frame_filepath.as_posix(), frame)

    return num_frames, fps, (frame_h, frame_w)


def convert_truth_to_array(
    text_filepath: Path, num_frames: int, id_mapping: typing.Dict[str, int]
) -> npt.NDArray[int]:
    """
    Convert a "skill_labels_by_frame" text truth file from BBN into labels for
    the given task and number of frames.

    **Frame Ranges**
    Truth files only specify ranges of frames (assumed inclusive) that a
    denoted step applies. All other frames are assumed to be ID 0, or
    "background".

    **Task Step Errors**
    Truth files seem to have a specification where some task steps played out
    have a known "error" with them, that is detailed in the annotation. These
    are separated out from the description but are not currently utilized for
    anything.

    :param text_filepath: Filesystem path to the BBN Truth text file.
    :param num_frames: Number of expected frames in the video to which this
        truth file pertains.
    :param id_mapping: Mapping of step descriptions to the integer ID of that
        step class.

    :raises KeyError: If we have no ID mapping for the description in the truth
        file. This likely means there is a typo in the truth file, or our
        classification configuration needs updating.

    :returns: Array of integers specifying the class ID for each frame in that
        video.
    """
    activity_gt = np.zeros(num_frames, dtype=int)
    # check on overlapping truth
    prev_end_frame = 0

    with open(text_filepath) as f:
        for l in f:
            m = RE_BBN_TRUTH_LINE.match(l.strip())
            if m:
                start_frame, end_frame, task, error, description = m.groups()
                # Not using annotated error indication currently.
                start_frame = int(start_frame)
                end_frame = int(end_frame)
                if start_frame < prev_end_frame:
                    warnings.warn(f"Found overlapping truth in '{text_filepath}'")
                if end_frame >= num_frames:
                    warnings.warn(f"Found end frame beyond video frame count, ignoring trailing: {text_filepath}")
                assert (
                    start_frame <= end_frame
                ), f"Found start/end violation ({start_frame} !< {end_frame}) in {text_filepath}"
                try:
                    step_id = id_mapping[description]
                except KeyError:
                    warnings.warn(f"Found key error in truth file: {text_filepath}")
                    raise
                activity_gt[start_frame:end_frame] = step_id
                prev_end_frame = end_frame

    return activity_gt


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "bbn_truth_root",
    type=click.Path(
        exists=True, file_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
)
@click.argument(
    "working_directory",
    type=click.Path(exists=False, file_okay=False, resolve_path=True, path_type=Path),
)
@click.argument(
    "activity_label_config",
    type=click.Path(
        exists=True, dir_okay=False, readable=True, resolve_path=True, path_type=Path
    ),
)
@click.argument(
    "output_coco_filepath",
    type=click.Path(dir_okay=False, resolve_path=True, path_type=Path),
)
@click.option(
    "--relative",
    is_flag=True,
    default=False,
    help=(
        "Output relative filepaths for videos and images in the COCO JSON "
        "file instead of relative paths. Relative paths will be rooted at the "
        "truth root directory and working directory, respecively."
    ),
)
def create_truth_coco(
    bbn_truth_root: Path,
    working_directory: Path,
    activity_label_config: Path,
    output_coco_filepath: Path,
    relative: bool,
) -> None:
    """
    Extract the component frames aof a directory of MP4 videos that have an
    associated "*.skill_labels_by_frame.txt" activity classification truth
    files into a configured output directory root.

    Videos will need to be exploded out into component video frames. This will
    be achieved via the `cv2.VideoCapture` functionality and will be output
    into a target working directory.
    We will want to extract all frames from the found video files, however not
    all of them are the same frame-rate or resolution.

    \b
    Positional Arguments:
        BBN_TRUTH_ROOT
            Root directory under which MP4 video files and paired
            *.skill_labels_by_frame.txt files are located.
        WORKING_DIRECTORY
            Root directory into which extracted MP4 video are located (should be
            extracted into).
        ACTIVITY_LABEL_CONFIG
            Path to the PTG-Angel system configuration file for activity labels,
            IDs and expected full-text strings to match against in truth files.
            E.g. `angel_system/config/activity_labels/medical/m2.yaml`.
            Parts of this will assume that the notional "background" class is ID 0.
        OUTPUT_COCO_FILEPATH
            Path to where the output COCO JSON file should be written to. If this
            is given with a `.zip` extension, then it will be compressed up into an
            archive.

    \b
    Example Invocation:
        $ bbn_create_truth_coco \\
            ~/data/darpa-ptg/bbn_data/lab_data-golden/m2_tourniquet/positive/3_tourns_122023 \\
            ~/data/darpa-ptg/bbn_data/lab_data-working/m2_tourniquet/positive/3_tourns_122023 \\
            ~/dev/darpa-ptg/angel_system/config/activity_labels/medical/m2.yaml \\
            ~/data/darpa-ptg/bbn_data/lab_data-working/m2_tourniquet/positive/3_tourns_122023-activity_truth.json
    """
    working_directory.mkdir(exist_ok=True)

    # Discover MP4 and truth text file pairs recursively.
    # video_info's keys should be a type that we can sort to perform actions
    # later in a deterministic order.
    video_info: typing.Dict[Path, VideoInfo] = {}
    for dirpath, dirnames, filenames in os.walk(bbn_truth_root):
        dirpath = Path(dirpath)
        for fname in filenames:
            m = RE_TRUTH_FILENAME.match(fname)
            if m is not None:
                # Check for matching video file
                truthpath = dirpath / fname
                videopath = dirpath / f"{m.groupdict()['basename']}.mp4"
                if videopath.is_file():
                    # We have a successful pair, register
                    video_info[dirpath / fname] = VideoInfo(truthpath, videopath)
                else:
                    warnings.warn(f"Found truth file without matching MP4: {truthpath}")

    ordered_vi_keys = sorted(video_info)

    # Pre-process video files into directories of frames.
    # TODO: Could use thread-pool and submit a job per video.
    for vi_key in tqdm(
        ordered_vi_keys,
        desc="Extracting frames from videos",
        unit="videos",
    ):
        vi = video_info[vi_key]
        frames_output_directory = working_directory / vi.mp4_file.relative_to(
            bbn_truth_root
        ).with_suffix(".frames")
        vi.frames_dir = frames_output_directory
        vi.num_frames, vi.fps, vi.frame_size = extract_bbn_video_frames(
            vi.mp4_file, frames_output_directory
        )

    # Home for our video, image and per-frame truth annotations.
    truth_ds = kwcoco.CocoDataset(img_root=working_directory.as_posix())

    # Prepopulate category metadata from config file.
    with open(activity_label_config) as f:
        config = yaml.safe_load(f)
    if config["version"] != "1":
        # If we grow additional versions, spin out methods to migrate to
        # the current format.
        raise RuntimeError("Unsupported version of activity label configuration.")
    # For when parsing the BBN truth files, we need a step description to ID
    # int mapping.
    map_descr_to_id: typing.Dict[str, int] = {}
    for item in config["labels"]:
        truth_ds.ensure_category(item["label"], id=item["id"])
        map_descr_to_id[item["full_str"]] = item["id"]

    for vi_key in tqdm(
        ordered_vi_keys,
        desc="Parsing video truth",
        unit="files",
    ):
        vi = video_info[vi_key]

        # Get the category IDs for each frame as specified in the GT file.
        frame_activity_gt = convert_truth_to_array(vi.truth_file, vi.num_frames, map_descr_to_id)

        # Video "name" is the relative path to the video file.
        vid_file = vi.mp4_file
        if relative:
            vid_file = vi.mp4_file.relative_to(bbn_truth_root)
        vid = truth_ds.ensure_video(
            vid_file.as_posix(),
            framerate=vi.fps,
        )
        frame_files = sorted(vi.frames_dir.iterdir())
        assert len(frame_activity_gt) == len(frame_files)
        for i, (gt_id, frame_file) in enumerate(zip(frame_activity_gt, frame_files)):
            gt_id: int
            frame_file: Path
            assert frame_file.is_file()
            if relative:
                frame_file = frame_file.relative_to(working_directory)
            gid = truth_ds.ensure_image(
                frame_file.as_posix(),
                video_id=vid,
                frame_index=i,
                height=vi.frame_size[0],
                width=vi.frame_size[1],
            )
            truth_ds.add_annotation(gid, gt_id)

    with open(output_coco_filepath, "w") as f:
        truth_ds.dump(f, newlines=True)


if __name__ == "__main__":
    create_truth_coco()
