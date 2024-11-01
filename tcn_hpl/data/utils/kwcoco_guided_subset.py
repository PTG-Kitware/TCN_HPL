from os.path import exists
from pathlib import Path

import click
import kwcoco


@click.command()
@click.help_option("-h", "--help")
@click.argument(
    "INPUT_COCO_FILEPATH",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "GUIDE_COCO_FILEPATH",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.argument(
    "OUTPUT_COCO_FILEPATH",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
def main(
    input_coco_filepath: Path,
    guide_coco_filepath: Path,
    output_coco_filepath: Path,
):
    """
    CLI Utility to create a subset of a CocoDataset based on the image/video
    content of some other dataset.

    This tool will assert that the video and image content of the guide dataset
    matches content present in the input dataset, as this filtering only makes
    sense if this is true.

    \b
    Positional Arguments:
        INPUT_COCO_FILEPATH:
            Path to the COCO JSON file to be filtered into a subset.
        GUIDE_COCO_FILEPATH
            Path to the COCO JSON file to provide the video/image content to
            guide the filtering.
        OUTPUT_COCO_FILEPATH
            Path to where we should save the output COCO JSON file.
    """
    dset_input = kwcoco.CocoDataset(input_coco_filepath)
    dset_guide = kwcoco.CocoDataset(guide_coco_filepath)

    # Assert that guide dataset video and image ID content is present in the
    # input dataset
    assert bool(dset_input.videos()) == bool(
        dset_guide.videos()
    ), "Input or guide has videos, but the other doesn't!"
    if dset_input.videos():
        # ensure video content in guide is present in input and matches exactly
        guide_vid_diff = set(dset_guide.videos()).difference(dset_input.videos())
        assert (
            not guide_vid_diff
        ), f"Guide dataset has video IDs not present in the input dataset: {guide_vid_diff}"
        unmatched_guide_vid = [
            vid
            for vid in dset_guide.videos()
            if dset_guide.index.videos[vid] != dset_input.index.videos[vid]
        ]
        assert (
            not unmatched_guide_vid
        ), f"Some guide videos are not present exactly in input dset: {unmatched_guide_vid}"
    guide_gid_diff = set(dset_guide.images()).difference(dset_input.images())
    assert (
        not guide_gid_diff
    ), f"Guide dataset has image IDs not present in the input dataset: {guide_gid_diff}"
    unmatched_guide_gid = [
        gid
        for gid in dset_guide.images()
        if dset_guide.index.imgs[gid] != dset_input.index.imgs[gid]
    ]
    assert (
        not unmatched_guide_gid
    ), f"Some guide images are not present exactly in the input dset: {unmatched_guide_gid}"

    dset_subset: kwcoco.CocoDataset = dset_input.subset(dset_guide.images().gids)
    output_coco_filepath.parent.mkdir(parents=True, exist_ok=True)
    dset_subset.dump(
        output_coco_filepath,
        newlines=True,
    )
