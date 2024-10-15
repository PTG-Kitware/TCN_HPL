import argparse

import kwcoco
import tcn_hpl.utils.utils as utils
import ubelt as ub
import yaml

from angel_system.data.medical.data_paths import LAB_TASK_TO_NAME

from tcn_hpl.data.utils.bbn import convert_truth_to_array


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    task_name = config["task"]
    raw_data_root = (
        f"{config['data_gen']['raw_data_root']}/{LAB_TASK_TO_NAME[task_name]}/"
    )

    dset = kwcoco.CocoDataset(config["data_gen"]["dataset_kwcoco"])

    with open(config["data_gen"]["activity_config_fn"], "r") as stream:
        activity_config = yaml.safe_load(stream)
    activity_labels = activity_config["labels"]

    activity_labels_desc_mapping = {}
    activity_labels_label_mapping = {}
    for label in activity_labels:
        i = label["id"]
        label_str = label["label"]
        if "description" in label.keys():
            activity_labels_desc_mapping[label["description"]] = label["id"]
        elif "full_str" in label.keys():
            activity_labels_desc_mapping[label["full_str"]] = label["id"]
            activity_labels_label_mapping[label_str] = label["id"]
        if label_str == "done":
            continue

    gt_paths_to_names_dict = {}
    gt_paths = utils.dictionary_contents(raw_data_root, types=["*.txt"])
    for gt_path in gt_paths:
        name = gt_path.split("/")[-1].split(".")[0]
        gt_paths_to_names_dict[name] = gt_path

    print(gt_paths_to_names_dict)

    if not "activity_gt" in list(dset.imgs.values())[0].keys():
        print("adding activity ground truth to the dataset")
        for video_id in ub.ProgIter(dset.index.videos.keys()):
            video = dset.index.videos[video_id]
            video_name = video["name"]
            if video_name in gt_paths_to_names_dict.keys():
                gt_text = gt_paths_to_names_dict[video_name]
            else:
                print(f"GT file does not exist for {video_name}. Continue...")
                continue

            image_ids = dset.index.vidid_to_gids[video_id]
            num_frames = len(image_ids)

            activity_gt_list = convert_truth_to_array(
                gt_text, num_frames, activity_labels_desc_mapping
            )

            for index, img_id in enumerate(image_ids):
                im = dset.index.imgs[img_id]
                frame_index = int(im["frame_index"])
                dset.index.imgs[img_id]["activity_gt"] = activity_gt_list[frame_index]

    dset.dump("test.mscoco.json", newlines=True)
    print(activity_labels_desc_mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="/home/local/KHQ/peri.akiva/projects/TCN_HPL/configs/experiment/r18/feat_v6.yaml",
        help="",
    )

    args = parser.parse_args()

    main(args.config)
