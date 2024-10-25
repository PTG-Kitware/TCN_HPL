import os
import yaml
import warnings
import kwcoco
import argparse
import kwimage

import numpy as np
import ubelt as ub

from pathlib import Path

from angel_system.activity_classification.train_activity_classifier import (
    data_loader,
    compute_feats,
)
from tcn_hpl.data.vectorize_classic import (
    feature_version_to_options,
    plot_feature_vec,
)

from angel_system.data.medical.data_paths import TASK_TO_NAME
from angel_system.data.medical.load_bbn_data import bbn_activity_txt_to_csv
from angel_system.data.common.kwcoco_utils import add_activity_gt_to_kwcoco


def bbn_to_dive(raw_data_root, dive_output_dir, task, label_mapping, label_version=1):
    used_videos = bbn_activity_txt_to_csv(
        task=task,
        root_dir=raw_data_root,
        output_dir=dive_output_dir,
        label_mapping=label_mapping,
        label_version=label_version,
    )
    return used_videos


def create_training_data(config_path):
    #####################
    # Input
    #####################
    # LOAD CONFIG
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    feat_version = config["feature_version"]
    topic = config["topic"]
    if topic == "medical":
        from angel_system.data.medical.load_bbn_data import time_from_name
    elif topic == "cooking":
        from angel_system.data.cooking.load_kitware_data import time_from_name

    task_name = config["task"]
    task_data_type = config["data_gen"]["data_type"]  # pro or lab
    filter_black_gloves = config["data_gen"].get("filter_black_gloves", False)
    black_glove_vids = config["data_gen"].get("black_glove_vids", [])
    filter_blue_gloves = config["data_gen"].get("filter_blue_gloves", False)
    blue_glove_vids = config["data_gen"].get("blue_glove_vids", [])
    activity_config_fn = config["data_gen"]["activity_config_fn"]
    top_k_objects = config["data_gen"]["top_k_objects"]
    pose_repeat_rate = config["data_gen"]["pose_repeat_rate"]
    exp_ext = config["data_gen"]["exp_ext"]
    raw_data_root = (
        f"{config['data_gen']['raw_data_root']}/{TASK_TO_NAME[task_name]}/Data"
    )
    dive_output_root = config["data_gen"]["dive_output_root"]

    def filter_dset_by_split(split, dataset_to_split):
        # Filter by video names
        video_lookup = dataset_to_split.index.name_to_video
        split_vidids = []
        split_img_ids = []
        for index in config["data_gen"][split]:
            video_name = f"{task_name.upper()}-{index}"
            print(f"video name: {video_name}")

            # Make sure we have the video
            if video_name in video_lookup:
                vid = video_lookup[video_name]["id"]
            else:
                warnings.warn(f"Video {video_name} is not present in the dataset")
                exit(1)

            # Optionally filter gloves
            if filter_black_gloves and index in black_glove_vids:
                continue
            if filter_blue_gloves and index in blue_glove_vids:
                continue

            split_vidids.append(video_name)
            split_img_ids.extend(list(dataset_to_split.index.vidid_to_gids[vid]))

        split_dset = dataset_to_split.subset(gids=split_img_ids, copy=True)
        return split_dset

    #####################
    # Output
    #####################
    exp_name = f"{task_name}_{task_data_type}_data_top_{top_k_objects}_objs_feat_v{feat_version}_pose_rate_{pose_repeat_rate}{exp_ext}"
    data_dir = f"/data/PTG/{topic}/training/activity_classifier"
    output_data_dir = f"{data_dir}/TCN_data/{task_name}/{exp_name}"
    print(output_data_dir)

    gt_dir = f"{output_data_dir}/groundTruth"
    frames_dir = f"{output_data_dir}/frames"
    bundle_dir = f"{output_data_dir}/splits"
    features_dir = f"{output_data_dir}/features"
    features_visualization_dir = f"{output_data_dir}/features_visualization"

    # Create directories
    for folder in [
        output_data_dir,
        gt_dir,
        frames_dir,
        bundle_dir,
        features_dir,
        features_visualization_dir,
    ]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    # Clear out the bundles
    filelist = [f for f in os.listdir(bundle_dir)]
    for f in filelist:
        os.remove(os.path.join(bundle_dir, f))

    #####################
    # Mapping
    #####################
    with open(activity_config_fn, "r") as stream:
        activity_config = yaml.safe_load(stream)
    activity_labels = activity_config["labels"]

    activity_labels_desc_mapping = {}
    with open(f"{output_data_dir}/mapping.txt", "w") as mapping:
        for label in activity_labels:
            i = label["id"]
            label_str = label["label"]
            if "description" in label.keys():
                # activity_labels_desc_mapping[label["description"]] = label["label"]
                activity_labels_desc_mapping[label["description"]] = label["id"]
            elif "full_str" in label.keys():
                # activity_labels_desc_mapping[label["full_str"]] = label["label"]
                activity_labels_desc_mapping[label["full_str"]] = label["id"]
            if label_str == "done":
                continue
            mapping.write(f"{i} {label_str}\n")

    print(f"label mapping: {activity_labels_desc_mapping}")
    # exit()
    dset = kwcoco.CocoDataset(config["data_gen"]["dataset_kwcoco"])
    # Check if the dest has activity gt, if it doesn't then add it
    if not "activity_gt" in list(dset.imgs.values())[0].keys():
        print("adding activity ground truth to the dataset")

        # generate dive files for videos in dataset if it does not exist
        video_id = list(dset.index.videos.keys())[0]
        video = dset.index.videos[video_id]
        video_name = video["name"]
        activity_gt_dir = f"{dive_output_root}/{task_name}_labels/"
        activity_gt_fn = f"{activity_gt_dir}/{video_name}_activity_labels_v1.csv"
        print(f"activity_gt_dir: {activity_gt_dir}, activity_gt_fn: {activity_gt_fn}")

        used_videos = bbn_to_dive(
            raw_data_root, activity_gt_dir, task_name, activity_labels_desc_mapping
        )
        video_ids_to_remove = [
            vid
            for vid, value in dset.index.videos.items()
            if value["name"] not in used_videos
        ]
        dset.remove_videos(video_ids_to_remove)

        dset = add_activity_gt_to_kwcoco(topic, task_name, dset, activity_config_fn)

    #####################
    # Features,
    # groundtruth and
    # bundles
    #####################
    for split in ["train", "val", "test"]:
        print(f"==== {split} ====")
        split_dset = filter_dset_by_split(f"{split}_vid_ids", dset)
        print(f"{split} num_images: {len(split_dset.imgs)}")

        for video_id in ub.ProgIter(
            split_dset.index.videos.keys(),
            desc=f"Creating features for videos in {split}",
        ):
            video = split_dset.index.videos[video_id]
            video_name = video["name"]
            if "_extracted" in video_name:
                video_name = video_name.split("_extracted")[0]

            image_ids = sorted(split_dset.index.vidid_to_gids[video_id])
            num_images = len(image_ids)

            print(f"there are {num_images} images in video {video_id}")

            video_dset = split_dset.subset(gids=image_ids, copy=True)

            # groundtruth
            with open(f"{gt_dir}/{video_name}.txt", "w") as gt_f, open(
                f"{frames_dir}/{video_name}.txt", "w"
            ) as frames_f:
                for image_id in image_ids:
                    image = split_dset.imgs[image_id]
                    image_n = image["file_name"]  # this is the shortened string

                    frame_idx, time = time_from_name(image_n)

                    activity_gt = image["activity_gt"]
                    if activity_gt is None:
                        activity_gt = "background"

                    gt_f.write(f"{activity_gt}\n")
                    frames_f.write(f"{image_n}\n")

            # features
            (
                act_map,
                inv_act_map,
                image_activity_gt,
                image_id_to_dataset,
                obj_label_to_ind,
                obj_ind_to_label,
                ann_by_image,
            ) = data_loader(video_dset, activity_config)
            X, y = compute_feats(
                act_map,
                image_activity_gt,
                image_id_to_dataset,
                obj_label_to_ind,
                obj_ind_to_label,
                ann_by_image,
                feat_version=feat_version,
                top_k_objects=top_k_objects,
                pose_repeat_rate=pose_repeat_rate,
            )
            print(X.shape)

            # Draw the feature vector for the first video
            if video_id == list(split_dset.index.videos.keys())[0]:
                opts = feature_version_to_options(feat_version)
                gid_to_aids = dset.index.gid_to_aids
                for image_id, feature_vec in zip(image_ids, X):
                    image = split_dset.imgs[image_id]
                    image_fn = image["file_name"]
                    print("fn", image_fn)

                    aids = gid_to_aids[image_id]
                    anns = ub.dict_subset(dset.anns, aids)

                    default_center = [0, 0]
                    right_hand_center = default_center
                    left_hand_center = default_center
                    for aid, ann in anns.items():
                        cat_id = ann["category_id"]
                        cat = dset.cats[cat_id]["name"]

                        if cat == "hand (right)":
                            right_hand_center = kwimage.Boxes(
                                [ann["bbox"]], "xywh"
                            ).center
                            right_hand_center = [
                                right_hand_center[0][0][0],
                                right_hand_center[1][0][0],
                            ]
                        if cat == "hand (left)":
                            left_hand_center = kwimage.Boxes(
                                [ann["bbox"]], "xywh"
                            ).center
                            left_hand_center = [
                                left_hand_center[0][0][0],
                                left_hand_center[1][0][0],
                            ]

                    plot_feature_vec(
                        image_fn,
                        right_hand_center,
                        left_hand_center,
                        feature_vec,
                        obj_label_to_ind,
                        output_dir=os.path.join(features_visualization_dir, video_name),
                        top_k_objects=top_k_objects,
                        **opts,
                    )

            X = X.T
            print(f"X after transpose: {X.shape}")

            np.save(f"{features_dir}/{video_name}.npy", X)

            # bundles
            with open(f"{bundle_dir}/{split}.split1.bundle", "a+") as bundle:
                bundle.write(f"{video_name}.txt\n")

    print("Done!")
    print(f"Saved training data to {output_data_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", default="configs/experiment/r18/feat_v6.yaml", help=""
    )

    args = parser.parse_args()

    create_training_data(args.config)


if __name__ == "__main__":
    main()
