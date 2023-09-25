import os
import cv2
import yaml
import glob
import kwcoco
import kwimage
import textwrap
import warnings
import random

import numpy as np
import ubelt as ub
import pandas as pd
import ubelt as ub

from pathlib import Path
from PIL import Image

from angel_system.data.common.load_data import (
    activities_from_dive_csv,
    objs_as_dataframe,
    time_from_name,
    sanitize_str,
)
from angel_system.berkeley.data.update_dets_utils import load_hl_hand_bboxes


def load_kwcoco(dset):
    """Load a kwcoco dataset from file

    :param dset: kwcoco object or a string pointing to a kwcoco file

    :return: The loaded kwcoco object
    :rtype: kwcoco.CocoDataset
    """
    # Load kwcoco file
    if type(dset) == str:
        dset_fn = dset
        dset = kwcoco.CocoDataset(dset_fn)
        dset.fpath = dset_fn
        print(f"Loaded dset from file: {dset_fn}")
    return dset


def preds_to_kwcoco(
    metadata,
    preds,
    save_dir,
    save_fn="result-with-contact.mscoco.json",
):
    """Save the predicitions in a kwcoco file
    used by the detector training

    :param metadata: Metadata dict of the dataset
    :param preds: Object detection results
    :param save_dir: Directory to save the kwcoco file to
    :param save_fn: The name of the resulting kwcoco file

    :return: The kwcoco object
    :rtype: kwcoco.CocoDataset
    """
    dset = kwcoco.CocoDataset()

    for class_ in metadata["thing_classes"]:
        dset.add_category(name=class_)

    for video_name, predictions in preds.items():
        dset.add_video(name=video_name)
        vid = dset.index.name_to_video[video_name]["id"]

        for time_stamp in sorted(predictions.keys()):
            dets = predictions[time_stamp]
            fn = dets["meta"]["file_name"]

            activity_gt = (
                dets["meta"]["activity_gt"]
                if "activity_gt" in dets["meta"].keys()
                else None
            )

            dset.add_image(
                file_name=fn,
                video_id=vid,
                frame_index=dets["meta"]["frame_idx"],
                width=dets["meta"]["im_size"]["width"],
                height=dets["meta"]["im_size"]["height"],
                activity_gt=activity_gt,
            )
            img = dset.index.file_name_to_img[fn]

            del dets["meta"]

            for class_, det in dets.items():
                for i in range(len(det)):
                    cat = dset.index.name_to_cat[class_]

                    xywh = (
                        kwimage.Boxes([det[i]["bbox"]], "tlbr")
                        .toformat("xywh")
                        .data[0]
                        .tolist()
                    )

                    ann = {
                        "area": xywh[2] * xywh[3],
                        "image_id": img["id"],
                        "category_id": cat["id"],
                        "segmentation": [],
                        "bbox": xywh,
                        "confidence": det[i]["confidence_score"],
                    }

                    print(det[i])
                    if "obj_obj_contact_state" in det[i].keys():
                        ann["obj-obj_contact_state"] = det[i]["obj_obj_contact_state"]
                        ann["obj-obj_contact_conf"] = det[i]["obj_obj_contact_conf"]
                    if "obj_hand_contact_state" in det[i].keys():
                        ann["obj-hand_contact_state"] = det[i]["obj_hand_contact_state"]
                        ann["obj-hand_contact_conf"] = det[i]["obj_hand_contact_conf"]

                    dset.add_annotation(**ann)

    dset.fpath = f"{save_dir}/{save_fn}" if save_dir != "" else save_fn
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved predictions to {dset.fpath}")

    return dset


def add_hl_hands_to_kwcoco(dset, remove_existing=True, using_contact=True):
    """Add bounding boxes for the hands based on the Hololen's joint positions

    This requires the projected hololens joint information generated by
    running ``scripts/hand_pose_converter.py``

    :param dset: kwcoco object or a string pointing to a kwcoco file
    :param using_contact: If True, adds contact state and contact conf
        for obj-obj and obj-hand states for each detection
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids

    for video_id in ub.ProgIter(
        dset.index.videos.keys(),
        desc=f"Adding hololens hands for videos in {dset.fpath}",
    ):
        image_ids = dset.index.vidid_to_gids[video_id]

        all_hand_pose_2d_image_space = None
        remove_aids = []

        for gid in sorted(image_ids):
            im = dset.imgs[gid]
            frame_idx, time = time_from_name(im["file_name"])

            aids = gid_to_aids[gid]
            anns = ub.dict_subset(dset.anns, aids)

            # Mark hand detections to be removed
            if remove_existing:
                for aid, ann in anns.items():
                    cat = dset.cats[ann["category_id"]]["name"]
                    if "hand" in cat:
                        remove_aids.append(aid)

            # Find HL hands, should only run once per video
            if not all_hand_pose_2d_image_space:
                # <video_folder>/_extracted/images/<file_name>
                extr_video_folder = im["file_name"].split("/")[:-2]
                extr_video_folder = ("/").join(extr_video_folder)
                all_hand_pose_2d_image_space = load_hl_hand_bboxes(extr_video_folder)

            all_hands = (
                all_hand_pose_2d_image_space[time]
                if time in all_hand_pose_2d_image_space.keys()
                else []
            )

            # Add HL hand bounding boxes if we have them
            if all_hands != []:
                # print("Adding hand bboxes from the hololens joints")
                for joints in all_hands:
                    keys = list(joints["joints"].keys())
                    hand_label = joints["hand"]

                    all_x_values = [joints["joints"][k]["projected"][0] for k in keys]
                    all_y_values = [joints["joints"][k]["projected"][1] for k in keys]

                    hand_bbox = [
                        min(all_x_values),
                        min(all_y_values),
                        max(all_x_values),
                        max(all_y_values),
                    ]  # tlbr
                    xywh = (
                        kwimage.Boxes([hand_bbox], "tlbr")
                        .toformat("xywh")
                        .data[0]
                        .tolist()
                    )

                    cat = dset.index.name_to_cat[hand_label]
                    ann = {
                        "area": xywh[2] * xywh[3],
                        "image_id": im["id"],
                        "category_id": cat["id"],
                        "segmentation": [],
                        "bbox": xywh,
                        "confidence": 1,
                    }

                    if using_contact:
                        ann["obj-obj_contact_state"] = False
                        ann["obj-obj_contact_conf"] = 0

                        ann["obj-hand_contact_state"] = False
                        ann["obj-hand_contact_conf"] = 0

                    dset.add_annotation(**ann)

        # print(f"Removing annotations {remove_aids} in video {video_id}")
        dset.remove_annotations(remove_aids)

    fpath = dset.fpath.split(".mscoco")[0]
    if remove_existing:
        dset.fpath = f"{fpath}_hl_hands_only.mscoco.json"
    else:
        dset.fpath = f"{fpath}_plus_hl_hands.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved predictions to {dset.fpath}")


def add_activity_gt_to_kwcoco(dset, activity_config_fn, activity_gt_dir):
    """Takes an existing kwcoco file and fills in the "activity_gt"
    field on each image based on the activity annotations.

    This saves to a new file (the original kwcoco file name with "_fixed"
    appended to the end).

    :param dset: kwcoco object or a string pointing to a kwcoco file
    :param activity_config_fn: Path to the activity labels config file
    :param activity_gt_dir: Path to the activity annotation csv files
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    # Load activity labels config
    with open(activity_config_fn, "r") as stream:
        activity_config = yaml.safe_load(stream)
    activity_labels = activity_config["labels"]

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    # Update the activity gt for each image
    for gid in ub.ProgIter(sorted(gids), desc="Adding activity ground truth to images"):
        im = dset.imgs[gid]

        video_id = im["video_id"]
        video_name = dset.index.videos[video_id]["name"]

        activity_gt_fn = f"{activity_gt_dir}/{video_name}.csv"
        gt = activities_from_dive_csv(activity_gt_fn)
        gt = activities_as_dataframe(gt)

        frame_idx, time = time_from_name(im["file_name"])
        matching_gt = gt.loc[(gt["start"] <= time) & (gt["end"] >= time)]

        if matching_gt.empty:
            label = "background"
            activity_label = label
        else:
            label = matching_gt.iloc[0]["class_label"]
            activity = [
                x
                for x in activity_labels[1:-1]
                if sanitize_str(x["full_str"]) == sanitize_str(label)
            ]
            if not activity:
                if "timer" in label:
                    # Ignoring timer based labels
                    label = "background"
                    activity_label = label
                else:
                    warnings.warn(
                        f"Label: {label} is not in the activity labels config, ignoring"
                    )
                    continue
            else:
                activity = activity[0]
                activity_label = activity["label"]

        dset.imgs[gid]["activity_gt"] = activity_label

    dset.fpath = dset.fpath.split(".")[0] + "_fixed.mscoco.json"
    dset.dump(dset.fpath, newlines=True)


def print_class_freq(dset):
    freq_per_class = dset.category_annotation_frequency()
    stats = []

    for cat in dset.cats.values():
        freq = freq_per_class[cat["name"]]
        class_ = {
            "id": cat["id"],
            "name": cat["name"],
            #'instances_count': freq,
            #'def': '',
            #'synonyms': [],
            #'image_count': freq,
            #'frequency': '',
            #'synset': ''
        }

        stats.append(class_)

    print(f"MC50_CATEGORIES = {stats}")


def class_freq_per_step(dset, activity_config_fn):
    """Calculate the number of objects detected in each activity"""
    dset = load_kwcoco(dset)

    # Load activity labels config
    with open(activity_config_fn, "r") as stream:
        activity_config = yaml.safe_load(stream)
    activity_labels = activity_config["labels"]

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    step_vals = np.zeros(len(dset.cats) + 2)

    act_labels = []
    obj_cats = []
    label_frame_freq = np.zeros(len(act_labels))

    freq_dict = {}
    cat_labels = np.zeros(len(dset.cats) + 2)

    cat_labels = [str(x) for x in cat_labels]
    for a in activity_labels:
        l = a["label"]
        if l not in ["done", "background"]:
            freq_dict[l] = np.zeros(len(dset.cats) + 2)

            act_labels.append(l)

    label_frame_freq = np.zeros(len(act_labels))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        act = im["activity_gt"]
        if act in ["done", "background"]:
            continue

        if act is None:
            continue

        act_id = act_labels.index(act)
        label_frame_freq[act_id] += 1

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        for aid, ann in anns.items():
            cat_id = ann["category_id"]
            cat = dset.cats[cat_id]["name"]

            conf = ann["confidence"]
            if "hand" in cat and conf == 1:
                # Hololens hand
                cat = cat + " (HoloLens)"
                if cat_id == 13:
                    cat_id = 43
                if cat_id == 14:
                    cat_id = 44

            cat_labels[cat_id - 1] = cat

            freq_dict[act][cat_id - 1] += 1

    print(freq_dict)
    print(f"Activity label freq: {label_frame_freq}")
    return freq_dict, act_labels, cat_labels, label_frame_freq


def intersect_per_step(dset, activity_config_fn):
    """Calculate the number of objects detected in each activity"""
    freq_dict = {}

    # Load kwcoco file
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    # Load activity labels config
    with open(activity_config_fn, "r") as stream:
        activity_config = yaml.safe_load(stream)
    activity_labels = activity_config["labels"]

    act_labels = []
    for a in activity_labels:
        l = a["label"]
        if l not in ["done", "background"]:
            freq_dict[l] = np.zeros(len(dset.cats) + 2)

            act_labels.append(l)

    # Load object labels
    cat_labels = []
    for c_id, c in dset.cats.items():
        cat_labels.append(c["name"])
    cat_labels.append("hand (left) (HoloLens)")
    cat_labels.append("hand (right) (HoloLens)")

    label_frame_freq = np.zeros(len(act_labels))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        act = im["activity_gt"]
        if act in ["done", "background"]:
            continue
        if act is None:
            continue

        act_id = act_labels.index(act)
        label_frame_freq[act_id] += 1

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        hand_id = 13

        hands = [ann for ann in anns.values() if ann["category_id"] == hand_id]
        hand = max(hands, key=lambda x: x["confidence"]) if len(hands) >= 1 else hands

        if not hand:
            continue

        hand_bbox = kwimage.Boxes([hand["bbox"]], "xywh")

        for aid, ann in anns.items():
            cat_id = ann["category_id"]
            if cat_id == hand_id:
                continue

            cat = dset.cats[cat_id]["name"]

            conf = ann["confidence"]
            """
            if "hand" in cat and conf == 1:
                # Hololens hand
                cat = cat + " (HoloLens)"
                if cat_id == 13:
                    cat_id = 43
                if cat_id == 14:
                    cat_id = 44
            """

            bbox = kwimage.Boxes(ann["bbox"], "xywh")

            iarea = hand_bbox.isect_area(bbox)
            hand_area = hand_bbox.area

            v = iarea / hand_area
            freq_dict[act][cat_id - 1] += v

    return freq_dict, act_labels, cat_labels, label_frame_freq


def plot_class_freq_per_step(freq_dict, act_labels, cat_labels, label_frame_freq):
    """Plot the number of objects detected, normalized by the number of frames
    per activity
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import normalize

    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 8

    matplotlib.rc("font", size=BIGGER_SIZE + 3)
    matplotlib.rc("xtick", labelsize=BIGGER_SIZE)
    matplotlib.rc("ytick", labelsize=BIGGER_SIZE)

    fig, ax = plt.subplots()

    mat = []
    for k, v in freq_dict.items():
        mat.append(v)

    mat = np.array(mat)

    norm_mat = mat / label_frame_freq[:, None]
    norm_mat = norm_mat.T

    plt.imshow(norm_mat)
    plt.colorbar()

    plt.xlabel("activities")
    plt.ylabel("object class")

    plt.xticks(range(len(act_labels)), act_labels, rotation="vertical")
    plt.yticks(range(len(cat_labels)), cat_labels)

    # plt.show()
    plt.savefig("obj_freq_per_act.png", bbox_inches="tight", dpi=300)


def visualize_kwcoco_by_contact(dset=None, save_dir=""):
    """Draw the bounding boxes from the kwcoco file on
    the associated images

    :param dset: kwcoco object or a string pointing to a kwcoco file
    :param save_dir: Directory to save the images to
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    red_patch = patches.Patch(color="r", label="obj")
    green_patch = patches.Patch(color="g", label="obj-obj contact")
    blue_patch = patches.Patch(color="b", label="obj-hand contact")

    empty_ims = 0
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]

        img_video_id = im["video_id"]
        # if img_video_id == 3:
        #    continue

        fn = im["file_name"].split("/")[-1]
        gt = im["activity_gt"]  # if hasattr(im, 'activity_gt') else ''
        if not gt:
            gt = ""

        fig, ax = plt.subplots()
        plt.title("\n".join(textwrap.wrap(gt, 55)))

        image = Image.open(im["file_name"])
        # image = image.resize(size=(760, 428), resample=Image.BILINEAR)
        image = np.array(image)

        ax.imshow(image)

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)
        using_contact = False
        for aid, ann in anns.items():
            conf = ann["confidence"]
            # if conf < 0.4:
            #    continue

            using_contact = True if "obj-obj_contact_state" in ann.keys() else False

            x, y, w, h = ann["bbox"]  # xywh
            cat = dset.cats[ann["category_id"]]["name"]
            if "tourniquet_tourniquet" in cat:
                tourniquet_im = image[int(y) : int(y + h), int(x) : int(x + w), ::-1]

                m2_fn = fn[:-4] + "_tourniquet_chip.png"
                m2_out = f"{save_dir}/video_{img_video_id}/images/chipped"
                Path(m2_out).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{m2_out}/{m2_fn}", tourniquet_im)

            label = f"{cat}: {round(conf, 2)}"

            color = "r"
            if using_contact and ann["obj-obj_contact_state"]:
                color = "g"
            if using_contact and ann["obj-hand_contact_state"]:
                color = "b"
            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
                clip_on=False,
            )

            ax.add_patch(rect)
            ax.annotate(label, (x, y), color="black", annotation_clip=False)

        if using_contact:
            plt.legend(handles=[red_patch, green_patch, blue_patch], loc="lower left")

        video_dir = f"{save_dir}/video_{img_video_id}/images/"
        Path(video_dir).mkdir(parents=True, exist_ok=True)

        plt.savefig(
            f"{video_dir}/{fn}",
        )
        plt.close(fig)  # needed to remove the plot because savefig doesn't clear it
    plt.close("all")


def visualize_kwcoco_by_label(dset=None, save_dir=""):
    """Draw the bounding boxes from the kwcoco file on
    the associated images

    :param dset: kwcoco object or a string pointing to a kwcoco file
    :param save_dir: Directory to save the images to
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.colors as mcolors

    # colors = list(mcolors.CSS4_COLORS.keys())
    # random.shuffle(colors)

    colors = [
        "yellow",
        "red",
        "turquoise",
        "beige",
        "dimgrey",
        "indigo",
        "springgreen",
        "green",
        "moccasin",
        "darkgoldenrod",
        "greenyellow",
        "violet",
        "cyan",
        "darkviolet",
        "darkturquoise",
        "skyblue",
        "navy",
        "azure",
        "lightcoral",
        "grey",
        "lemonchiffon",
        "gray",
        "deeppink",
        "wheat",
        "coral",
        "olivedrab",
        "lightgrey",
        "blue",
        "hotpink",
        "pink",
        "ghostwhite",
        "aquamarine",
        "orange",
        "deepskyblue",
        "darkorchid",
        "olive",
        "purple",
        "black",
        "limegreen",
        "darkslategray",
        "bisque",
        "steelblue",
    ]

    empty_ims = 0
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]

        img_video_id = im["video_id"]
        # if img_video_id == 3:
        #    continue

        fn = im["file_name"].split("/")[-1]
        gt = im["activity_gt"]  # if hasattr(im, 'activity_gt') else ''
        if not gt:
            gt = ""

        fig, ax = plt.subplots()
        plt.title("\n".join(textwrap.wrap(gt, 55)))

        image = Image.open(im["file_name"])
        # image = image.resize(size=(760, 428), resample=Image.BILINEAR)
        image = np.array(image)

        ax.imshow(image)

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)
        using_contact = False
        for aid, ann in anns.items():
            conf = ann["confidence"]
            # if conf < 0.4:
            #    continue

            x, y, w, h = ann["bbox"]  # xywh
            cat_id = ann["category_id"]
            cat = dset.cats[cat_id]["name"]

            label = f"{cat}: {round(conf, 2)}"

            color = colors[cat_id - 1]

            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
                clip_on=False,
            )

            ax.add_patch(rect)
            ax.annotate(label, (x, y), color="black", annotation_clip=False)

        video_dir = f"{save_dir}/video_{img_video_id}/images/"
        Path(video_dir).mkdir(parents=True, exist_ok=True)

        plt.savefig(
            f"{video_dir}/{fn}",
        )
        plt.close(fig)  # needed to remove the plot because savefig doesn't clear it
    plt.close("all")


def imgs_to_video(imgs_dir):
    """Convert directory of images to a video"""
    video_name = imgs_dir.split("/")[-1] + ".avi"

    images = glob.glob(f"{imgs_dir}/images/*.png")
    images = sorted(images, key=lambda x: time_from_name(x)[0])

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(f"{imgs_dir}/{video_name}", 0, 15, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()


def filter_kwcoco(dset):
    """Remove the inbetween classes from the kwcoco dataset

    :param dset: kwcoco object or a string pointing to a kwcoco file
    """
    experiment_name = "m2_all_data_cleaned_fixed_with_steps"
    stage = "stage2"

    print("Experiment: ", experiment_name)
    print("Stage: ", stage)

    dset = load_kwcoco(dset)

    # Remove in-between categories
    remove_cats = []
    for cat_id in dset.cats:
        cat_name = dset.cats[cat_id]["name"]
        if ".5)" in cat_name or "(before)" in cat_name or "(finished)" in cat_name:
            remove_cats.append(cat_id)

    print(f"removing cat ids: {remove_cats}")
    dset.remove_categories(remove_cats)

    # Remove images with these
    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    remove_images = []
    remove_anns = []
    for gid in sorted(gids):
        im = dset.imgs[gid]

        fn = im["file_name"].split("/")[-1]
        gt = im["activity_gt"]

        if gt == "not started" or "in between" in gt or gt == "finished":
            remove_images.append(gid)

        """
        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        for aid, ann in anns.items():
            conf = ann['confidence']
            if conf < 0.4:
                remove_anns.append(aid)
        """

    # print(f'removing {len(remove_anns)} annotations')
    # dset.remove_annotations(remove_anns)

    print(f"removing {len(remove_images)} images (and associated annotations)")
    dset.remove_images(remove_images)

    # Save to a new dataset to adjust ids
    new_dset = kwcoco.CocoDataset()
    new_cats = [
        {"id": 1, "name": "tourniquet_tourniquet (step 1)"},
        {"id": 2, "name": "tourniquet_tourniquet (step 2)"},
        {"id": 3, "name": "tourniquet_tourniquet (step 3)"},
        {"id": 4, "name": "tourniquet_tourniquet (step 4)"},
        {"id": 5, "name": "tourniquet_tourniquet (step 5)"},
        {"id": 6, "name": "tourniquet_tourniquet (step 6)"},
        {"id": 7, "name": "tourniquet_tourniquet (step 7)"},
        {"id": 8, "name": "tourniquet_tourniquet (step 8)"},
        {"id": 9, "name": "tourniquet_label (step 1)"},
        {"id": 10, "name": "tourniquet_label (step 2)"},
        {"id": 11, "name": "tourniquet_label (step 3)"},
        {"id": 12, "name": "tourniquet_label (step 4)"},
        {"id": 13, "name": "tourniquet_label (step 5)"},
        {"id": 14, "name": "tourniquet_label (step 6)"},
        {"id": 15, "name": "tourniquet_label (step 7)"},
        {"id": 16, "name": "tourniquet_label (step 8)"},
        {"id": 17, "name": "tourniquet_windlass (step 1)"},
        {"id": 18, "name": "tourniquet_windlass (step 2)"},
        {"id": 19, "name": "tourniquet_windlass (step 3)"},
        {"id": 20, "name": "tourniquet_windlass (step 4)"},
        {"id": 21, "name": "tourniquet_windlass (step 5)"},
        {"id": 22, "name": "tourniquet_windlass (step 6)"},
        {"id": 23, "name": "tourniquet_windlass (step 7)"},
        {"id": 24, "name": "tourniquet_windlass (step 8)"},
        {"id": 25, "name": "tourniquet_pen (step 1)"},
        {"id": 26, "name": "tourniquet_pen (step 2)"},
        {"id": 27, "name": "tourniquet_pen (step 3)"},
        {"id": 28, "name": "tourniquet_pen (step 4)"},
        {"id": 29, "name": "tourniquet_pen (step 5)"},
        {"id": 30, "name": "tourniquet_pen (step 6)"},
        {"id": 31, "name": "tourniquet_pen (step 7)"},
        {"id": 32, "name": "tourniquet_pen (step 8)"},
        {"id": 33, "name": "hand (step 1)"},
        {"id": 34, "name": "hand (step 2)"},
        {"id": 35, "name": "hand (step 3)"},
        {"id": 36, "name": "hand (step 4)"},
        {"id": 37, "name": "hand (step 5)"},
        {"id": 38, "name": "hand (step 6)"},
        {"id": 39, "name": "hand (step 7)"},
        {"id": 40, "name": "hand (step 8)"},
    ]
    for new_cat in new_cats:
        new_dset.add_category(name=new_cat["name"], id=new_cat["id"])

    for video_id, video in dset.index.videos.items():
        new_dset.add_video(**video)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        new_im = im.copy()

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        old_video = dset.index.videos[im["video_id"]]["name"]
        new_video = new_dset.index.name_to_video[old_video]

        del new_im["id"]
        new_im["video_id"] = new_video["id"]
        new_gid = new_dset.add_image(**new_im)

        for aid, ann in anns.items():
            old_cat = dset.cats[ann["category_id"]]["name"]
            new_cat = new_dset.index.name_to_cat[old_cat]

            new_ann = ann.copy()
            del new_ann["id"]
            new_ann["category_id"] = new_cat["id"]
            new_ann["image_id"] = new_gid

            new_dset.add_annotation(**new_ann)

    fpath = dset.fpath.split(".mscoco")[0]
    new_dset.fpath = f"{fpath}_filtered.mscoco.json"
    new_dset.dump(new_dset.fpath, newlines=True)
    print(f"Saved predictions to {new_dset.fpath}")


def filter_kwcoco_by_conf(dset, conf_thr=0.4):
    """Filter the kwcoco dataset by confidence

    :param dset: kwcoco object or a string pointing to a kwcoco file
    :param conf_thr: Minimum confidence to be left in the dataset
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    remove_anns = []
    for gid in sorted(gids):
        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        for aid, ann in anns.items():
            conf = ann["confidence"]

            if conf < conf_thr:
                remove_anns.append(aid)

    print(f"removing {len(remove_anns)} annotations")
    dset.remove_annotations(remove_anns)

    fpath = dset.fpath.split(".mscoco")[0]
    dset.fpath = f"{fpath}_conf_{conf_thr}.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def add_background_images(dset, background_imgs):
    """Add images without annotations to a kwcoco dataset"""
    # Load kwcoco file
    dset = load_kwcoco(dset)

    print(f"Images: {len(dset.imgs)}")

    for im in ub.ProgIter(glob.glob(f"{background_imgs}/*.png"), desc="Adding images"):
        image = Image.open(im)
        w, h = image.size

        new_im = {
            "width": w,
            "height": h,
            "file_name": im.split("2022-11-05_whole")[-1][1:],
        }
        new_gid = dset.add_image(**new_im)

    print(f"Images: {len(dset.imgs)}")
    fpath = dset.fpath.split(".json")[0]
    dset.fpath = f"{fpath}_plus_bkgd.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def main():
    ptg_root = "/home/local/KHQ/hannah.defazio/angel_system/angel_system/berkeley"
    ptg_root = "/data/PTG/cooking/"

    kw = "coffee_base_results_test_conf_0.1_plus_hl_hands.mscoco.json"

    n = kw[:-12].split("_")
    name = "_".join(n[:-1])
    split = n[-1]

    split = "test"

    stage = "results"
    stage_dir = f"{ptg_root}/annotations/coffee/{stage}"
    # stage_dir = ""
    exp = "coffee_base"
    save_dir = f"{stage_dir}/{exp}/visualization/conf_0.1_plus_hl_hands/{split}"

    # save_dir = "visualization"
    print(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if stage == "stage1":
        visualize_kwcoco_by_label(f"{stage_dir}/{kw}", save_dir)
    else:
        # visualize_kwcoco(f"{kw}", save_dir)
        visualize_kwcoco_by_label(f"{stage_dir}/{exp}/{kw}", save_dir)


if __name__ == "__main__":
    main()