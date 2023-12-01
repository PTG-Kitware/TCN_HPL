import os
import cv2
import yaml
import glob
import kwcoco
import kwimage
import textwrap
import warnings
import random
import matplotlib
import shutil

import numpy as np
import ubelt as ub
import pandas as pd
import ubelt as ub
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

from sklearn.preprocessing import normalize

from pathlib import Path
from PIL import Image

from angel_system.data.common.load_data import (
    activities_from_dive_csv,
    objs_as_dataframe,
    time_from_name,
    sanitize_str,
)
from angel_system.data.common.load_data import Re_order
from angel_system.data.common.hl_hands_utils import load_hl_hand_bboxes


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


def replace_compound_label(
    preds,
    obj,
    detected_classes,
    using_contact,
    obj_hand_contact_state=False,
    obj_obj_contact_state=False,
):
    """
    Check if some subset of obj is in `detected_classes`

    Designed to catch and correct incorrect compound classes
    Ex: obj is "mug + filter cone + filter" but we detected "mug + filter cone"
    """

    compound_objs = [x.strip() for x in obj.split("+")]
    detected_objs = [[y.strip() for y in x.split("+")] for x in detected_classes]

    replaced_classes = []
    for detected_class, objs in zip(detected_classes, detected_objs):
        for obj_ in objs:
            if obj_ in compound_objs:
                replaced_classes.append(detected_class)
                break

    if replaced_classes == []:
        # Case 0, We didn't detect any of the objects
        replaced = None
    elif len(replaced_classes) == 1:
        # Case 1, we detected a subset of the compound as one detection
        replaced = replaced_classes[0]
        # print(f'replaced {replaced} with {obj}')

        preds[obj] = preds.pop(replaced)

        if using_contact:
            for i in range(len(preds[obj])):
                preds[obj][i]["obj_hand_contact_state"] = obj_hand_contact_state
                preds[obj][i]["obj_obj_contact_state"] = obj_obj_contact_state
    else:
        # Case 2, the compound was detected as separate boxes
        replaced = replaced_classes
        # print(f'Combining {replaced} detections into compound \"{obj}\"')

        new_bbox = None
        new_conf = None
        new_obj_obj_conf = None
        new_obj_hand_conf = None
        for det_obj in replaced:
            assert len(preds[det_obj]) == 1
            bbox = preds[det_obj][0]["bbox"]
            conf = preds[det_obj][0]["confidence_score"]

            if new_bbox is None:
                new_bbox = bbox
            else:
                # Find mix of bboxes
                # TODO: first double check these are close enough that it makes sense to combine?
                new_tlx, new_tly, new_brx, new_bry = new_bbox
                tlx, tly, brx, bry = bbox

                new_bbox = [
                    min(new_tlx, tlx),
                    min(new_tly, tly),
                    max(new_brx, brx),
                    max(new_bry, bry),
                ]

            new_conf = conf if new_conf is None else (new_conf + conf) / 2  # average???
            if using_contact:
                obj_obj_conf = preds[det_obj][0]["obj_obj_contact_conf"]
                new_obj_obj_conf = (
                    obj_obj_conf
                    if new_obj_obj_conf is None
                    else (new_obj_obj_conf + obj_obj_conf) / 2
                )
                obj_hand_conf = preds[det_obj][0]["obj_hand_contact_conf"]
                new_obj_hand_conf = (
                    obj_hand_conf
                    if new_obj_hand_conf is None
                    else (new_obj_hand_conf + obj_hand_conf) / 2
                )

            # remove old preds
            preds.pop(det_obj)

        new_pred = {
            "confidence_score": new_conf,
            "bbox": new_bbox,
        }
        if using_contact:
            new_pred["obj_obj_contact_state"] = obj_obj_contact_state
            new_pred["obj_obj_contact_conf"] = new_obj_obj_conf
            new_pred["obj_hand_contact_state"] = obj_hand_contact_state
            new_pred["obj_hand_contact_conf"] = new_obj_hand_conf

        preds[obj] = [new_pred]

    return preds, replaced


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


def add_activity_gt_to_kwcoco(dset):
    """Takes an existing kwcoco file and fills in the "activity_gt"
    field on each image based on the activity annotations.

    This saves to a new file (the original kwcoco file name with "_fixed"
    appended to the end).

    :param dset: kwcoco object or a string pointing to a kwcoco file
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    data_dir = "/data/PTG/cooking/"
    activity_gt_dir = f"{data_dir}/activity_anns"

    for video_id in dset.index.videos.keys():
        video = dset.index.videos[video_id]
        video_name = video["name"]
        print(video_name)

        if "_extracted" in video_name:
            video_name = video_name.split("_extracted")[0]
        video_recipe = video["recipe"]

        with open(
            f"../config/activity_labels/recipe_{video_recipe}.yaml", "r"
        ) as stream:
            recipe_activity_config = yaml.safe_load(stream)
        recipe_activity_labels = recipe_activity_config["labels"]

        recipe_activity_gt_dir = f"{activity_gt_dir}/{video_recipe}_labels/"
        if video_recipe == "coffee":
            activity_gt_fn = (
                f"{recipe_activity_gt_dir}/{video_name}_activity_labels_v_1.1.csv"
            )
            gt = activities_from_dive_csv(activity_gt_fn)
        else:
            activity_gt_fn = (
                f"{recipe_activity_gt_dir}/{video_name}_activity_labels_v_1.csv"
            )
            gt = activities_from_dive_csv(activity_gt_fn)
        gt = objs_as_dataframe(gt)

        image_ids = dset.index.vidid_to_gids[video_id]

        # Update the activity gt for each image
        for gid in sorted(image_ids):
            im = dset.imgs[gid]
            frame_idx, time = time_from_name(im["file_name"])

            matching_gt = gt.loc[(gt["start"] <= time) & (gt["end"] >= time)]

            if matching_gt.empty:
                label = "background"
                activity_label = label
            else:
                label = matching_gt.iloc[0]["class_label"]
                if type(label) == float or type(label) == int:
                    label = int(label)
                label = str(label)

                try:
                    activity = [
                        x
                        for x in recipe_activity_labels
                        if int(x["id"]) == int(float(label))
                    ]
                except:
                    activity = []

                if not activity:
                    if "timer" in label:
                        # Ignoring timer based labels
                        label = "background"
                        activity_label = label
                    else:
                        warnings.warn(
                            f"Label: {label} is not in the activity labels config, ignoring"
                        )
                        print(f"LABEL: {label}, {type(label)}")
                        continue
                else:
                    activity = activity[0]
                    activity_label = activity["label"]

                    # Temp fix until we can update the groundtruth labels
                    if activity_label in ["microwave-30-sec", "microwave-60-sec"]:
                        activity_label = "microwave"
                    if activity_label in ["stir-again"]:
                        activity_label = "oatmeal-stir"
                    if activity_label in [
                        "measure-half-cup-water",
                        "measure-12oz-water",
                    ]:
                        activity_label = "measure-water"
                    if activity_label in ["insert-toothpick-1", "insert-toothpick-2"]:
                        activity_label = "insert-toothpick"
                    if activity_label in ["slice-tortilla", "continue-slicing"]:
                        activity_label = "floss-slice-tortilla"
                    if activity_label in ["steep", "check-thermometer"]:
                        activity_label = "background"
                    if activity_label in ["dq-clean-knife", "pinwheel-clean-knife"]:
                        activity_label = "clean-knife"
                    if activity_label in ["zero-scale", "scale-turn-on"]:
                        activity_label = "scale-press-btn"
                    if activity_label in ["pour-water-grounds-wet"]:
                        activity_label = "pour-water-grounds-circular"

            dset.imgs[gid]["activity_gt"] = activity_label

    # dset.fpath = dset.fpath.split(".")[0] + "_fixed.mscoco.json"
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
                B
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

    dset = load_kwcoco(dset)

    colors = list(mcolors.CSS4_COLORS.keys())
    random.shuffle(colors)

    obj_labels = [v["name"] for k, v in dset.cats.items()]

    empty_ims = 0

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]

        img_video_id = im.get("video_id", None)

        fn = im["file_name"].split("/")[-1]
        gt = im.get("activity_gt", "")
        if not gt:
            gt = ""
        # act_pred = im.get("activity_pred", "")

        fig, ax = plt.subplots()
        # title = f"GT: {gt}, PRED: {act_pred}"
        plt.title("\n".join(textwrap.wrap(gt, 55)))

        image = Image.open(im["file_name"])
        # image = image.resize(size=(760, 428), resample=Image.BILINEAR)
        image = np.array(image)

        ax.imshow(image)

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)
        using_contact = False
        for aid, ann in anns.items():
            conf = ann.get("confidence", 1)
            # if conf < 0.1:
            #    continue

            x, y, w, h = ann["bbox"]  # xywh
            cat_id = ann["category_id"]
            cat = dset.cats[cat_id]["name"]

            label = f"{cat}: {round(conf, 2)}"

            color = colors[obj_labels.index(cat)]

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

        video_dir = (
            f"{save_dir}/video_{img_video_id}/images/"
            if img_video_id is not None
            else f"{save_dir}/images/"
        )
        Path(video_dir).mkdir(parents=True, exist_ok=True)

        plt.savefig(
            f"{video_dir}/{fn}",
        )
        plt.close(fig)  # needed to remove the plot because savefig doesn't clear it

    plt.close("all")


def squish_conf_vals(dset):
    # Load kwcoco file
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        for aid, ann in anns.items():
            conf = ann["confidence"]

            # v = ((conf - 0.95) / 0.05) * 0.9 + 0.1
            new_conf = 1 if conf >= 0.1 else 0  # max(v, 0.1)

            dset.anns[aid]["confidence"] = new_conf

    dset.fpath = (
        dset.fpath.split(".mscoco.json")[0] + "_0.1_conf_to_1.mscoco.json"
    )  # "_squished_conf.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def reorder_images(dset):
    # Data paths
    data_dir = "/data/PTG/cooking/"
    ros_bags_dir = f"{data_dir}/ros_bags/"
    coffee_ros_bags_dir = f"{ros_bags_dir}/coffee/coffee_extracted/"
    tea_ros_bags_dir = f"{ros_bags_dir}/tea/tea_extracted/"

    # Load kwcoco file
    dset = load_kwcoco(dset)
    gid_to_aids = dset.index.gid_to_aids

    new_dset = kwcoco.CocoDataset()

    new_dset.dataset["info"] = dset.dataset["info"].copy()
    for cat_id, cat in dset.cats.items():
        new_dset.add_category(**cat)

    for video_id, video in dset.index.videos.items():
        # Add video to new dataset
        if "_extracted" in video["name"]:
            video["name"] = video["name"].split("_extracted")[0]
        new_dset.add_video(**video)

        # Find folder of images for video
        video_name = video["name"]
        if "tea" in video_name:
            images_dir = f"{tea_ros_bags_dir}/{video_name}_extracted/images"
        else:
            images_dir = f"{coffee_ros_bags_dir}/{video_name}_extracted/images"

        images = glob.glob(f"{images_dir}/*.png")
        if not images:
            warnings.warn(f"No images found in {video_name}")
        images = Re_order(images, len(images))

        for image in images:
            # Find image in old dataset
            image_lookup = dset.index.file_name_to_img
            old_img = image_lookup[image]

            new_im = old_img.copy()
            del new_im["id"]

            new_gid = new_dset.add_image(**new_im)

            # Add annotations for image
            old_aids = gid_to_aids[old_img["id"]]
            old_anns = ub.dict_subset(dset.anns, old_aids)

            for old_aid, old_ann in old_anns.items():
                new_ann = old_ann.copy()

                del new_ann["id"]
                new_ann["image_id"] = new_gid
                new_dset.add_annotation(**new_ann)

    new_dset.fpath = dset.fpath.split(".mscoco.json")[0] + "_reordered_imgs.mscoco.json"
    new_dset.dump(new_dset.fpath, newlines=True)
    print(f"Saved dset to {new_dset.fpath}")


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


def background_images_dset(background_imgs):
    """Add images without annotations to a kwcoco dataset"""
    # Load kwcoco file
    import shutil

    dset = kwcoco.CocoDataset()

    for video in sorted(glob.glob(f"{background_imgs}/*/")):
        video_name = os.path.basename(os.path.normpath(video))
        video_lookup = dset.index.name_to_video
        if video_name in video_lookup:
            vid = video_lookup[video_name]["id"]
        else:
            vid = dset.add_video(name=video_name)

        for im in ub.ProgIter(glob.glob(f"{video}/*.png"), desc="Adding images"):
            new_dir = f"{video}/../"
            fn = os.path.basename(im)
            frame_num, time = time_from_name(fn)
            shutil.copy(im, new_dir)

            image = Image.open(im)
            w, h = image.size

            new_im = {
                "width": w,
                "height": h,
                "file_name": os.path.abspath(f"{new_dir}/{fn}"),
                "video_id": vid,
                "frame_index": frame_num,
            }
            new_gid = dset.add_image(**new_im)

    print(f"Images: {len(dset.imgs)}")
    dset.fpath = f"{background_imgs}/bkgd.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def draw_activity_preds(dset, save_dir="."):
    # Load kwcoco file
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        img_video_id = im["video_id"]

        fn = im["file_name"].split("/")[-1]

        gt = im["activity_gt"]
        if type(gt) is int:
            gt = dset.dataset["info"][0]["activity_labels"][str(gt)]
        pred = im["activity_pred"]
        if type(pred) is int:
            pred = dset.dataset["info"][0]["activity_labels"][str(pred)]

        fig, ax = plt.subplots()
        title = f"GT: {gt}\nPRED: {pred}"
        plt.title("\n".join(textwrap.wrap(title, 55)))

        image = Image.open(im["file_name"])
        image = np.array(image)

        ax.imshow(image)

        video_dir = f"{save_dir}/video_{img_video_id}/images/"
        Path(video_dir).mkdir(parents=True, exist_ok=True)

        plt.savefig(
            f"{video_dir}/{fn}",
        )
        plt.close(fig)  # needed to remove the plot because savefig doesn't clear it
    plt.close("all")


def dive_csv_to_kwcoco(dive_folder, object_config_fn, data_dir, dst_dir, output_dir=""):
    """Convert object annotations in DIVE csv file(s) to a kwcoco file

    :param dive_folder: Path to the csv files
    :param object_config_fn: Path to the object label config file
    """
    import shutil

    dset = kwcoco.CocoDataset()

    # Load object labels config
    with open(object_config_fn, "r") as stream:
        object_config = yaml.safe_load(stream)
    object_labels = object_config["labels"]

    label_ver = object_config["version"]
    title = object_config["title"].lower()
    dset.dataset["info"].append({f"{title}_object_label_version": label_ver})

    # Add categories
    for object_label in object_labels:
        if object_label["label"] == "background":
            continue
        dset.add_category(name=object_label["label"], id=object_label["id"])

    # Add boxes
    for csv_file in ub.ProgIter(
        glob.glob(f"{dive_folder}/*.csv"), desc="Loading video annotations"
    ):
        print(csv_file)
        video_name = os.path.basename(csv_file).split("_object_labels")[0]

        video_lookup = dset.index.name_to_video
        if video_name in video_lookup:
            vid = video_lookup[video_name]["id"]
        else:
            vid = dset.add_video(name=video_name)

        dive_df = pd.read_csv(csv_file)
        print(f"Loaded {csv_file}")
        for i, row in dive_df.iterrows():
            if i == 0:
                continue

            frame = row["2: Video or Image Identifier"]
            im_fp = f"{data_dir}/{video_name}_extracted/images/{frame}"

            frame_fn = f"{dst_dir}/{frame}"
            if not os.path.isfile(frame_fn):
                shutil.copy(im_fp, dst_dir)

            # Temp for coffee
            splits = frame.split("-")
            frame_num, time = time_from_name(splits[-1])

            image_lookup = dset.index.file_name_to_img
            if frame_fn in image_lookup:
                img_id = image_lookup[frame_fn]["id"]
            else:
                img_id = dset.add_image(
                    file_name=frame_fn,
                    video_id=vid,
                    frame_index=frame_num,
                    width=1280,
                    height=720,
                )

            bbox = (
                [
                    float(row["4-7: Img-bbox(TL_x"]),
                    float(row["TL_y"]),
                    float(row["BR_x"]),
                    float(row["BR_y)"]),
                ],
            )

            xywh = kwimage.Boxes([bbox], "tlbr").toformat("xywh").data[0][0].tolist()

            obj_id = row["10-11+: Repeated Species"]

            ann = {
                "area": xywh[2] * xywh[3],
                "image_id": img_id,
                "category_id": obj_id,
                "segmentation": [],
                "bbox": xywh,
            }

            dset.add_annotation(**ann)

    dset.fpath = f"{output_dir}/{title}_obj_annotations_v{label_ver}.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def mixed_dive_csv_to_kwcoco(
    dive_folder, object_config_fn, data_dir, dst_dir, output_dir=""
):
    """Convert object annotations in DIVE csv file(s) to a kwcoco file,
    for the use case where there is one DIVE file for multiple videos

    :param dive_folder: Path to the csv files
    :param object_config_fn: Path to the object label config file
    """
    dset = kwcoco.CocoDataset()

    # Load object labels config
    with open(object_config_fn, "r") as stream:
        object_config = yaml.safe_load(stream)
    object_labels = object_config["labels"]

    label_ver = object_config["version"]
    title = object_config["title"].lower()
    dset.dataset["info"].append({f"{title}_object_label_version": label_ver})

    # Add categories
    for object_label in object_labels:
        if object_label["label"] == "background":
            continue
        dset.add_category(name=object_label["label"], id=object_label["id"])

    # Add boxes
    for csv_file in ub.ProgIter(
        glob.glob(f"{dive_folder}/*.csv"), desc="Loading video annotations"
    ):
        print(csv_file)

        dive_df = pd.read_csv(csv_file)
        print(f"Loaded {csv_file}")
        for i, row in dive_df.iterrows():
            if i == 0:
                continue

            frame = row["2: Video or Image Identifier"]
            img_fn = os.path.basename(frame)
            frame_num, time = time_from_name(img_fn)
            frame_fn = f"{dst_dir}/images/{img_fn}"
            assert os.path.isfile(frame_fn)

            # Attempt to find the original file
            original_file = glob.glob(f"{data_dir}/*/*/*/images/{img_fn}")
            # import pdb; pdb.set_trace()
            assert len(original_file) == 1
            original_folder = os.path.dirname(original_file[0])
            video_name = original_folder.split("/")[-2]
            if "_extracted" in video_name:
                video_name = video_name.split("_extracted")[0]
            print(video_name)

            video_lookup = dset.index.name_to_video
            if video_name in video_lookup:
                vid = video_lookup[video_name]["id"]
            else:
                vid = dset.add_video(name=video_name)

            image_lookup = dset.index.file_name_to_img
            if frame_fn in image_lookup:
                img_id = image_lookup[frame_fn]["id"]
            else:
                img_id = dset.add_image(
                    file_name=frame_fn,
                    video_id=vid,
                    frame_index=frame_num,
                    width=1280,
                    height=720,
                )

            bbox = (
                [
                    float(row["4-7: Img-bbox(TL_x"]),
                    float(row["TL_y"]),
                    float(row["BR_x"]),
                    float(row["BR_y)"]),
                ],
            )

            xywh = kwimage.Boxes([bbox], "tlbr").toformat("xywh").data[0][0].tolist()

            obj_id = row["10-11+: Repeated Species"]

            ann = {
                "area": xywh[2] * xywh[3],
                "image_id": img_id,
                "category_id": obj_id,
                "segmentation": [],
                "bbox": xywh,
            }

            dset.add_annotation(**ann)

    dset.fpath = f"{title}_obj_annotations_v{label_ver}.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved dset to {dset.fpath}")


def update_obj_labels(dset, object_config_fn):
    """Change the object labels to match those provided
    in ``object_config_fn``
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    new_dset = kwcoco.CocoDataset()

    # Load object labels config
    with open(object_config_fn, "r") as stream:
        object_config = yaml.safe_load(stream)
    object_labels = object_config["labels"]

    label_ver = object_config["version"]
    new_dset.dataset["info"].append({"object_label_version": label_ver})

    # Add categories
    for object_label in object_labels:
        new_dset.add_category(name=object_label["label"], id=object_label["id"])

    for video_id, video in dset.index.videos.items():
        new_dset.add_video(**video)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        new_im = im.copy()

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)

        # Add video
        if hasattr(im, "video_id"):
            old_video = dset.index.videos[im["video_id"]]["name"]
            new_video = new_dset.index.name_to_video[old_video]
            new_im["video_id"] = new_video["id"]

        del new_im["id"]
        new_im["file_name"] = im["file_name"]
        new_gid = new_dset.add_image(**new_im)

        for aid, ann in anns.items():
            old_cat = dset.cats[ann["category_id"]]["name"]

            # Fix some deprecated labels
            if old_cat in ["timer", "timer (20)", "timer (30)", "timer (else)"]:
                old_cat = "timer (on)"
            if old_cat in ["kettle"]:
                old_cat = "kettle (closed)"
            if old_cat in ["water"]:
                old_cat = "water jug (open)"
            if old_cat in ["grinder (close)"]:
                old_cat = "grinder (closed)"
            if old_cat in ["thermometer (close)"]:
                old_cat = "thermometer (closed)"
            if old_cat in ["switch"]:
                old_cat = "kettle switch"
            if old_cat in ["lid (kettle)"]:
                old_cat = "kettle lid"
            if old_cat in ["lid (grinder)"]:
                old_cat = "grinder lid"
            if old_cat in ["coffee grounds + paper filter + filter cone"]:
                old_cat = "coffee grounds + paper filter (quarter - open) + dripper"
            if old_cat in ["coffee grounds + paper filter + filter cone + mug"]:
                old_cat = (
                    "coffee grounds + paper filter (quarter - open) + dripper + mug"
                )
            if old_cat in ["paper filter + filter cone"]:
                old_cat = "paper filter (quarter) + dripper"
            if old_cat in ["paper filter + filter cone + mug"]:
                old_cat = "paper filter (quarter) + dripper + mug"
            if old_cat in ["used paper filter + filter cone"]:
                old_cat = "used paper filter (quarter - open) + dripper"
            if old_cat in ["used paper filter + filter cone + mug"]:
                old_cat = "used paper filter (quarter) + dripper + mug"
            if old_cat in ["filter cone"]:
                old_cat = "dripper"
            if old_cat in ["filter cone + mug"]:
                old_cat = "dripper + mug"
            if old_cat in ["water + coffee grounds + paper filter + filter cone + mug"]:
                old_cat = "used paper filter (quarter - open) + dripper + mug"
            new_cat = new_dset.index.name_to_cat[old_cat]

            new_ann = ann.copy()

            del new_ann["id"]
            new_ann["category_id"] = new_cat["id"]
            new_ann["image_id"] = new_gid

            new_dset.add_annotation(**new_ann)

    fpath = dset.fpath.split(".mscoco")[0]
    new_dset.fpath = f"{fpath}_new_obj_labels.mscoco.json"
    new_dset.dump(new_dset.fpath, newlines=True)
    print(f"Saved predictions to {new_dset.fpath}")


def temp(dset):
    root_dir = "/data/PTG/cooking/ros_bags/tea/tea_extracted/"
    dst_dir = "/data/PTG/cooking/images/tea/berkeley/"

    # Load kwcoco file
    dset = load_kwcoco(dset)

    gid_to_aids = dset.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = dset.imgs[gid]
        video = dset.index.videos[im["video_id"]]["name"]
        fn = im["file_name"]

        im_fp = f"{root_dir}/{video}_extracted/images/{fn}"
        print(im_fp)
        assert os.path.isfile(im_fp)

        shutil.copy(im_fp, dst_dir)

        im = dset.imgs[gid]["file_name"] = f"{dst_dir}/{fn}"

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(dset.anns, aids)
        for aid, ann in anns.items():
            if hasattr(ann, "confidence"):
                del dset.anns[aid]["confidence"]

    dset.dump(f"{dset.fpath}_fixed_filepaths.mscoco.json", newlines=True)
    print(f"Saved predictions to {new_dset.fpath}")


def remap_category_ids_demo(dset):
    """Adjust the category ids in a kwcoco dataset
    (From Jon Crall)
    """
    # Load kwcoco file
    dset = load_kwcoco(dset)

    existing_cids = dset.categories().lookup("id")
    cid_mapping = {cid: cid - 1 for cid in existing_cids}

    for cat in dset.dataset["categories"]:
        old_cid = cat["id"]
        new_cid = cid_mapping[old_cid]
        cat["id"] = new_cid

    for ann in dset.dataset["annotations"]:
        old_cid = ann["category_id"]
        new_cid = cid_mapping[old_cid]
        ann["category_id"] = new_cid

    dset._build_index()
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved predictions to {dset.fpath}")


def dset_from_obj_dets(obj_dets, good_imgs_dir, output_dir):
    """Create a new kwcoco dataset and set of images
    based on a kwcoco file of object annotations and a
    folder of the desired final images
    """
    obj_dets = load_kwcoco(obj_dets)
    dset = kwcoco.CocoDataset()

    # Copy categories
    for i, cat in obj_dets.cats.items():
        dset.add_category(**cat)

    gid_to_aids = obj_dets.index.gid_to_aids
    gids = ub.argsort(ub.map_vals(len, gid_to_aids))

    for gid in sorted(gids):
        im = obj_dets.imgs[gid]

        video = obj_dets.index.videos[im["video_id"]]
        video_name = video["name"]
        if "_extracted" in video_name:
            video_name = video_name.split("_extracted")[0]

        fn = os.path.basename(im["file_name"])

        temp_fp = f"{good_imgs_dir}/{video_name}/{fn}"
        print("temp fp", temp_fp)
        if os.path.isfile(temp_fp):
            recipe_name = video["recipe"]
            if recipe_name == "dessertquesadilla":
                recipe_name = "dessert_quesadilla"
            if recipe_name == "pinwheel":
                recipe_name = "pinwheels"
            actual_fp = f"/data/PTG/cooking/ros_bags/{recipe_name}/{recipe_name}_extracted/{video_name}_extracted/images/{fn}"
            shutil.copy(actual_fp, f"{output_dir}/images")
        else:
            continue

        # Add video
        video_lookup = dset.index.name_to_video
        if video_name in video_lookup:
            vid = video_lookup[video_name]["id"]
        else:
            new_video = video.copy()
            del new_video["id"]
            new_video["name"] = video_name
            vid = dset.add_video(**new_video)

        # Add image
        new_fp = f"images/{fn}"
        print("new fp", new_fp)
        new_im = im.copy()
        del new_im["id"]
        new_im["file_name"] = new_fp
        new_im["video_id"] = vid

        img_id = dset.add_image(**new_im)

        aids = gid_to_aids[gid]
        anns = ub.dict_subset(obj_dets.anns, aids)
        for aid, ann in anns.items():
            # Add annotation
            new_ann = ann.copy()
            del new_ann["id"]
            del new_ann["image_id"]
            new_ann["image_id"] = img_id

            dset.add_annotation(**new_ann)

    dset.fpath = f"{output_dir}/hannah_additional_objs.mscoco.json"
    dset.dump(dset.fpath, newlines=True)
    print(f"Saved predictions to {dset.fpath}")
