# coding=utf-8
"""
The data is already preprocessed into frames. 

At this point this script is not needed.
"""


import os
import cv2
from glob import glob
import json
import utils


def main():
    # the dir including videos you want to process
    videos_src_path = "/data/datasets/ptg/m2_tourniquet/"
    # the save path
    videos_save_path = "/data/datasets/ptg/m2_tourniquet/imgs"

    videos = utils.dictionary_contents(
        videos_src_path, types=["*.mp4", "*.MP4"], recursive=True
    )

    coco_json = {
        "info": {
            "description": "Medical Pose Estimation",
            "year": 2023,
            "contributer": "Kitware",
            "version": "0.1",
        },
        "images": [],
        "annotations": [],
    }

    for index, each_video in enumerate(videos):
        # get the name of each video, and make the directory to save frames
        video_name = each_video.split("/")[-1].split(".")[0]
        print("Video Name :", video_name)
        # each_video_name, _ = each_video.split('.')each_video.split('.')
        dir_path = f"{videos_save_path}/{video_name}"
        utils.create_dir_if_doesnt_exist(dir_path)
        # each_video_save_full_path = os.path.join(videos_save_path, each_video_name) + '/'

        # get the full path of each video, which will open the video tp extract frames
        # each_video_full_path = os.path.join(videos_src_path, each_video)

        cap = cv2.VideoCapture(each_video)

        frame_count = 1

        frame_rate = 1
        success = True
        # 计数
        num = 0
        while success:
            success, frame = cap.read()
            if success == True:
                # if not os.path.exists(each_video_save_full_path + video_name):
                #     os.makedirs(each_video_save_full_path + video_name)
                height, width, channels = frame.shape
                # print(frame.shape)
                file_name = f"{video_name}_{frame_count}.jpg"
                image_dict = {
                    "id": f"{index}{frame_count}",
                    "file_name": file_name,
                    "video_name": f"{video_name}",
                    "video_id": index,
                    "width": width,
                    "height": height,
                }
                coco_json["images"].append(image_dict)
                if frame_count % frame_rate == 0:
                    # cv2.imwrite(each_video_save_full_path + video_name + '/'+ "%06d.jpg" % num, frame)
                    cv2.imwrite(f"{dir_path}/{file_name}", frame)
                    image_dict["path"] = f"{dir_path}/{file_name}"
                    num += 1

            frame_count = frame_count + 1
        print("Final frame:", num)

    coco_json_save_path = "medical_coco.json"

    with open(coco_json_save_path, "w") as outfile:
        json.dump(coco_json, outfile)


if __name__ == "__main__":
    main()
