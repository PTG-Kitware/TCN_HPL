import torch
from torch.utils.data import Dataset
import kwcoco
from tqdm import tqdm
import logging
import tcn_hpl.utils.utils as utils

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


def standarize_online_inputs(inputs):
    pass


def standarize_offline_inputs(data):
    dset = data[0]
    frames = data[1]


def collect_inputs(data, offline=True):
    """
    Collects inputs from either an offline or real-time source.

    Args:
        data: data stream input. For offline, a coco dataset. For online, a set of ROS messages

        offline (bool): If True, fetches data from offline source; otherwise, real-time.

    Returns:
        inputs: The collected inputs from either a real-time or offline source in standarized format.
            Designed as a dict of lists. all lists must have the size of the widnow_size:
                {object_dets: [], pose_estimations:[], left_hand: [], right_hand: []}

    """
    if offline:
        inputs = standarize_offline_inputs(data)
    else:
        inputs = standarize_online_inputs(data)

    return inputs


def define_tcn_vector(inputs):
    """
    Define the TCN vector using the collected inputs.

    Args:
        inputs: The inputs collected from either real-time or offline source.

    Returns:
        tcn_vector: The defined TCN vector.
    """
    tcn_vector = torch.tensor(inputs).float()  # Dummy transformation
    return tcn_vector


class TCNDataset(Dataset):

    def __init__(self, kwcoco_path: str, sample_rate: int, window_size: int):
        """
        Initializes the dataset.

        Args:
            kwcoco_path: The source of data (can be real-time or offline).
            sample_rate:
            window_size:
        """
        self.sample_rate = sample_rate
        self.window_size = window_size

        self.dset = kwcoco.CocoDataset(kwcoco_path)

        # for offline training, pre-cut videos into clips according to window size for easy batching
        self.frames = []

        logger.info(f"Generating dataset with {len(list(self.dset.index.videos.keys()))} videos")
        pber = tqdm(self.dset.index.videos.keys(), total=len(list(self.dset.index.videos.keys())))
        for vid in pber:
            video_dict = self.dset.index.videos[vid]

            vid_frames = self.dset.index.vidid_to_gids[vid]

            for index in range(0, len(vid_frames)-window_size-1, sample_rate):
                video_slice = vid_frames[index: index+window_size]
                window_frame_dicts = [self.dset.index.imgs[gid] for gid in video_slice]

                # start_frame = window_frame_dicts[0]['frame_index']
                # end_frame = window_frame_dicts[-1]['frame_index']

                # n_frames = end_frame - start_frame + 1

                self.frames.append(window_frame_dicts)

    def __getitem__(self, index):
        """
        Fetches the data point and defines its TCN vector.

        Args:
            index: The index of the data point.

        Returns:
            tcn_vector: The TCN vector for the given index.
        """

        data = self.frames[index]

        inputs = collect_inputs([self.dset, data], offline=True)

        # tcn_vector = define_tcn_vector(inputs)

        return

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            length: Length of the dataset.
        """
        return len(self.frames)


if __name__ == "__main__":
# Example usage:
    kwcoco_path = "/data/PTG/medical/training/yolo_object_detector/detect/r18_all/r18_all_all_obj_results_with_dets_and_pose.mscoco.json"

    dataset = TCNDataset(kwcoco_path=kwcoco_path,
                         sample_rate=1,
                         window_size=25)

    print(f"dataset: {len(dataset)}")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for index, batch in enumerate(data_loader):
        print(batch)  # This will print the TCN vectors for the batch
        if index > 15:
            exit()
