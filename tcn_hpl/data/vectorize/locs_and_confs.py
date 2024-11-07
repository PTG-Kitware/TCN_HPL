from typing import List

import numpy as np
from numpy import typing as npt

from tcn_hpl.data.frame_data import FrameObjectDetections
from tcn_hpl.data.vectorize._interface import Vectorize, FrameData


NUM_POSE_JOINTS = 22


class LocsAndConfs(Vectorize):
    """
    Previous manual approach to vectorization.

    Arguments:
        top_k: The number of top per-class examples to use in vector
            construction.
        num_classes: the number of classes in the object detector.
        use_joint_confs: use the confidence of each pose joint.
            (changes the length of the input vector, which needs to
            be manually updated if this flag changes.)
        use_pixel_norm: Normalize pixel coordinates by dividing by
            frame height and width, respectively. Normalized values
            are between 0 and 1. Does not change input vector length.
        use_joint_obj_offsets: add abs(X and Y offsets) for between joints and
            each object.
            (changes the length of the input vector, which needs to
            be manually updated if this flag changes.)
    """

    def __init__(
        self,
        top_k: int = 1,
        num_classes: int = 7,
        use_joint_confs: bool = True,
        use_pixel_norm: bool = True,
        use_joint_obj_offsets: bool = False,
        background_idx: int = 0,
    ):
        super().__init__()

        self._top_k = top_k
        self._num_classes = num_classes
        self._use_joint_confs = use_joint_confs
        self._use_pixel_norm = use_pixel_norm
        self._use_joint_obj_offsets = use_joint_obj_offsets
        self._background_idx = background_idx

    @staticmethod
    def get_top_k_indexes_of_one_obj_type(
        f_dets: FrameObjectDetections,
        k: int,
        label_ind: int,
    ) -> List[int]:
        """
        Find all instances of a label index in object detections.
        Then sort them and return the top K.
        Inputs:
        - object_dets:
        """
        scores = f_dets.scores
        # Get all labels of an obj type
        filtered_idxs = [i for i, e in enumerate(f_dets.labels) if e == label_ind]
        # Sort filtered indices return by highest score
        filtered_scores = [scores[i] for i in filtered_idxs]
        return [
            i[1] for i in sorted(zip(filtered_scores, filtered_idxs), reverse=True)[:k]
        ]

    @staticmethod
    def append_vector(frame_feat, i, number):
        frame_feat[i] = number
        return frame_feat, i + 1

    def determine_vector_length(self) -> int:
        #########################
        # Feature vector
        #########################
        # Length: pose confs * 22, pose X's * 22, pose Y's * 22,
        #         obj confs * num_objects(7 for M2),
        #         obj X * num_objects(7 for M2),
        #         obj Y * num_objects(7 for M2)
        #         obj W * num_objects(7 for M2)
        #         obj H * num_objects(7 for M2)
        #         casualty conf * 1
        vector_length = 0
        # [Conf, X, Y, W, H] for k instances of each object class.
        vector_length += 5 * self._top_k * self._num_classes
        # Pose confidence score
        vector_length += 1
        # Joint confidences
        if self._use_joint_confs:
            vector_length += NUM_POSE_JOINTS
        # X and Y for each joint
        vector_length += 2 * NUM_POSE_JOINTS
        return vector_length

    def vectorize(self, data: FrameData) -> npt.NDArray[np.float32]:
        # I tried utilizing range assignment into frame_feat, but this was
        # empirically not as fast as this method in the context of being run
        # within a torch DataLoader.
        # E.g. instead of
        #       for i, det_idx in enumerate(top_det_idxs):
        #           topk_offset = obj_offset + (i * 5)
        #           frame_feat[topk_offset + 0] = f_dets.scores[det_idx]
        #           frame_feat[topk_offset + 1] = f_dets.boxes[det_idx][0] / w
        #           frame_feat[topk_offset + 2] = f_dets.boxes[det_idx][1] / h
        #           frame_feat[topk_offset + 3] = f_dets.boxes[det_idx][2] / w
        #           frame_feat[topk_offset + 4] = f_dets.boxes[det_idx][3] / h
        # doing:
        #       obj_end_idx = obj_offset + (len(top_det_idxs) * 5)
        #       frame_feat[obj_offset + 0:obj_end_idx:5] = f_dets.scores[top_det_idxs]
        #       frame_feat[obj_offset + 1:obj_end_idx:5] = f_dets.boxes[top_det_idxs, 0] / w
        #       frame_feat[obj_offset + 2:obj_end_idx:5] = f_dets.boxes[top_det_idxs, 1] / h
        #       frame_feat[obj_offset + 3:obj_end_idx:5] = f_dets.boxes[top_det_idxs, 2] / w
        #       frame_feat[obj_offset + 4:obj_end_idx:5] = f_dets.boxes[top_det_idxs, 3] / h
        # Was *slower* in the context of batched computation.

        vector_len = self.determine_vector_length()
        frame_feat = np.zeros(vector_len, dtype=np.float32)

        if self._use_pixel_norm:
            w = data.size[0]
            h = data.size[1]
        else:
            w = 1
            h = 1

        obj_num_classes = self._num_classes
        obj_top_k = self._top_k

        # Indices into the feature vector where components start
        objs_start_offset = 0
        pose_start_offset = obj_num_classes * obj_top_k * 5

        f_dets = data.object_detections
        if f_dets:
            for obj_ind in range(obj_num_classes):
                obj_offset = objs_start_offset + (obj_ind * obj_top_k * 5)
                top_det_idxs = self.get_top_k_indexes_of_one_obj_type(
                    f_dets, obj_top_k, obj_ind
                )
                for i, det_idx in enumerate(top_det_idxs):
                    topk_offset = obj_offset + (i * 5)
                    frame_feat[topk_offset + 0] = f_dets.scores[det_idx]
                    frame_feat[topk_offset + 1] = f_dets.boxes[det_idx][0] / w
                    frame_feat[topk_offset + 2] = f_dets.boxes[det_idx][1] / h
                    frame_feat[topk_offset + 3] = f_dets.boxes[det_idx][2] / w
                    frame_feat[topk_offset + 4] = f_dets.boxes[det_idx][3] / h
                # If there are less than top_k indices returned, the vector was
                # already initialized to zero so nothing else to do.

        f_poses = data.poses
        if f_poses:
            # Find most confident body detection
            confident_pose_idx = np.argmax(f_poses.scores)
            frame_feat[pose_start_offset] = f_poses.scores[confident_pose_idx]
            pose_kp_offset = pose_start_offset + 1
            for joint_ind in range(NUM_POSE_JOINTS):
                joint_offset = pose_kp_offset + (joint_ind * 3)
                if self._use_joint_confs:
                    frame_feat[joint_offset] = f_poses.joint_scores[
                        confident_pose_idx, joint_ind
                    ]
                frame_feat[joint_offset + 1] = (
                    f_poses.joint_positions[confident_pose_idx, joint_ind, 0] / w
                )
                frame_feat[joint_offset + 2] = (
                    f_poses.joint_positions[confident_pose_idx, joint_ind, 1] / h
                )

        return frame_feat
