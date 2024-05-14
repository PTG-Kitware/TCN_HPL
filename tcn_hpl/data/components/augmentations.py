import random
import torch

import numpy as np


# from angel_system.activity_classification.utils import feature_version_to_options

class MoveCenterPts(torch.nn.Module):
    """Simulate moving the center points of the bounding boxes by
    adjusting the distances for each frame
    """

    def __init__(
        self,
        hand_dist_delta,
        obj_dist_delta,
        window_size,
        im_w,
        im_h,
        num_obj_classes,
        feat_version,
    ):
        """
        :param hand_dist_delta: Decimal percentage to calculate the +-offset in
            pixels for the hands
        :param obj_dist_delta: Decimal percentage to calculate the +-offset in
            pixels for the objects
        :param w: Width of the frames
        :param h: Height of the frames
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.hand_dist_delta = hand_dist_delta
        self.window_size = window_size
        self.obj_dist_delta = obj_dist_delta

        self.im_w = im_w
        self.im_h = im_h
        self.num_obj_classes = num_obj_classes

        self.feat_version = feat_version

        # Deltas
        self.hand_delta_x = self.im_w * self.hand_dist_delta
        self.hand_delta_y = self.im_h * self.hand_dist_delta

        self.obj_ddelta_x = self.im_w * self.obj_dist_delta
        self.obj_ddelta_y = self.im_h * self.obj_dist_delta

    def init_deltas(self):
        rhand_delta_x = random.uniform(-self.hand_delta_x, self.hand_delta_x)
        rhand_delta_y = random.uniform(-self.hand_delta_y, self.hand_delta_y)

        lhand_delta_x = random.uniform(-self.hand_delta_x, self.hand_delta_x)
        lhand_delta_y = random.uniform(-self.hand_delta_y, self.hand_delta_y)

        obj_delta_x = random.uniform(-self.obj_ddelta_x, self.obj_ddelta_x)
        obj_delta_y = random.uniform(-self.obj_ddelta_y, self.obj_ddelta_y)

        return (
            [rhand_delta_x, rhand_delta_y],
            [lhand_delta_x, lhand_delta_y],
            [obj_delta_x, obj_delta_y],
        )

    def forward(self, features):
        for i in range(features.shape[0]):
            frame = features[i]

            (
                [rhand_delta_x, rhand_delta_y],
                [lhand_delta_x, lhand_delta_y],
                [obj_delta_x, obj_delta_y],
            ) = self.init_deltas()

            if self.feat_version == 1:
                # No distances to move
                pass

            elif self.feat_version == 2 or self.feat_version == 5:
                num_obj_feats = self.num_obj_classes - 2  # not including hands in count
                num_obj_points = num_obj_feats * 2

                # Distance from hand to object
                right_dist_idx2 = num_obj_points + 1
                left_dist_idx1 = num_obj_points + 2
                left_dist_idx2 = left_dist_idx1 + num_obj_points

                for hand_delta_x, hand_delta_y, start_idx, end_idx in zip(
                    [rhand_delta_x, lhand_delta_x],
                    [rhand_delta_y, lhand_delta_y],
                    [1, left_dist_idx1],
                    [right_dist_idx2, left_dist_idx2],
                ):
                    frame[start_idx:end_idx:2] = np.where(
                        frame[start_idx:end_idx:2] != 0,
                        frame[start_idx:end_idx:2] + hand_delta_x + obj_delta_x,
                        frame[start_idx:end_idx:2],
                    )

                    frame[start_idx + 1 : end_idx : 2] = np.where(
                        frame[start_idx + 1 : end_idx : 2] != 0,
                        frame[start_idx + 1 : end_idx : 2] + hand_delta_y + obj_delta_y,
                        frame[start_idx + 1 : end_idx : 2],
                    )

                # Distance between hands
                hands_dist_idx = left_dist_idx2

                frame[hands_dist_idx] = np.where(
                    frame[hands_dist_idx] != 0,
                    frame[hands_dist_idx] + rhand_delta_x + lhand_delta_x,
                    frame[hands_dist_idx],
                )

                frame[hands_dist_idx + 1] = np.where(
                    frame[hands_dist_idx + 1] != 0,
                    frame[hands_dist_idx + 1] + rhand_delta_y + lhand_delta_y,
                    frame[hands_dist_idx + 1],
                )

            elif self.feat_version == 3:
                # Right and left hand distances
                right_idx1 = 1
                right_idx2 = 2
                left_idx1 = 4
                left_idx2 = 5
                for hand_delta_x, hand_delta_y, start_idx, end_idx in zip(
                    [rhand_delta_x, lhand_delta_x],
                    [rhand_delta_y, lhand_delta_y],
                    [right_idx1, left_idx1],
                    [right_idx2, left_idx2],
                ):
                    frame[start_idx] = frame[start_idx] + hand_delta_x

                    frame[end_idx] = frame[end_idx] + hand_delta_y

                # Object distances
                start_idx = 10
                while start_idx < len(frame):
                    frame[start_idx] = frame[start_idx] + obj_delta_x
                    frame[start_idx + 1] = frame[start_idx + 1] + obj_delta_y
                    start_idx += 5

            else:
                NotImplementedError(f"Unhandled version '{self.feat_version}'")

            features[i] = frame
        return features

    def __repr__(self) -> str:
        detail = f"(hand_dist_delta={self.hand_dist_delta}, obj_dist_delta={self.obj_dist_delta}, im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"


class ActivationDelta(torch.nn.Module):
    """Update the activation feature of each class by +-``conf_delta``"""

    def __init__(self, conf_delta, num_obj_classes, feat_version):
        """
        :param conf delta:
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.conf_delta = conf_delta

        self.num_obj_classes = num_obj_classes

        self.feat_version = feat_version

    def init_delta(self):
        delta = random.uniform(-self.conf_delta, self.conf_delta)

        return delta

    def forward(self, features):
        delta = self.init_delta()

        if self.feat_version == 1:
            activation_idxs = range(features.shape[1])

        elif self.feat_version == 2:
            num_obj_feats = self.num_obj_classes - 2  # not including hands in count
            num_obj_points = num_obj_feats * 2

            obj_acts_idx = num_obj_points + 1 + num_obj_points + 2 + 1
            activation_idxs = [0, num_obj_points + 1] + list(
                range(obj_acts_idx, features.shape[1])
            )

        elif self.feat_version == 3:
            activation_idxs = [0, 3] + list(range(7, features.shape[1], 5))

        elif self.feat_version == 5:
            num_obj_feats = self.num_obj_classes - 2  # not including hands in count
            num_obj_points = num_obj_feats * 2

            obj_acts_idx = num_obj_points + 1 + num_obj_points + 2 + 1 + 1
            activation_idxs = [0, num_obj_points + 1] + list(
                range(obj_acts_idx, features.shape[1], 3)
            )

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        features[:, activation_idxs] = np.where(
            features[:, activation_idxs] != 0,
            np.clip(features[:, activation_idxs] + delta, 0, 1),
            features[:, activation_idxs],
        )

        return features

    def __repr__(self) -> str:
        detail = f"(conf_delta={self.conf_delta}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"


class NormalizePixelPts(torch.nn.Module):
    """Normalize the distances from -1 to 1 with respect to the image size"""

    def __init__(self, im_w, im_h, num_obj_classes, feat_version, top_k_objects):
        """
        :param w: Width of the frames
        :param h: Height of the frames
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.im_w = im_w
        self.im_h = im_h

        self.num_obj_classes = num_obj_classes
        self.num_non_obj_classes = 2 # hands
        self.num_good_obj_classes = self.num_obj_classes - self.num_non_obj_classes

        self.top_k_objects = top_k_objects

        self.feat_version = feat_version
        
        # self.opts = feature_version_to_options(self.feat_version)
        # print(self.opts)

        # self.use_activation = self.opts.get("use_activation", False)
        # print("use_activation", self.use_activation)
        # self.use_hand_dist = self.opts.get("use_hand_dist", False)
        # self.use_intersection = self.opts.get("use_intersection", False)
        # self.use_center_dist = self.opts.get("use_center_dist", False)
        # self.use_joint_hand_offset = self.opts.get("use_joint_hand_offset", False)
        # self.use_joint_object_offset = self.opts.get("use_joint_object_offset", False)
    
    def forward(self, features):
        for i in range(features.shape[0]):
            frame = features[i]
        
            # HANDS
            if self.use_activation:
                ind = 0 # right hand confidence
                ind +=1 # left hand confidence
            # Right - left hand
            if self.use_hand_dist:
                # Right - left hand distance
                ind += 1 # Right - left hand distance x
                rh_lh_dist_x = frame[ind]
                frame[ind] = rh_lh_dist_x / self.im_w

                ind += 1 # Right - left hand distance y
                rh_lh_dist_y = frame[ind]
                frame[ind] = rh_lh_dist_y / self.im_h
            if self.use_intersection:
                ind += 1 # left / right hand intersection
            if self.use_center_dist:
                ind += 2 # right hand - image center x/y
                ind +=2 # left hand - image center x/y
            if self.use_joint_hand_offset:
                # right hand - joints distances
                for i in range(22):
                    ind += 1
                    rh_jointi_dist_x = frame[ind]
                    frame[ind] = rh_jointi_dist_x / self.im_w

                    ind += 1
                    rh_jointi_dist_y = frame[ind]
                    frame[ind] = rh_jointi_dist_y / self.im_h
            
                # left hand - joints distances
                for i in range(22):
                    ind += 1
                    lh_jointi_dist_x = frame[ind]
                    frame[ind] = lh_jointi_dist_x / self.im_w

                    ind += 1
                    lh_jointi_dist_y = frame[ind]
                    frame[ind] = lh_jointi_dist_y / self.im_h

            # TOP K OBJECTS
            for object_k_index in range(self.top_k_objects):
                if self.use_hand_dist:
                    # Right hand distances
                    for obj_ind in range(self.num_good_obj_classes):
                        ind += 1
                        obj_rh_dist_x = frame[ind]
                        frame[ind] = obj_rh_dist_x / self.im_w

                        ind += 1
                        obj_rh_dist_y = frame[ind]
                        frame[ind] = obj_rh_dist_y / self.im_h

                    # Left hand distances
                    for obj_ind in range(self.num_good_obj_classes):
                        ind += 1
                        obj_lh_dist_x = frame[ind]
                        frame[ind] = obj_lh_dist_x / self.im_w

                        ind += 1
                        obj_lh_dist_y = frame[ind]
                        frame[ind] = obj_lh_dist_y / self.im_h

                for obj_ind in range(self.num_good_obj_classes):
                    if self.use_activation:
                        ind += 1 # Object confidence

                    if self.use_intersection:
                        ind += 1 # obj - right hand intersection
                        ind += 1 # obj - left hand intersection

                    if self.use_center_dist:
                        ind += 2 # image center - obj distances x/y


                if self.use_joint_object_offset:
                    # obj - joints distances
                    for obj_ind in range(self.num_good_obj_classes):
                        joints_dists = []
                        for i in range(22):
                            ind += 1
                            obj_jointi_dist_x = frame[ind]
                            frame[ind] = obj_jointi_dist_x / self.im_w

                            ind += 1
                            obj_jointi_dist_y = frame[ind]
                            frame[ind] = obj_jointi_dist_y / self.im_h
            features[i] = frame
        return features

    def __repr__(self) -> str:
        detail = f"(im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"


class NormalizeFromCenter(torch.nn.Module):
    """Normalize the distances from -1 to 1 with respect to the image center

    Missing objects will be set to (2, 2)
    """

    def __init__(self, im_w, im_h, feat_version):
        """
        :param w: Width of the frames
        :param h: Height of the frames
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.im_w = im_w
        self.half_w = im_w / 2
        self.im_h = im_h
        self.half_h = im_h / 2

        self.feat_version = feat_version

    def forward(self, features):
        if self.feat_version == 1:
            # No distances to normalize
            pass

        elif self.feat_version == 2 or self.feat_version == 5:
            # Distances are relative to the image size, skip
            pass

        elif self.feat_version == 3:
            # Right and left hand distances
            right_idx1 = 1
            right_idx2 = 2
            left_idx1 = 4
            left_idx2 = 5
            for start_idx, end_idx in zip(
                [right_idx1, left_idx1],
                [right_idx2, left_idx2],
            ):

                features[:, start_idx] = features[:, start_idx] / self.half_w
                features[:, end_idx] = features[:, end_idx] / self.half_h

            # Object distances
            start_idx = 10
            while start_idx < features.shape[1]:
                features[:, start_idx] = features[:, start_idx] / self.half_w
                features[:, start_idx + 1] = features[:, start_idx + 1] / self.half_h
                start_idx += 5

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return features

    def __repr__(self) -> str:
        detail = (
            f"(im_w={self.im_w}, im_h={self.im_h}, feat_version={self.feat_version})"
        )
        return f"{self.__class__.__name__}{detail}"
