import random
import torch
import yaml

import numpy as np

from angel_system.berkeley.data.objects.coffee_activity_objects import original_sub_steps


class MoveCenterPts(torch.nn.Module):
    """Simulate moving the center points of the bounding boxes by
    adjusting the distances
    """

    def __init__(
        self, hand_dist_delta, obj_dist_delta, im_w, im_h, num_obj_classes, feat_version
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

    def forward(self, window):
        (
            [rhand_delta_x, rhand_delta_y],
            [lhand_delta_x, lhand_delta_y],
            [obj_delta_x, obj_delta_y],
        ) = self.init_deltas()

        if self.feat_version == 1:
            pass

        elif self.feat_version == 2:
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
                window[:, start_idx:end_idx:2] = np.where(
                    window[:, start_idx:end_idx:2] != 0,
                    window[:, start_idx:end_idx:2] + hand_delta_x + obj_delta_x,
                    window[:, start_idx:end_idx:2],
                )

                window[:, start_idx + 1 : end_idx : 2] = np.where(
                    window[:, start_idx + 1 : end_idx : 2] != 0,
                    window[:, start_idx + 1 : end_idx : 2] + hand_delta_y + obj_delta_y,
                    window[:, start_idx + 1 : end_idx : 2],
                )

            # Distance between hands
            hands_dist_idx = left_dist_idx2 + 1

            window[:, hands_dist_idx] = np.where(
                window[:, hands_dist_idx] != 0,
                window[:, hands_dist_idx] + rhand_delta_x + lhand_delta_x,
                window[:, hands_dist_idx],
            )

            window[:, hands_dist_idx + 1] = np.where(
                window[:, hands_dist_idx + 1] != 0,
                window[:, hands_dist_idx + 1] + rhand_delta_y + lhand_delta_y,
                window[:, hands_dist_idx + 1],
            )

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return window

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

    def forward(self, window):
        delta = self.init_delta()

        if self.feat_version == 1:
            window[:] = np.where(
                window[:] != 0, np.clip(window[:] + delta, 0, 1), window[:]
            )

        elif self.feat_version == 2:
            num_obj_feats = self.num_obj_classes - 2  # not including hands in count
            num_obj_points = num_obj_feats * 2

            obj_acts_idx = num_obj_points + 1 + num_obj_points + 2 + 1
            activation_idxs = [0, num_obj_points + 1] + list(
                range(obj_acts_idx, len(window))
            )

            window[:, activation_idxs] = np.where(
                window[:, activation_idxs] != 0,
                np.clip(window[:, activation_idxs] + delta, 0, 1),
                window[:, activation_idxs],
            )

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return window

    def __repr__(self) -> str:
        detail = f"(conf_delta={self.conf_delta}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"


class NormalizePixelPts(torch.nn.Module):
    """Normalize the distances from -1 to 1 with respect to the image size"""

    def __init__(self, im_w, im_h, num_obj_classes, feat_version):
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

        self.feat_version = feat_version

    def forward(self, window):
        if self.feat_version == 1:
            pass

        elif self.feat_version == 2:
            num_obj_feats = self.num_obj_classes - 2  # not including hands in count
            num_obj_points = num_obj_feats * 2

            # Distance from hand to object
            right_dist_idx2 = num_obj_points + 1
            left_dist_idx1 = num_obj_points + 2
            left_dist_idx2 = left_dist_idx1 + num_obj_points

            for start_idx, end_idx in zip(
                [1, left_dist_idx1], [right_dist_idx2, left_dist_idx2]
            ):
                window[:, start_idx:end_idx:2] = (
                    window[:, start_idx:end_idx:2] / self.im_w
                )
                window[:, start_idx + 1 : end_idx : 2] = (
                    window[:, start_idx + 1 : end_idx : 2] / self.im_h
                )

            # Distance between hands
            hands_dist_idx = left_dist_idx2 + 1

            window[:, hands_dist_idx] = window[:, hands_dist_idx] / self.im_w
            window[:, hands_dist_idx + 1] = window[:, hands_dist_idx + 1] / self.im_h

        else:
            NotImplementedError(f"Unhandled version '{self.feat_version}'")

        return window

    def __repr__(self) -> str:
        detail = f"(im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version})"
        return f"{self.__class__.__name__}{detail}"

class IrrelevantClassDropout(torch.nn.Module):
    def __init__(self, dropout_rate, actions_dict, obj_config_fn, num_obj_classes, feat_version=2):
        """
        :param dropout_rate: Percentage (in decimal) of irrelevant objects to remove
            from the data
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        """
        super().__init__()

        self.dropout_rate = dropout_rate
        self.actions_dict = dict(zip(actions_dict.values(), actions_dict.keys()))

        self.num_obj_classes = num_obj_classes
        self.feat_version = feat_version

        with open(obj_config_fn, "r") as stream:
            obj_config = yaml.safe_load(stream)
        objs = obj_config["labels"]
        self.obj_labels = [o["label"] for o in objs]
        self.obj_ids = [o["id"] for o in objs]
        print("obj ids", self.obj_ids)

        self.obj_dict = dict(zip(self.obj_labels, self.obj_ids))

        self.act_to_objs = {}
        for step, substep in original_sub_steps.items():
            for sub_step in substep:
                act = sub_step[0]
                objs = sub_step[1]

                self.act_to_objs[act] = objs
        self.act_to_objs["background"] = [["hand"]]

    def forward(self, window, targets):
        class_ids = self.obj_ids
        num_obj_feats = self.num_obj_classes - 2  # not including hands in count
        num_obj_points = num_obj_feats * 2

        # Distance from hand to object
        right_dist_idx1 = 1
        right_dist_idx2 = num_obj_points + 1
        left_dist_idx1 = num_obj_points + 2
        left_dist_idx2 = left_dist_idx1 + num_obj_points

        # Distance between hands
        hands_dist_idx = left_dist_idx2 + 1

        # Object activations
        obj_act_idx = hands_dist_idx + 2

        for i, (features, target) in enumerate(zip(window, targets)):
            class_ids = self.obj_ids

            relevant_objs = self.act_to_objs[self.actions_dict[target]]
            relevant_objs = list(set([obj for pair in relevant_objs for obj in pair]))
            print("relevant objs", relevant_objs)
            if "hand" in relevant_objs:
                relevant_objs.remove("hand")
                relevant_objs.append("hand (left)")
                relevant_objs.append("hand (right)")

            
            relevant_ids = [self.obj_dict[obj] for obj in relevant_objs]
            print(relevant_ids)

            for r in relevant_ids:
                class_ids.remove(r)

            zero_class_ids = random.sample(class_ids, len(class_ids) * self.dropout_rate)

            obj_ids_no_hands = self.obj_ids[3:]
            for start_idx in [right_dist_idx1, left_dist_idx1]:
                for class_id in zero_class_ids:
                    print(f"ids: {obj_ids_no_hands}, class id: {class_id}")
                    idx = obj_ids_no_hands.index(class_id)
                    features[start_idx+idx:start_idx+idx+1] = 0

            start_idx = obj_act_idx
            for class_id in zero_class_ids:
                idx = obj_ids_no_hands.index(class_id)
                features[start_idx+idx] = 0
            
            window[i] = features
        return window
