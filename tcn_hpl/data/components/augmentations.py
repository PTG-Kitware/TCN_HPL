import random
import torch

from tcn_hpl.data.vectorize_classic import feature_version_to_options


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


class MoveCenterPts(torch.nn.Module):
    """Simulate moving the center points of the bounding boxes by
    adjusting the distances for each frame
    """

    def __init__(
        self,
        hand_dist_delta,
        obj_dist_delta,
        joint_dist_delta,
        im_w,
        im_h,
        num_obj_classes,
        feat_version,
        top_k_objects,
    ):
        """
        :param hand_dist_delta: Decimal percentage to calculate the +-offset in
            pixels for the hands
        :param obj_dist_delta: Decimal percentage to calculate the +-offset in
            pixels for the objects
        :param joint_dist_delta: Decimal percentage to calculate the +-offset in
            pixels for the joints
        :param w: Width of the frames
        :param h: Height of the frames
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        :param top_k_objects: Number top confidence objects to use per label
        """
        super().__init__()

        self.hand_dist_delta = hand_dist_delta
        self.obj_dist_delta = obj_dist_delta
        self.joint_dist_delta = joint_dist_delta

        self.im_w = im_w
        self.im_h = im_h

        self.num_obj_classes = num_obj_classes
        self.num_non_obj_classes = 2  # hands
        self.num_good_obj_classes = self.num_obj_classes - self.num_non_obj_classes

        self.top_k_objects = top_k_objects

        self.feat_version = feat_version
        self.opts = feature_version_to_options(self.feat_version)

        self.use_activation = self.opts.get("use_activation", False)
        self.use_hand_dist = self.opts.get("use_hand_dist", False)
        self.use_intersection = self.opts.get("use_intersection", False)
        self.use_center_dist = self.opts.get("use_center_dist", False)
        self.use_joint_hand_offset = self.opts.get("use_joint_hand_offset", False)
        self.use_joint_object_offset = self.opts.get("use_joint_object_offset", False)

        # Deltas
        self.hand_delta_x = self.im_w * self.hand_dist_delta
        self.hand_delta_y = self.im_h * self.hand_dist_delta

        self.obj_ddelta_x = self.im_w * self.obj_dist_delta
        self.obj_ddelta_y = self.im_h * self.obj_dist_delta

        self.joint_delta_x = self.im_w * self.joint_dist_delta
        self.joint_delta_y = self.im_h * self.joint_dist_delta

    def init_deltas(self):
        rhand_delta_x = random.uniform(-self.hand_delta_x, self.hand_delta_x)
        rhand_delta_y = random.uniform(-self.hand_delta_y, self.hand_delta_y)

        lhand_delta_x = random.uniform(-self.hand_delta_x, self.hand_delta_x)
        lhand_delta_y = random.uniform(-self.hand_delta_y, self.hand_delta_y)

        obj_delta_x = random.uniform(-self.obj_ddelta_x, self.obj_ddelta_x)
        obj_delta_y = random.uniform(-self.obj_ddelta_y, self.obj_ddelta_y)

        joint_delta_x = random.uniform(-self.joint_delta_x, self.joint_delta_x)
        joint_delta_y = random.uniform(-self.joint_delta_y, self.joint_delta_y)

        return (
            [rhand_delta_x, rhand_delta_y],
            [lhand_delta_x, lhand_delta_y],
            [obj_delta_x, obj_delta_y],
            [joint_delta_x, joint_delta_y],
        )

    def forward(self, features):
        for i in range(features.shape[0]):
            frame_features = features[i]

            (
                [rhand_delta_x, rhand_delta_y],
                [lhand_delta_x, lhand_delta_y],
                [obj_delta_x, obj_delta_y],
                [joint_delta_x, joint_delta_y],
            ) = self.init_deltas()

            ind = -1
            for object_k_index in range(self.top_k_objects):
                # RIGHT HAND
                if self.use_activation:
                    ind += 1  # right hand conf
                    right_hand_conf = frame_features[ind]

                if self.use_hand_dist:
                    for obj_ind in range(self.num_good_obj_classes):
                        ind += 1
                        obj_rh_dist_x = frame_features[ind]
                        new_val = (
                            obj_rh_dist_x + rhand_delta_x + obj_delta_x
                            if obj_rh_dist_x != 0
                            else obj_rh_dist_x
                        )
                        frame_features[ind] = clamp(new_val, -self.im_w, self.im_w)

                        ind += 1
                        obj_rh_dist_y = frame_features[ind]
                        new_val = (
                            obj_rh_dist_y + rhand_delta_y + obj_delta_y
                            if obj_rh_dist_y != 0
                            else obj_rh_dist_y
                        )
                        frame_features[ind] = clamp(new_val, -self.im_h, self.im_h)

                if self.use_center_dist:
                    ind += 1
                    rh_im_center_dist_x = frame_features[ind]
                    new_val = (
                        rh_im_center_dist_x + rhand_delta_x
                        if rh_im_center_dist_x != 0
                        else rh_im_center_dist_x
                    )
                    frame_features[ind] = frame_features[ind] = clamp(
                        new_val, -self.im_w, self.im_w
                    )

                    ind += 1
                    rh_im_center_dist_y = frame_features[ind]
                    new_val = (
                        rh_im_center_dist_y + rhand_delta_y
                        if rh_im_center_dist_y != 0
                        else rh_im_center_dist_y
                    )
                    frame_features[ind] = clamp(new_val, -self.im_h, self.im_h)

                # LEFT HAND
                if self.use_activation:
                    ind += 1  # left hand conf

                if self.use_hand_dist:
                    # Left hand distances
                    for obj_ind in range(self.num_good_obj_classes):
                        ind += 1
                        obj_lh_dist_x = frame_features[ind]
                        new_val = (
                            obj_lh_dist_x + lhand_delta_x + obj_delta_x
                            if obj_lh_dist_x != 0
                            else obj_lh_dist_x
                        )
                        frame_features[ind] = clamp(new_val, -self.im_w, self.im_w)

                        ind += 1
                        obj_lh_dist_y = frame_features[ind]
                        new_val = (
                            obj_lh_dist_y + lhand_delta_y + obj_delta_y
                            if obj_lh_dist_y != 0
                            else obj_lh_dist_y
                        )
                        frame_features[ind] = clamp(new_val, -self.im_h, self.im_h)

                if self.use_center_dist:
                    ind += 1
                    lh_im_center_dist_x = frame_features[ind]
                    new_val = (
                        lh_im_center_dist_x + lhand_delta_x
                        if lh_im_center_dist_x != 0
                        else lh_im_center_dist_x
                    )
                    frame_features[ind] = clamp(new_val, -self.im_w, self.im_w)

                    ind += 1
                    lh_im_center_dist_y = frame_features[ind]
                    new_val = (
                        lh_im_center_dist_y + lhand_delta_y
                        if lh_im_center_dist_y != 0
                        else lh_im_center_dist_y
                    )
                    frame_features[ind] = clamp(new_val, -self.im_h, self.im_h)

                # Right - left hand
                if self.use_hand_dist:
                    # Right - left hand distance
                    ind += 1
                    rh_lh_dist_x = frame_features[ind]
                    new_val = (
                        rh_lh_dist_x + rhand_delta_x + lhand_delta_x
                        if rh_lh_dist_x != 0
                        else rh_lh_dist_x
                    )
                    frame_features[ind] = clamp(new_val, -self.im_w, self.im_w)

                    ind += 1
                    rh_lh_dist_y = frame_features[ind]
                    new_val = (
                        rh_lh_dist_y + rhand_delta_y + lhand_delta_y
                        if rh_lh_dist_y != 0
                        else rh_lh_dist_y
                    )
                    frame_features[ind] = clamp(new_val, -self.im_h, self.im_h)

                if self.use_intersection:
                    ind += 1  # lh - rh intersection

                # OBJECTS
                for obj_ind in range(self.num_good_obj_classes):
                    if self.use_activation:
                        ind += 1  # Object confidence

                    if self.use_intersection:
                        ind += 2  # obj - hands intersection

                    if self.use_center_dist:
                        # image center - obj distances
                        ind += 1
                        obj_im_center_dist_x = frame_features[ind]
                        new_val = (
                            obj_im_center_dist_x + obj_delta_x
                            if obj_im_center_dist_x != 0
                            else obj_im_center_dist_x
                        )
                        frame_features[ind] = clamp(new_val, -self.im_w, self.im_w)

                        ind += 1
                        obj_im_center_dist_y = frame_features[ind]
                        new_val = (
                            obj_im_center_dist_y + obj_delta_y
                            if obj_im_center_dist_y != 0
                            else obj_im_center_dist_y
                        )
                        frame_features[ind] = clamp(new_val, -self.im_h, self.im_h)

            # HANDS-JOINTS
            # Don't clamp the joint values because they can be off the frame originally
            if self.use_joint_hand_offset:
                # left hand - joints distances
                for i in range(22):
                    ind += 1
                    lh_jointi_dist_x = frame_features[ind]
                    frame_features[ind] = (
                        lh_jointi_dist_x + lhand_delta_x + joint_delta_x
                        if lh_jointi_dist_x != 0
                        else lh_jointi_dist_x
                    )

                    ind += 1
                    lh_jointi_dist_y = frame_features[ind]
                    frame_features[ind] = (
                        lh_jointi_dist_y + lhand_delta_y + joint_delta_y
                        if lh_jointi_dist_y != 0
                        else lh_jointi_dist_y
                    )

                # right hand - joints distances
                for i in range(22):
                    ind += 1
                    rh_jointi_dist_x = frame_features[ind]
                    frame_features[ind] = (
                        rh_jointi_dist_x + rhand_delta_x + joint_delta_x
                        if rh_jointi_dist_x != 0
                        else rh_jointi_dist_x
                    )

                    ind += 1
                    rh_jointi_dist_y = frame_features[ind]
                    frame_features[ind] = (
                        rh_jointi_dist_y + rhand_delta_y + joint_delta_y
                        if rh_jointi_dist_y != 0
                        else rh_jointi_dist_y
                    )

            # OBJS-JOINTS
            # Don't clamp the joint values because they can be off the frame originally
            if self.use_joint_object_offset:
                for object_k_index in range(self.top_k_objects):
                    # obj - joints distances
                    for obj_ind in range(self.num_good_obj_classes):
                        joints_dists = []
                        for i in range(22):
                            ind += 1
                            obj_jointi_dist_x = frame_features[ind]
                            frame_features[ind] = (
                                obj_jointi_dist_x + obj_delta_x + joint_delta_x
                                if obj_jointi_dist_x != 0
                                else obj_jointi_dist_x
                            )

                            ind += 1
                            obj_jointi_dist_y = frame_features[ind]
                            frame_features[ind] = (
                                obj_jointi_dist_y + obj_delta_y + joint_delta_y
                                if obj_jointi_dist_y != 0
                                else obj_jointi_dist_y
                            )

            features[i] = frame_features
        return features

    def __repr__(self) -> str:
        detail = f"(hand_dist_delta={self.hand_dist_delta}, obj_dist_delta={self.obj_dist_delta}, joint_dist_delta={self.joint_dist_delta}, im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version}, top_k_object={self.top_k_objects})"
        return f"{self.__class__.__name__}{detail}"


class ActivationDelta(torch.nn.Module):
    """Update the activation feature of each class by +-``conf_delta``"""

    def __init__(self, conf_delta, num_obj_classes, feat_version, top_k_objects):
        """
        :param conf delta:
        :param num_obj_classes: Number of object classes from the detections
            used to generate the features
        :param feat_version: Algorithm version used to generate the input features
        :param top_k_objects: Number top confidence objects to use per label
        """
        super().__init__()

        self.conf_delta = conf_delta

        self.num_obj_classes = num_obj_classes
        self.num_non_obj_classes = 2  # hands
        self.num_good_obj_classes = self.num_obj_classes - self.num_non_obj_classes

        self.top_k_objects = top_k_objects

        self.feat_version = feat_version
        self.opts = feature_version_to_options(self.feat_version)

        self.use_activation = self.opts.get("use_activation", False)
        self.use_hand_dist = self.opts.get("use_hand_dist", False)
        self.use_intersection = self.opts.get("use_intersection", False)
        self.use_center_dist = self.opts.get("use_center_dist", False)
        self.use_joint_hand_offset = self.opts.get("use_joint_hand_offset", False)
        self.use_joint_object_offset = self.opts.get("use_joint_object_offset", False)

    def init_delta(self):
        delta = random.uniform(-self.conf_delta, self.conf_delta)

        return delta

    def forward(self, features):
        delta = self.init_delta()

        for i in range(features.shape[0]):
            frame_features = features[i]

            ind = -1
            for object_k_index in range(self.top_k_objects):
                # RIGHT HAND
                if self.use_activation:
                    ind += 1
                    right_hand_conf = frame_features[ind]

                    if right_hand_conf != 0:
                        frame_features[ind] = frame_features[ind] = clamp(
                            right_hand_conf + delta, 0, 1
                        )

                if self.use_hand_dist:
                    for obj_ind in range(self.num_good_obj_classes):
                        ind += 2  # rh - obj distance

                if self.use_center_dist:
                    ind += 2  # rh - center dist

                # LEFT HAND
                if self.use_activation:
                    ind += 1
                    left_hand_conf = frame_features[ind]

                    if left_hand_conf != 0:
                        frame_features[ind] = clamp(left_hand_conf + delta, 0, 1)

                if self.use_hand_dist:
                    # Left hand distances
                    for obj_ind in range(self.num_good_obj_classes):
                        ind += 2  # lh - obj dist

                if self.use_center_dist:
                    ind += 2  # lh - center dist

                # Right - left hand
                if self.use_hand_dist:
                    ind += 2  # Right - left hand distance
                if self.use_intersection:
                    ind += 1  # rh - lh intersection

                # OBJECTS
                for obj_ind in range(self.num_good_obj_classes):
                    if self.use_activation:
                        # Object confidence
                        ind += 1
                        obj_conf = frame_features[ind]

                        if obj_conf != 0:
                            frame_features[ind] = clamp(obj_conf + delta, 0, 1)

                    if self.use_intersection:
                        # obj - hand intersection
                        ind += 2

                    if self.use_center_dist:
                        ind += 2  # image center - obj distances

            # HANDS-JOINTS
            if self.use_joint_hand_offset:
                # left hand - joints distances
                for i in range(22):
                    ind += 2

                # right hand - joints distances
                for i in range(22):
                    ind += 2

            # OBJS-JOINTS
            if self.use_joint_object_offset:
                for object_k_index in range(self.top_k_objects):
                    # obj - joints distances
                    for obj_ind in range(self.num_good_obj_classes):
                        joints_dists = []
                        for i in range(22):
                            ind += 2

            features[i] = frame_features
        return features

    def __repr__(self) -> str:
        detail = f"(conf_delta={self.conf_delta}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version}, top_k_objects={self.top_k_objects})"
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
        :param top_k_objects: Number top confidence objects to use per label
        """
        super().__init__()

        self.im_w = im_w
        self.im_h = im_h

        self.num_obj_classes = num_obj_classes
        self.num_non_obj_classes = 2  # hands
        self.num_good_obj_classes = self.num_obj_classes - self.num_non_obj_classes

        self.top_k_objects = top_k_objects

        self.feat_version = feat_version
        self.opts = feature_version_to_options(self.feat_version)

        self.use_activation = self.opts.get("use_activation", False)
        self.use_hand_dist = self.opts.get("use_hand_dist", False)
        self.use_intersection = self.opts.get("use_intersection", False)
        self.use_center_dist = self.opts.get("use_center_dist", False)
        self.use_joint_hand_offset = self.opts.get("use_joint_hand_offset", False)
        self.use_joint_object_offset = self.opts.get("use_joint_object_offset", False)

    def forward(self, features):
        for i in range(features.shape[0]):
            frame_features = features[i]

            ind = -1
            for object_k_index in range(self.top_k_objects):
                # RIGHT HAND
                if self.use_activation:
                    ind += 1  # right hand confidence

                if self.use_hand_dist:
                    for obj_ind in range(self.num_good_obj_classes):
                        ind += 1
                        obj_rh_dist_x = frame_features[ind]
                        frame_features[ind] = obj_rh_dist_x / self.im_w

                        ind += 1
                        obj_rh_dist_y = frame_features[ind]
                        frame_features[ind] = obj_rh_dist_y / self.im_h

                if self.use_center_dist:
                    # right hand - image center distance
                    ind += 2

                # LEFT HAND
                if self.use_activation:
                    ind += 1  # left hand confidence

                if self.use_hand_dist:
                    # Left hand distances
                    for obj_ind in range(self.num_good_obj_classes):
                        ind += 1
                        obj_lh_dist_x = frame_features[ind]
                        frame_features[ind] = obj_lh_dist_x / self.im_w

                        ind += 1
                        obj_lh_dist_y = frame_features[ind]
                        frame_features[ind] = obj_lh_dist_y / self.im_h

                if self.use_center_dist:
                    # left hand - image center distance
                    ind += 2

                # Right - left hand
                if self.use_hand_dist:
                    # Right - left hand distance
                    ind += 1
                    rh_lh_dist_x = frame_features[ind]
                    frame_features[ind] = rh_lh_dist_x / self.im_w

                    ind += 1
                    rh_lh_dist_y = frame_features[ind]
                    frame_features[ind] = rh_lh_dist_y / self.im_h
                if self.use_intersection:
                    ind += 1  # right - left hadn intersection

                # OBJECTS
                for obj_ind in range(self.num_good_obj_classes):
                    if self.use_activation:
                        ind += 1  # Object confidence
                        obj_conf = frame_features[ind]

                    if self.use_intersection:
                        # obj - hands intersection
                        ind += 2

                    if self.use_center_dist:
                        # image center - obj distances
                        ind += 2

            # HANDS-JOINTS
            if self.use_joint_hand_offset:
                # left hand - joints distances
                for i in range(22):
                    ind += 1
                    lh_jointi_dist_x = frame_features[ind]
                    frame_features[ind] = lh_jointi_dist_x / self.im_w

                    ind += 1
                    lh_jointi_dist_y = frame_features[ind]
                    frame_features[ind] = lh_jointi_dist_y / self.im_h

                # right hand - joints distances
                for i in range(22):
                    ind += 1
                    rh_jointi_dist_x = frame_features[ind]
                    frame_features[ind] = rh_jointi_dist_x / self.im_w

                    ind += 1
                    rh_jointi_dist_y = frame_features[ind]
                    frame_features[ind] = rh_jointi_dist_y / self.im_h

            # OBJS-JOINTS
            if self.use_joint_object_offset:
                for object_k_index in range(self.top_k_objects):
                    # obj - joints distances
                    for obj_ind in range(self.num_good_obj_classes):
                        joints_dists = []
                        for i in range(22):
                            ind += 1
                            obj_jointi_dist_x = frame_features[ind]
                            frame_features[ind] = obj_jointi_dist_x / self.im_w

                            ind += 1
                            obj_jointi_dist_y = frame_features[ind]
                            frame_features[ind] = obj_jointi_dist_y / self.im_h

            features[i] = frame_features
        return features

    def __repr__(self) -> str:
        detail = f"(im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version}, top_k_objects={self.top_k_objects})"
        return f"{self.__class__.__name__}{detail}"


class NormalizeFromCenter(torch.nn.Module):
    """Normalize the distances from -1 to 1 with respect to the image center

    Missing objects will be set to (2, 2)
    """

    def __init__(self, im_w, im_h, num_obj_classes, feat_version, top_k_objects):
        """
        :param w: Width of the frames
        :param h: Height of the frames
        :param feat_version: Algorithm version used to generate the input features
        :param top_k_objects: Number top confidence objects to use per label
        """
        super().__init__()

        self.im_w = im_w
        self.half_w = im_w / 2
        self.im_h = im_h
        self.half_h = im_h / 2

        self.num_obj_classes = num_obj_classes
        self.num_non_obj_classes = 2  # hands
        self.num_good_obj_classes = self.num_obj_classes - self.num_non_obj_classes

        self.top_k_objects = top_k_objects

        self.feat_version = feat_version
        self.opts = feature_version_to_options(self.feat_version)

        self.use_activation = self.opts.get("use_activation", False)
        self.use_hand_dist = self.opts.get("use_hand_dist", False)
        self.use_intersection = self.opts.get("use_intersection", False)
        self.use_center_dist = self.opts.get("use_center_dist", False)
        self.use_joint_hand_offset = self.opts.get("use_joint_hand_offset", False)
        self.use_joint_object_offset = self.opts.get("use_joint_object_offset", False)

    def forward(self, features):
        for i in range(features.shape[0]):
            frame_features = features[i]

            ind = -1
            for object_k_index in range(self.top_k_objects):
                # RIGHT HAND
                if self.use_activation:
                    ind += 1  # right hand conf

                if self.use_hand_dist:
                    for obj_ind in range(self.num_good_obj_classes):
                        # right hand - obj distance
                        ind += 2

                if self.use_center_dist:
                    ind += 1
                    rh_im_center_dist_x = frame_features[ind]
                    frame_features[ind] = rh_im_center_dist_x / self.half_w

                    ind += 1
                    rh_im_center_dist_y = frame_features[ind]
                    frame_features[ind] = rh_im_center_dist_y / self.half_h

                # LEFT HAND
                if self.use_activation:
                    ind += 1  # left hand conf

                if self.use_hand_dist:
                    # Left hand distances
                    for obj_ind in range(self.num_good_obj_classes):
                        # left hand - obj dist
                        ind += 2

                if self.use_center_dist:
                    ind += 1
                    lh_im_center_dist_x = frame_features[ind]
                    frame_features[ind] = lh_im_center_dist_x / self.half_w

                    ind += 1
                    lh_im_center_dist_y = frame_features[ind]
                    frame_features[ind] = lh_im_center_dist_y / self.half_h

                # Right - left hand
                if self.use_hand_dist:
                    # Right - left hand distance
                    ind += 2
                if self.use_intersection:
                    ind += 1  # right - left hand intersection

                # OBJECTS
                for obj_ind in range(self.num_good_obj_classes):
                    if self.use_activation:
                        ind += 1  # Object confidence

                    if self.use_intersection:
                        # obj - hand intersection
                        ind += 2

                    if self.use_center_dist:
                        # image center - obj distances
                        ind += 1
                        obj_im_center_dist_x = frame_features[ind]
                        frame_features[ind] = obj_im_center_dist_x / self.half_w

                        ind += 1
                        obj_im_center_dist_y = frame_features[ind]
                        frame_features[ind] = obj_im_center_dist_y / self.half_h

            # HANDS-JOINTS
            if self.use_joint_hand_offset:
                # left hand - joints distances
                for i in range(22):
                    ind += 2

                # right hand - joints distances
                for i in range(22):
                    ind += 2

            # OBJS-JOINTS
            if self.use_joint_object_offset:
                for object_k_index in range(self.top_k_objects):
                    # obj - joints distances
                    for obj_ind in range(self.num_good_obj_classes):
                        joints_dists = []
                        for i in range(22):
                            ind += 2

            features[i] = frame_features
        return features

    def __repr__(self) -> str:
        detail = f"(im_w={self.im_w}, im_h={self.im_h}, feat_version={self.feat_version}, top_k_objects={self.top_k_objects})"
        return f"{self.__class__.__name__}{detail}"


class DropoutObjects(torch.nn.Module):
    """Drop out Objects"""

    def __init__(
        self, dropout_probablity, num_obj_classes, feat_version, top_k_objects
    ):
        """
        :param dropout_probablity: probablity that a given frame will NOT have an object
        """
        super().__init__()

        raise NotImplementedError

        self.dropout_probablity = dropout_probablity

        self.num_obj_classes = num_obj_classes
        self.num_non_obj_classes = 2  # hands
        self.num_good_obj_classes = self.num_obj_classes - self.num_non_obj_classes

        self.top_k_objects = top_k_objects

        self.feat_version = feat_version
        self.opts = feature_version_to_options(self.feat_version)

        self.use_activation = self.opts.get("use_activation", False)
        self.use_hand_dist = self.opts.get("use_hand_dist", False)
        self.use_intersection = self.opts.get("use_intersection", False)
        self.use_center_dist = self.opts.get("use_center_dist", False)
        self.use_joint_hand_offset = self.opts.get("use_joint_hand_offset", False)
        self.use_joint_object_offset = self.opts.get("use_joint_object_offset", False)

        self.obj_feature_mask = list()

        ind = -1
        for object_k_index in range(self.top_k_objects):
            # RIGHT HAND
            if self.use_activation:
                ind += 1  # right hand confidence
                self.obj_feature_mask.append(0)

            if self.use_hand_dist:
                self.obj_feature_mask += [0] * 2 * self.num_good_obj_classes
                ind += 2 * self.num_good_obj_classes

            if self.use_center_dist:
                # right hand - image center distance
                ind += 2
                self.obj_feature_mask.append(0)
                self.obj_feature_mask.append(0)

            # LEFT HAND
            if self.use_activation:
                ind += 1  # left hand confidence
                self.obj_feature_mask.append(0)

            if self.use_hand_dist:
                # Left hand distances
                self.obj_feature_mask += [0] * 2 * self.num_good_obj_classes
                ind += 2 * self.num_good_obj_classes

            if self.use_center_dist:
                # left hand - image center distance
                ind += 2
                self.obj_feature_mask.append(0)
                self.obj_feature_mask.append(0)

            # Right - left hand
            if self.use_hand_dist:
                # Right - left hand distance
                ind += 2
                self.obj_feature_mask.append(0)
                self.obj_feature_mask.append(0)
            if self.use_intersection:
                ind += 1  # right - left hadn intersection
                self.obj_feature_mask.append(0)
            # OBJECTS
            for obj_ind in range(self.num_good_obj_classes):
                if self.use_activation:
                    ind += 1  # Object confidence
                    self.obj_feature_mask.append(0)

                if self.use_intersection:
                    # obj - hands intersection
                    ind += 2
                    self.obj_feature_mask.append(0)
                    self.obj_feature_mask.append(0)

                if self.use_center_dist:
                    # image center - obj distances
                    ind += 2
                    self.obj_feature_mask.append(0)
                    self.obj_feature_mask.append(0)

        # HANDS-JOINTS
        if self.use_joint_hand_offset:
            # left hand - joints distances
            ind += 44
            self.obj_feature_mask += [1] * 44

            # right hand - joints distances
            ind += 44
            self.obj_feature_mask += [1] * 44

        # OBJS-JOINTS
        if self.use_joint_object_offset:
            self.obj_feature_mask += (
                [0] * 44 * self.top_k_objects * self.num_good_obj_classes
            )
            ind += 44 * self.top_k_objects * self.num_good_obj_classes

        self.obj_feature_mask = torch.tensor(self.obj_feature_mask)

    def forward(self, features):
        num_frames = features.shape[0]
        # Pick starting location of random mask
        start = random.randint(0, self.skip_stride)
        # Create mask (one element for each frame)
        mask = torch.rand(num_frames) > self.dropout_probablity

        if self.dropout_last:
            mask[-1] = 0

        return features

    def __repr__(self) -> str:
        detail = f"(im_w={self.im_w}, im_h={self.im_h}, num_obj_classes={self.num_obj_classes}, feat_version={self.feat_version}, top_k_objects={self.top_k_objects})"
        return f"{self.__class__.__name__}{detail}"
