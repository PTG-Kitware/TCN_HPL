import os
from .MC50 import register_MC50_instances


cooking_data_root = '/angel_workspace/angel_system/berkeley/'
_PREDEFINED_SPLITS_KITWARE_COOKING = {
    "KITWARE_COOKING_COFFEE": {
        "COFFEE_UCB_train": (f"{cooking_data_root}/berkeley/2022-11-05_whole/ft_file/images", f"{cooking_data_root}/berkeley/2022-11-05_whole/ft_file/fine-tuning.json"),
        "COFFEE_UCB_val": (f"{cooking_data_root}/berkeley/2022-11-05_whole/ft_file/images", f"{cooking_data_root}/berkeley/2022-11-05_whole/ft_file/fine-tuning.json"),
    
        # Generated annotations on videos with contact metadata
        "COFFEE_train": (f"{cooking_data_root}/coffee_recordings/extracted", f"{cooking_data_root}/coffee_recordings/extracted/coffee_preds_with_contact_train.mscoco.json"),
        "COFFEE_val": (f"{cooking_data_root}/coffee_recordings/extracted", f"{cooking_data_root}/coffee_recordings/extracted/coffee_preds_with_contact_val.mscoco.json"),
    },
}

ALL_COOOKING_CATEGORIES = {
    'KITWARE_COOKING_COFFEE': [{'id': 1, 'name': 'coffee + mug', 'instances_count': 85, 'def': '', 'synonyms': [], 'image_count': 85, 'frequency': '', 'synset': ''}, {'id': 2, 'name': 'coffee bag', 'instances_count': 545, 'def': '', 'synonyms': [], 'image_count': 545, 'frequency': '', 'synset': ''}, {'id': 3, 'name': 'coffee beans + container', 'instances_count': 36, 'def': '', 'synonyms': [], 'image_count': 36, 'frequency': '', 'synset': ''}, {'id': 4, 'name': 'coffee beans + container + scale', 'instances_count': 38, 'def': '', 'synonyms': [], 'image_count': 38, 'frequency': '', 'synset': ''}, {'id': 5, 'name': 'coffee grounds + paper filter + filter cone', 'instances_count': 40, 'def': '', 'synonyms': [], 'image_count': 40, 'frequency': '', 'synset': ''}, {'id': 6, 'name': 'coffee grounds + paper filter + filter cone + mug', 'instances_count': 102, 'def': '', 'synonyms': [], 'image_count': 102, 'frequency': '', 'synset': ''}, {'id': 7, 'name': 'container', 'instances_count': 653, 'def': '', 'synonyms': [], 'image_count': 653, 'frequency': '', 'synset': ''}, {'id': 8, 'name': 'container + scale', 'instances_count': 121, 'def': '', 'synonyms': [], 'image_count': 121, 'frequency': '', 'synset': ''}, {'id': 9, 'name': 'filter cone', 'instances_count': 161, 'def': '', 'synonyms': [], 'image_count': 161, 'frequency': '', 'synset': ''}, {'id': 10, 'name': 'filter cone + mug', 'instances_count': 138, 'def': '', 'synonyms': [], 'image_count': 138, 'frequency': '', 'synset': ''}, {'id': 11, 'name': 'grinder (close)', 'instances_count': 745, 'def': '', 'synonyms': [], 'image_count': 745, 'frequency': '', 'synset': ''}, {'id': 12, 'name': 'grinder (open)', 'instances_count': 209, 'def': '', 'synonyms': [], 'image_count': 209, 'frequency': '', 'synset': ''}, {'id': 13, 'name': 'hand (left)', 'instances_count': 911, 'def': '', 'synonyms': [], 'image_count': 911, 'frequency': '', 'synset': ''}, {'id': 14, 'name': 'hand (right)', 'instances_count': 1026, 'def': '', 'synonyms': [], 'image_count': 1026, 'frequency': '', 'synset': ''}, {'id': 15, 'name': 'kettle', 'instances_count': 812, 'def': '', 'synonyms': [], 'image_count': 812, 'frequency': '', 'synset': ''}, {'id': 16, 'name': 'kettle (open)', 'instances_count': 97, 'def': '', 'synonyms': [], 'image_count': 97, 'frequency': '', 'synset': ''}, {'id': 17, 'name': 'lid (grinder)', 'instances_count': 127, 'def': '', 'synonyms': [], 'image_count': 127, 'frequency': '', 'synset': ''}, {'id': 18, 'name': 'lid (kettle)', 'instances_count': 39, 'def': '', 'synonyms': [], 'image_count': 39, 'frequency': '', 'synset': ''}, {'id': 19, 'name': 'measuring cup', 'instances_count': 1073, 'def': '', 'synonyms': [], 'image_count': 1073, 'frequency': '', 'synset': ''}, {'id': 20, 'name': 'mug', 'instances_count': 162, 'def': '', 'synonyms': [], 'image_count': 162, 'frequency': '', 'synset': ''}, {'id': 21, 'name': 'paper filter', 'instances_count': 38, 'def': '', 'synonyms': [], 'image_count': 38, 'frequency': '', 'synset': ''}, {'id': 22, 'name': 'paper filter (quarter)', 'instances_count': 102, 'def': '', 'synonyms': [], 'image_count': 102, 'frequency': '', 'synset': ''}, {'id': 23, 'name': 'paper filter (semi)', 'instances_count': 43, 'def': '', 'synonyms': [], 'image_count': 43, 'frequency': '', 'synset': ''}, {'id': 24, 'name': 'paper filter + filter cone', 'instances_count': 39, 'def': '', 'synonyms': [], 'image_count': 39, 'frequency': '', 'synset': ''}, {'id': 25, 'name': 'paper filter + filter cone + mug', 'instances_count': 251, 'def': '', 'synonyms': [], 'image_count': 251, 'frequency': '', 'synset': ''}, {'id': 26, 'name': 'paper filter bag', 'instances_count': 655, 'def': '', 'synonyms': [], 'image_count': 655, 'frequency': '', 'synset': ''}, {'id': 27, 'name': 'paper towel', 'instances_count': 412, 'def': '', 'synonyms': [], 'image_count': 412, 'frequency': '', 'synset': ''}, {'id': 28, 'name': 'scale (off)', 'instances_count': 688, 'def': '', 'synonyms': [], 'image_count': 688, 'frequency': '', 'synset': ''}, {'id': 29, 'name': 'scale (on)', 'instances_count': 80, 'def': '', 'synonyms': [], 'image_count': 80, 'frequency': '', 'synset': ''}, {'id': 30, 'name': 'switch', 'instances_count': 513, 'def': '', 'synonyms': [], 'image_count': 513, 'frequency': '', 'synset': ''}, {'id': 31, 'name': 'thermometer (close)', 'instances_count': 453, 'def': '', 'synonyms': [], 'image_count': 453, 'frequency': '', 'synset': ''}, {'id': 32, 'name': 'thermometer (open)', 'instances_count': 147, 'def': '', 'synonyms': [], 'image_count': 147, 'frequency': '', 'synset': ''}, {'id': 33, 'name': 'timer', 'instances_count': 624, 'def': '', 'synonyms': [], 'image_count': 624, 'frequency': '', 'synset': ''}, {'id': 34, 'name': 'timer (20)', 'instances_count': 29, 'def': '', 'synonyms': [], 'image_count': 29, 'frequency': '', 'synset': ''}, {'id': 35, 'name': 'timer (30)', 'instances_count': 28, 'def': '', 'synonyms': [], 'image_count': 28, 'frequency': '', 'synset': ''}, {'id': 36, 'name': 'timer (else)', 'instances_count': 111, 'def': '', 'synonyms': [], 'image_count': 111, 'frequency': '', 'synset': ''}, {'id': 37, 'name': 'trash can', 'instances_count': 29, 'def': '', 'synonyms': [], 'image_count': 29, 'frequency': '', 'synset': ''}, {'id': 38, 'name': 'used paper filter', 'instances_count': 60, 'def': '', 'synonyms': [], 'image_count': 60, 'frequency': '', 'synset': ''}, {'id': 39, 'name': 'used paper filter + filter cone', 'instances_count': 69, 'def': '', 'synonyms': [], 'image_count': 69, 'frequency': '', 'synset': ''}, {'id': 40, 'name': 'used paper filter + filter cone + mug', 'instances_count': 8, 'def': '', 'synonyms': [], 'image_count': 8, 'frequency': '', 'synset': ''}, {'id': 41, 'name': 'water', 'instances_count': 603, 'def': '', 'synonyms': [], 'image_count': 603, 'frequency': '', 'synset': ''}, {'id': 42, 'name': 'water + coffee grounds + paper filter + filter cone + mug', 'instances_count': 252, 'def': '', 'synonyms': [], 'image_count': 252, 'frequency': '', 'synset': ''}]
}

def _get_cooking_instances_meta_v1(dataset_name):
    # assert len(LVIS_V1_CATEGORIES) == 20
    MC50_CATEGORIES = ALL_COOOKING_CATEGORIES[dataset_name]
    cat_ids = [k["id"] for k in MC50_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    MC50_categories = sorted(MC50_CATEGORIES, key=lambda x: x["id"])
    # thing_classes = [k["synonyms"][0] for k in lvis_categories]
    thing_classes = [k["name"] for k in MC50_categories]

    if dataset_name == 'KITWARE_COOKING_COFFEE':
        from angel_system.berkeley.utils.data.objects import coffee_activity_objects as cooking_activity_objects

    meta = {
        "thing_classes": thing_classes,
        "class_image_count": MC50_CATEGORIES,
        "sub_steps": cooking_activity_objects.sub_steps,
        "original_sub_steps": cooking_activity_objects.original_sub_steps,
        "contact_pairs_details": cooking_activity_objects.contact_pairs_details,
        "CONTACT_PAIRS": cooking_activity_objects.CONTACT_PAIRS,
        "States_Pairs": cooking_activity_objects.States_Pairs
    }
    return meta

def register_all_kitware_cooking_data():
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_KITWARE_COOKING.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_MC50_instances(
                key,
                _get_cooking_instances_meta_v1(dataset_name),
                json_file,
                image_root,
            )
