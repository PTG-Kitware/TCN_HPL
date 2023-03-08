# Copyright (c) Facebook, Inc. and its affiliates.
# Autogen with
# with open("lvis_v1_train.json", "r") as f:
#     a = json.load(f)
# c = a["categories"]
# for x in c:
# del x["name"]
# del x["instance_count"]
# del x["def"]
# del x["synonyms"]
# del x["frequency"]
# del x["synset"]
# LVIS_CATEGORY_IMAGE_COUNT = repr(c) + "  # noqa"
# with open("/tmp/lvis_category_image_count.py", "wt") as f:
#     f.write(f"LVIS_CATEGORY_IMAGE_COUNT = {LVIS_CATEGORY_IMAGE_COUNT}")
# Then paste the contents of that file below

# fmt: off

LVIS_CATEGORY_IMAGE_COUNT = LVIS_CATEGORIES = [{'id': 1, 'name': 'coffee + mug', 'instances_count': 1, 'def': '', 'synonyms': [], 'image_count': 1, 'frequency': '', 'synset': ''}, {'id': 2, 'name': 'coffee bag', 'instances_count': 45, 'def': '', 'synonyms': [], 'image_count': 44, 'frequency': '', 'synset': ''}, {'id': 3, 'name': 'coffee beans + container', 'instances_count': 1, 'def': '', 'synonyms': [], 'image_count': 1, 'frequency': '', 'synset': ''}, {'id': 4, 'name': 'coffee beans + contrainer + scale', 'instances_count': 4, 'def': '', 'synonyms': [], 'image_count': 4, 'frequency': '', 'synset': ''}, {'id': 5, 'name': 'coffee grounds + paper filter + filter cone + mug', 'instances_count': 21, 'def': '', 'synonyms': [], 'image_count': 21, 'frequency': '', 'synset': ''}, {'id': 6, 'name': 'container', 'instances_count': 160, 'def': '', 'synonyms': [], 'image_count': 159, 'frequency': '', 'synset': ''}, {'id': 7, 'name': 'container + scale', 'instances_count': 6, 'def': '', 'synonyms': [], 'image_count': 6, 'frequency': '', 'synset': ''}, {'id': 8, 'name': 'filter cone', 'instances_count': 20, 'def': '', 'synonyms': [], 'image_count': 19, 'frequency': '', 'synset': ''}, {'id': 9, 'name': 'filter cone + mug', 'instances_count': 9, 'def': '', 'synonyms': [], 'image_count': 9, 'frequency': '', 'synset': ''}, {'id': 10, 'name': 'grinder', 'instances_count': 147, 'def': '', 'synonyms': [], 'image_count': 147, 'frequency': '', 'synset': ''}, {'id': 11, 'name': 'kettle', 'instances_count': 119, 'def': '', 'synonyms': [], 'image_count': 118, 'frequency': '', 'synset': ''}, {'id': 12, 'name': 'kettle (empty)', 'instances_count': 3, 'def': '', 'synonyms': [], 'image_count': 3, 'frequency': '', 'synset': ''}, {'id': 13, 'name': 'kettle (full)', 'instances_count': 8, 'def': '', 'synonyms': [], 'image_count': 8, 'frequency': '', 'synset': ''}, {'id': 14, 'name': 'measuring cup (empty)', 'instances_count': 156, 'def': '', 'synonyms': [], 'image_count': 156, 'frequency': '', 'synset': ''}, {'id': 15, 'name': 'measuring cup (full)', 'instances_count': 5, 'def': '', 'synonyms': [], 'image_count': 5, 'frequency': '', 'synset': ''}, {'id': 16, 'name': 'mug', 'instances_count': 21, 'def': '', 'synonyms': [], 'image_count': 21, 'frequency': '', 'synset': ''}, {'id': 17, 'name': 'paper filter', 'instances_count': 12, 'def': '', 'synonyms': [], 'image_count': 12, 'frequency': '', 'synset': ''}, {'id': 18, 'name': 'paper filter (quarter)', 'instances_count': 7, 'def': '', 'synonyms': [], 'image_count': 7, 'frequency': '', 'synset': ''}, {'id': 19, 'name': 'paper filter (semi)', 'instances_count': 4, 'def': '', 'synonyms': [], 'image_count': 4, 'frequency': '', 'synset': ''}, {'id': 20, 'name': 'paper filter + filter cone + mug', 'instances_count': 36, 'def': '', 'synonyms': [], 'image_count': 36, 'frequency': '', 'synset': ''}, {'id': 21, 'name': 'paper filter bag', 'instances_count': 94, 'def': '', 'synonyms': [], 'image_count': 94, 'frequency': '', 'synset': ''}, {'id': 22, 'name': 'paper towel', 'instances_count': 94, 'def': '', 'synonyms': [], 'image_count': 92, 'frequency': '', 'synset': ''}, {'id': 23, 'name': 'scale (off)', 'instances_count': 143, 'def': '', 'synonyms': [], 'image_count': 143, 'frequency': '', 'synset': ''}, {'id': 24, 'name': 'scale (on)', 'instances_count': 9, 'def': '', 'synonyms': [], 'image_count': 9, 'frequency': '', 'synset': ''}, {'id': 25, 'name': 'thermometer', 'instances_count': 128, 'def': '', 'synonyms': [], 'image_count': 128, 'frequency': '', 'synset': ''}, {'id': 26, 'name': 'timer', 'instances_count': 87, 'def': '', 'synonyms': [], 'image_count': 87, 'frequency': '', 'synset': ''}, {'id': 27, 'name': 'timer (0)', 'instances_count': 2, 'def': '', 'synonyms': [], 'image_count': 2, 'frequency': '', 'synset': ''}, {'id': 28, 'name': 'timer (20)', 'instances_count': 2, 'def': '', 'synonyms': [], 'image_count': 2, 'frequency': '', 'synset': ''}, {'id': 29, 'name': 'timer (30)', 'instances_count': 1, 'def': '', 'synonyms': [], 'image_count': 1, 'frequency': '', 'synset': ''}, {'id': 30, 'name': 'used paper filter', 'instances_count': 0, 'def': '', 'synonyms': [], 'image_count': 0, 'frequency': '', 'synset': ''}, {'id': 31, 'name': 'used paper filter + filter cone', 'instances_count': 1, 'def': '', 'synonyms': [], 'image_count': 1, 'frequency': '', 'synset': ''}, {'id': 32, 'name': 'used paper filter + filter cone + mug', 'instances_count': 4, 'def': '', 'synonyms': [], 'image_count': 4, 'frequency': '', 'synset': ''}, {'id': 33, 'name': 'water', 'instances_count': 18, 'def': '', 'synonyms': [], 'image_count': 18, 'frequency': '', 'synset': ''}, {'id': 34, 'name': 'water + coffee grounds + paper filter + filter cone + mug', 'instances_count': 74, 'def': '', 'synonyms': [], 'image_count': 74, 'frequency': '', 'synset': ''}]