import os
import os.path as osp
import torch
import numpy as np
import scipy.misc as m
import json
from torch.utils import data
from tqdm import tqdm
from data.city_utils import recursive_glob
from data.augmentations import *
import imageio

def gen_cityscapes_label2img(root, split="train"):
    print('start generating cityscapes_ids2path')
    save_path = './data/cityscapes_ids2path.json'
    if osp.exists(save_path):
        print(save_path+' already exist')
        return
    images_base = os.path.join(root, "leftImg8bit", split)
    annotations_base = os.path.join(root, "gtFine", split)
    files = recursive_glob(rootdir=images_base, suffix=".png")
    void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, ]
    ignore_index = 255
    class_map = dict(zip(valid_classes, range(19)))
    if not files:
        raise Exception("No files for split=[%s] found in %s" % (split, images_base))
    res_to_save = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                   '10': [], '11': [], '12': [], '13': [], '14': [], '15': [], '16': [], '17': [], '18': []}
    t_files = tqdm(files)
    for img_path in t_files:
        lbl_path = os.path.join(
            annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )
        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)
        for _voidc in void_classes:
            lbl[lbl == _voidc] = ignore_index
        for _validc in valid_classes:
            lbl[lbl == _validc] = class_map[_validc]
        for i in range(19):
            if i in lbl:
                res_to_save[str(i)].append([img_path, lbl_path])
    with open(save_path, 'w') as f:
        json.dump(res_to_save, f, indent=4, separators=(',', ': '))

def gen_gta5_label2img(root, list_path = './data/gta5_list/train.txt', split="train"):
    print('start generating gta5_ids2path')
    save_path = './data/gta5_ids2path.json'
    if osp.exists(save_path):
        print(save_path + ' already exist')
        return
    img_ids = [i_id.strip() for i_id in open(list_path)]
    id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                     26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
    res_to_save = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                   '10': [], '11': [], '12': [], '13': [], '14': [], '15': [], '16': [], '17': [], '18': []}
    t_files = tqdm(img_ids)

    index = 0
    for name in t_files:
        # index = index + 1
        # if index == 1000:
        #     break
        img_path = osp.join(root, "images/%s" % name)
        label_path = osp.join(root, "labels/%s" % name)
        lbl = Image.open(label_path)
        lbl = np.array(lbl, dtype=np.uint8)
        label_copy = 255 * np.ones(lbl.shape, dtype=np.uint8)
        for k, v in id_to_trainid.items():
            label_copy[lbl == k] = v
        for i in range(19):
            if i in label_copy:
                res_to_save[str(i)].append(name)
    with open(save_path, 'w') as f:
        json.dump(res_to_save, f, indent=4, separators=(',', ': '))

def gen_synthia_label2img(root, list_path = './data/synthia_list/train.txt', split="train"):
    print('start generating synthia_ids2path')
    save_path = './data/synthia_ids2path.json'
    if osp.exists(save_path):
        print(save_path + ' already exist')
        return
    img_ids = [i_id.strip()[-11:] for i_id in open(list_path)]
    id_to_trainid = {1: 9, 2: 2, 3: 0, 4: 1, 5: 4, 6: 8,
                     7: 5, 8: 12, 9: 7, 10: 10, 11: 15, 12: 14, 15: 6,
                     17: 11, 19: 13, 21: 3}
    res_to_save = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],
                   '10': [], '11': [], '12': [], '13': [], '14': [], '15': []}
    t_files = tqdm(img_ids)

    index = 0
    for name in t_files:
        # index = index + 1
        # if index == 1000:
        #     break
        img_path = osp.join(root, "RGB/%s" % name)
        label_path = osp.join(root, "GT/LABELS/%s" % name)
        lbl = np.asarray(imageio.imread(label_path, format='PNG-FI'))[:,:,0]  # uint16
        label_copy = 255 * np.ones(lbl.shape, dtype=np.uint8)
        for k, v in id_to_trainid.items():
            label_copy[lbl == k] = v
        for i in range(16):
            if i in label_copy:
                res_to_save[str(i)].append(name)
    with open(save_path, 'w') as f:
        json.dump(res_to_save, f, indent=4, separators=(',', ': '))