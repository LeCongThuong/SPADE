"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_mask(image):
    try:
        # Step 1: Identify pixels not equal to 255
        not_255_y, not_255_x = np.where(np.squeeze(image) != 1)

        # Step 2: Determine the bounding rectangle's min and max coordinates
        x_min, y_min = not_255_x.min(), not_255_y.min()
        x_max, y_max = not_255_x.max(), not_255_y.max()

        # Step 3: Create a mask for the bounding rectangle
        mask = np.zeros_like(image, dtype=bool)
        mask[y_min:y_max+1, x_min:x_max+1] = True
    except Exception as e:
        print(e)
    return mask


def save_imgs(img_batch, print_img_batch, file_name_path_list, dest_img_dir):
    
    depth_img_dir = os.path.join(dest_img_dir, "depth")
    depth_map_dir = os.path.join(dest_img_dir, "np")
    Path(depth_img_dir).mkdir(exist_ok=True, parents=True)
    Path(depth_map_dir).mkdir(exist_ok=True, parents=True)

    dest_depth_map_path_list = [Path(depth_map_dir) / Path(file_name_path).name for file_name_path in file_name_path_list]
    dest_depth_img_path_list = [Path(depth_img_dir) / Path(file_name_path).name for file_name_path in file_name_path_list]

    num_imgs = img_batch.shape[0]
    for index in range(num_imgs):
        img = img_batch[index, :, :, :]
        print_img = np.transpose(print_img_batch[index, :, :, :], (1, 2, 0))
        mask = create_mask(print_img)
        depth_map_file_path = dest_depth_map_path_list[index]
        depth_img_file_path = dest_depth_img_path_list[index]
        np_file_path = os.path.join(depth_map_dir, Path(depth_map_file_path).stem + ".npy")
        depth_file_path = os.path.join(depth_img_dir, Path(depth_img_file_path).stem + ".png")
        img = np.transpose(img, (1, 2, 0))
        with open(np_file_path, "wb") as f:
            np.save(f, img)
        img = np.where(mask, img, 1)
        img = np.uint8(255*img).reshape(img.shape[0], img.shape[1])
        Image.fromarray(img).save(depth_file_path)

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()



# test
for i, data_i in tqdm(enumerate(dataloader)):
    # if i * opt.batchSize >= opt.how_many:
    #     break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    input_data = data_i["label"]
    
    save_imgs(generated.detach().cpu().numpy(), input_data.detach().cpu().numpy(), img_path, opt.results_dir)