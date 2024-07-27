import shutil
from pathlib import Path
import random
import os
from tqdm import tqdm




def split_dataset(root_dir, dest_dir, depth_img_dir, np_depth_img_dir, print_img_dir, ply_img_dir):
    print_img_path_list = list(Path(print_img_dir).rglob('*.png'))
    random.shuffle(print_img_path_list)
    # num_files = len(print_img_path_list)
    test_size = 2312

    test_print_img_path_list = print_img_path_list[:test_size]

    for test_print_img_path in tqdm(test_print_img_path_list):
        img_id = test_print_img_path.stem
        test_np_depth_img_path = os.path.join(np_depth_img_dir, f"{img_id}.npy")
        test_depth_img_path = os.path.join(depth_img_dir, f"{img_id}.png")
        test_ply_img_path = os.path.join(ply_img_dir, f"{img_id}.ply")


        dest_print_dir = os.path.join(dest_dir, "print_512")
        Path(dest_print_dir).mkdir(parents=True, exist_ok=True)

        dest_np_depth_dir = os.path.join(dest_dir, "np_depth_512")
        Path(dest_np_depth_dir).mkdir(parents=True, exist_ok=True)

        dest_depth_dir = os.path.join(dest_dir, "depth_512")
        Path(dest_depth_dir).mkdir(parents=True, exist_ok=True)

        dest_ply_dir = os.path.join(dest_dir, "ply_512")
        Path(dest_ply_dir).mkdir(parents=True, exist_ok=True)

       # move files
        shutil.move(test_print_img_path, os.path.join(dest_print_dir, f"{img_id}.png"))
        shutil.move(test_np_depth_img_path, os.path.join(dest_np_depth_dir, f"{img_id}.npy"))
        shutil.move(test_depth_img_path, os.path.join(dest_depth_dir, f"{img_id}.png"))
        shutil.move(test_ply_img_path, os.path.join(dest_ply_dir, f"{img_id}.ply"))

root_dir = "/mnt/hmi/thuong/wb_train_val_test_dataset/test"
depth_img_dir = os.path.join(root_dir, "depth_512")
np_depth_img_dir = os.path.join(root_dir, "np_depth_512")
print_img_dir = os.path.join(root_dir, "print_512")
ply_img_dir = os.path.join(root_dir, "ply_512")
dest_dir = "/mnt/hmi/thuong/wb_train_val_test_dataset/subtest"
split_dataset(root_dir, dest_dir, depth_img_dir, np_depth_img_dir, print_img_dir, ply_img_dir)