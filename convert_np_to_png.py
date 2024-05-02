from PIL import Image
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm


def convert_np_to_depth(np_path, dest_path):
    # np_path = "/mnt/hmi/thuong/wb_train_val_test_dataset/valid/np_depth_512/03590_mk15_180.npy"
    # print(np_path)
    np_img = np.load(np_path)

    # np_img = np.load(np_path, allow_pickle=True)
    # with open(np_path, "rb") as f:
    #     np_img = np.load(f)
    img = np.uint8(255*np_img).reshape(np_img.shape[0], np_img.shape[1])
    Image.fromarray(img).save(dest_path)
    
    
def run(np_dir, dest_dir):
    np_path_list = list(Path(np_dir).glob("*.npy"))
    for np_path in tqdm(np_path_list):
        c_name = np_path.stem
        dest_path = os.path.join(dest_dir, f"{c_name}.png")
        convert_np_to_depth(str(np_path), dest_path)
        
        
np_dir = "/mnt/hmi/thuong/wb_train_val_test_dataset/test/np_depth_512"
dest_dir = "/mnt/hmi/thuong/wb_train_val_test_dataset/test/depth_512"
Path(dest_dir).mkdir(exist_ok=True, parents=True)
# convert_np_to_depth(np_dir, dest_path="./snv.png")
run(np_dir, dest_dir)