from pathlib import Path
import os
import shutil
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter

def gaussian_blur(image_array, sigma=1):
    return gaussian_filter(image_array, sigma=sigma)

def post_process(gt_dir, pred_dir, dest_dir):
    Path(dest_dir).mkdir(exist_ok=True, parents=True)
    np_dest_dir = os.path.join(dest_dir, "post_np")
    img_dest_dir = os.path.join(dest_dir, "depth_512")
    Path(np_dest_dir).mkdir(exist_ok=True, parents=True)
    Path(img_dest_dir).mkdir(exist_ok=True, parents=True)

    pred_path_list = list(Path(pred_dir).glob("*.npy"))
    for pred_path in tqdm(pred_path_list):
        pred_id = pred_path.name
        gt_path = os.path.join(gt_dir, pred_id)
        np_gt = np.load(str(gt_path))
        np_pred = np.load(str(pred_path))
        np_pred = np_pred.squeeze()
        # print(np_pred.shape, np_gt.shape)
        # np_pred = gaussian_blur(np_pred, sigma=3)
        mask = np_gt != np_gt.max()
        # mask = np.expand_dims(mask, axis=2)
        post_pred = np.where(mask, np_pred, 1)
        np_dest_path = os.path.join(np_dest_dir, pred_id)
        with open(np_dest_path, "wb") as f:
            np.save(f, post_pred)
        img_dest_path = os.path.join(img_dest_dir, f"{pred_path.stem}.png")
        img = np.uint8(255*post_pred).reshape(post_pred.shape[0], post_pred.shape[1])
        Image.fromarray(img).save(img_dest_path)
        

def parse_aug():
    parser = argparse.ArgumentParser(prog='Convert numpy depth to ply 3D object')
    parser.add_argument('-gt', '--gt_dir', type=str, help='gt_dir')
    parser.add_argument('-pred', '--pred_dir', type=str, help='pred dir')
    parser.add_argument('-dest', '--dest_dir', help='dest dir to save output')
    args = parser.parse_args()
    return args

def main():
    args = parse_aug()
    post_process(args.gt_dir, args.pred_dir, args.dest_dir)
    
if __name__ == "__main__":
    main()

