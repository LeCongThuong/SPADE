from IQA_pytorch import LPIPSvgg, DISTS, SSIM
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import point_cloud_utils as pcu
import argparse
import json


def show_img(img):
    plt.imshow(img, cmap='gray')

def read_img(img_path, img_size=256):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (img_size, img_size), interpolation = cv2.INTER_AREA)

def preprocess_file(file_path):
    img = read_img(str(file_path))
    # Convert the image to a PyTorch tensor
    t_img = torch.from_numpy(img).float()
    t_img = t_img.unsqueeze(0).unsqueeze(1).cuda()
    return t_img


def get_score(img_path_1, img_path_2, model):
    t_img_1 = preprocess_file(img_path_1)
    t_img_2 = preprocess_file(img_path_2)
    score = model(t_img_1, t_img_2, as_loss=False)
    return score.item()


def get_metric_score(gt_dir, pred_dir, model):
    gt_path_list = list(Path(gt_dir).glob("*.png"))
    pred_path_list = [os.path.join(pred_dir, gt_path.name) for gt_path in gt_path_list]
    score_list = []
    for index, gt_path in tqdm(enumerate(gt_path_list)):
        pred_path = pred_path_list[index] 
        score = get_score(gt_path, pred_path, model)
        score_list.append(score)
    mean_score = np.mean(score_list)
    return mean_score

def get_l1_l2_mean_score(gt_path, pred_path):
    # Load the ground truth and prediction images
    gt_image = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
    pred_image = np.array(Image.open(pred_path)).astype(np.float32) / 255.0
    
    # Ensure both images have the same dimensions
    if gt_image.shape != pred_image.shape:
        raise ValueError("The ground truth and prediction images have different shapes.")
    
    # Create a mask for pixels in GT that are not equal to 1
    mask = gt_image != np.max(gt_image)
    
    # Apply the mask to both images
    gt_masked = np.where(mask, gt_image, 0)
    pred_masked = np.where(mask, pred_image, 0)
    
    # Compute the mean L1 score (mean Manhattan distance) on the masked areas
    l1_score = np.sum(np.abs(gt_masked - pred_masked)) / np.count_nonzero(mask)
    
    # Compute the mean L2 score (RMSE) on the masked areas
    l2_score = np.sqrt(np.sum((gt_masked - pred_masked) ** 2) / np.count_nonzero(mask))
    
    return l1_score, l2_score

def l1_l2_metric_score(gt_dir, pred_dir):
    gt_path_list = list(Path(gt_dir).glob("*.png"))
    pred_path_list = [os.path.join(pred_dir, gt_path.name) for gt_path in gt_path_list]
    l1_score_list, l2_score_list = [], []
    for index, gt_path in tqdm(enumerate(gt_path_list)):
        pred_path = pred_path_list[index] 
        l1_score, l2_score = get_l1_l2_mean_score(gt_path, pred_path)
        l1_score_list.append(l1_score)
        l2_score_list.append(l2_score)
    l1_mean_score = np.mean(l1_score_list)
    l2_mean_score = np.mean(l2_score_list)
    return l1_mean_score, l2_mean_score


def get_chamfer_score(gt_path, pred_path):
    pcd_gt = pcu.load_mesh_v(gt_path)
    pcd_pred = pcu.load_mesh_v(pred_path)
    chamfer_score = pcu.chamfer_distance(pcd_gt, pcd_pred)
    return chamfer_score
    

def chamfer_score(gt_dir, pred_dir):
    gt_path_list = list(Path(gt_dir).glob("*.ply"))
    pred_path_list = [os.path.join(pred_dir, gt_path.name) for gt_path in gt_path_list]
    chamfer_score_list = []
    for index, gt_path in tqdm(enumerate(gt_path_list)):
        pred_path = pred_path_list[index] 
        chamfer_score = get_chamfer_score(gt_path, pred_path)
        chamfer_score_list.append(chamfer_score)
    chamfer_mean_score = np.mean(chamfer_score_list)
    return chamfer_mean_score


def run(gt_depth_dir, gt_ply_dir, pred_depth_dir, pred_ply_dir):
    # gt_dir = "/mnt/hmi/thuong/wb_train_val_test_dataset/valid/depth_512"
    # pred_dir = "results/depth_512"
    # ply_gt_dir  = "/mnt/hmi/thuong/wb_train_val_test_dataset/valid/ply_512"
    # ply_pred_dir = "results/ply"
    chamfer_score_result = chamfer_score(gt_ply_dir, pred_ply_dir)
    print("Chamfer result: ", chamfer_score_result)
    l1_score, l2_score = l1_l2_metric_score(gt_depth_dir, pred_depth_dir)
    print("L1, L2 score: ", l1_score, l2_score)
    D = SSIM(channels=1).cuda()
    ssim_score = get_metric_score(gt_depth_dir, pred_depth_dir, D)
    print("SSIM score: ", ssim_score)
    D = DISTS().cuda()
    dists_score = get_metric_score(gt_depth_dir, pred_depth_dir, D)
    print("DISTS score: ", dists_score)
    D = LPIPSvgg().cuda()
    lpips_score = get_metric_score(gt_depth_dir, pred_depth_dir, D)
    print("LPIPS score: ", lpips_score)
    return {
        "chamfer_distance": float(chamfer_score_result),
        "l1_score": float(l1_score),
        "l2_score": float(l2_score),
        "ssim_score": float(ssim_score), 
        "dists_score": float(dists_score),
        "lpips_score": float(lpips_score)
    }

def parse_aug():
    parser = argparse.ArgumentParser(prog='Evaluation score')
    parser.add_argument('-gt', '--gt_dir', help='path to gt directory')
    parser.add_argument('-pred', '--pred_dir', help='path to pred directory')
    parser.add_argument('-res', '--res_path', help='save result path')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_aug()
    gt_depth_dir = str(os.path.join(args.gt_dir, "depth_512"))
    gt_ply_dir = str(os.path.join(args.gt_dir, "ply_512"))
    pred_depth_dir = str(os.path.join(args.pred_dir, "depth_512"))
    pred_ply_dir = str(os.path.join(args.pred_dir, "ply_512"))

    res_dict = run(gt_depth_dir, gt_ply_dir, pred_depth_dir, pred_ply_dir)

    with open(args.res_path, "w") as f:
        json.dump(res_dict, f, indent=4)

    
# how to calculate FID score: python -m pytorch_fid "/mnt/hmi/thuong/SPADE/results/depth" "/mnt/hmi/thuong/wb_train_val_test_dataset/valid/depth_512"
