from tqdm import tqdm
import open3d as o3d
from torch.utils.data import DataLoader
from pathlib import PosixPath
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import argparse


def _convert_depth_to_pc(depth_img, inverted_matrix):
    if len(depth_img.shape) == 2:
        width, height= depth_img.shape
    else:
        width, height, _= depth_img.shape
    x_range = np.arange(width)
    y_range = np.arange(height)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    x_mesh_flat = x_mesh.reshape((-1,))
    y_mesh_flat = y_mesh.reshape((-1,))
    z_mesh = depth_img[y_mesh_flat, x_mesh_flat]

    x_mesh_flat = x_mesh_flat.reshape(-1, 1)
    y_mesh_flat = y_mesh_flat.reshape(-1, 1)
    z_mesh = z_mesh.reshape(-1, 1)
    np_ones = np.ones((x_mesh_flat.shape[0], 1))

    homo_points = np.hstack([x_mesh_flat, y_mesh_flat, z_mesh, np_ones])
    pc_points = (inverted_matrix @ homo_points.T).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_points[:, :3])
    return pcd


def convert_gt_depth_to_pc(depth_path, min_bound_z=-0.45, outlier_neighbors=50, outlier_std_ratio=0.6, depth_res=512):
    matrix_2dto3d = np.asarray([[1 / depth_res, 0, 0, -0.5], [0, -1 / depth_res, 0, 0.5], [0, 0, -1, 0.5], [0, 0, 0, 1]])
    if isinstance(depth_path, str) or isinstance(depth_path, PosixPath):
        np_depth_map = np.load(str(depth_path))
    else:
        np_depth_map = depth_path
    print(len(np_depth_map.shape))
    wb_pcd = _convert_depth_to_pc(np_depth_map, matrix_2dto3d)
    np_vertices = np.asarray(wb_pcd.points)

    filtered_np_vertices = np_vertices[np_vertices[:, 2] > min_bound_z]
    wb_pcd.points = o3d.utility.Vector3dVector(filtered_np_vertices)
    return wb_pcd

def convert_pred_depth_to_pc(depth_path, min_bound_z=-0.45, outlier_neighbors=50, outlier_std_ratio=0.6, depth_res=512):
    matrix_2dto3d = np.asarray([[1 / depth_res, 0, 0, -0.5], [0, -1 / depth_res, 0, 0.5], [0, 0, -1, 0.5], [0, 0, 0, 1]])
    if isinstance(depth_path, str) or isinstance(depth_path, PosixPath):
        np_depth_map = np.load(str(depth_path))
    else:
        np_depth_map = depth_path
    print(len(np_depth_map.shape))
    wb_pcd = _convert_depth_to_pc(np_depth_map, matrix_2dto3d)
    np_vertices = np.asarray(wb_pcd.points)

    filtered_np_vertices = np_vertices[np_vertices[:, 2] > min_bound_z]
    wb_pcd.points = o3d.utility.Vector3dVector(filtered_np_vertices)
    cl, ind = wb_pcd.remove_statistical_outlier(nb_neighbors=outlier_neighbors, std_ratio=outlier_std_ratio)
    wb_filtered_pcd = wb_pcd.select_by_index(ind)

    wb_filtered_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    wb_filtered_pcd.estimate_normals()

    return wb_filtered_pcd



def gt_run():
    np_gt_depth_dir = "/mnt/hmi/thuong/wb_train_val_test_dataset/test/np_depth_512"
    dest_gt_ply_dir = "/mnt/hmi/thuong/wb_train_val_test_dataset/test/ply_512"
    Path(dest_gt_ply_dir).mkdir(exist_ok=True, parents=True)
    np_gt_path_list = list(Path(np_gt_depth_dir).glob("*.npy"))
    for np_gt_path in tqdm(np_gt_path_list):
        dest_ply_path = os.path.join(dest_gt_ply_dir, f"{np_gt_path.stem}.ply")
        wb_pcd = convert_gt_depth_to_pc(str(np_gt_path))
        o3d.io.write_point_cloud(dest_ply_path, wb_pcd)
        
def pred_run(np_gt_depth_dir, dest_gt_ply_dir):
    # np_gt_depth_dir = "/mnt/hmi/thuong/SPADE/results/np"
    # dest_gt_ply_dir = "/mnt/hmi/thuong/SPADE/results/ply"
    Path(dest_gt_ply_dir).mkdir(exist_ok=True, parents=True)
    np_gt_path_list = list(Path(np_gt_depth_dir).glob("*.npy"))
    for np_gt_path in tqdm(np_gt_path_list):
        dest_ply_path = os.path.join(dest_gt_ply_dir, f"{np_gt_path.stem}.ply")
        wb_pcd = convert_pred_depth_to_pc(str(np_gt_path))
        o3d.io.write_point_cloud(dest_ply_path, wb_pcd)      


def parse_aug():
    parser = argparse.ArgumentParser(prog='Convert numpy depth to ply 3D object')
    parser.add_argument('-np', '--np_dir', help='path to numpy file')
    parser.add_argument('-ply', '--ply_dir', help='path to ply file')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_aug()
    pred_run(args.np_dir, args.ply_dir)