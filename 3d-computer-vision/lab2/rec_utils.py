import numpy as np
import os
from copy import deepcopy
from scipy.io import loadmat
from tqdm import tqdm
import open3d as o3d

def read_all_mat_files(path):
    mat_files = [f for f in os.listdir(path) if f.endswith('.mat')]
    loaded_files = [loadmat(os.path.join(path, mat_f))[mat_f.split('.')[0]] for mat_f in mat_files]
    return loaded_files

def np_to_o3d(np_pts):
    pcds = []
    for np_pts_i in np_pts:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_pts_i)
        pcds.append(pcd)
    return pcds

def generate_transform(a_deg, b_deg, c_deg, t):
    a, b, c = np.radians([a_deg, b_deg, c_deg])
    r_z = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a), np.cos(a), 0],
        [0, 0, 1]
    ])
    r_y = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    r_x = np.array([
        [1, 0, 0],
        [0, np.cos(c), -np.sin(c)],
        [0, np.sin(c), np.cos(c)]
    ])
    rot = r_z @ r_y @ r_x
    m = np.hstack((rot, t.reshape(-1, 1)))
    m = np.vstack((m, np.array([0, 0, 0, 1])))
    return m

def icp_from_multiple_views(cloud_pts, target_idx, transforms, range_, thresholds):
    target = cloud_pts[target_idx]
    pts = [target]
    for i in tqdm(range_):
        source = cloud_pts[i]
        tr = transforms[i-1]
        init_transform = generate_transform(tr[0], tr[1], tr[2], tr[3])
        reg_p2p, _ = icp(source, target, init_transform, thresholds[i-1])
        source.transform(reg_p2p.transformation)
        pts.append(source)
    return pts

def icp(source, target, init_transform, threshold):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, 
        threshold, init_transform, 
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    ) 
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target,threshold, init_transform)
    return reg_p2p, evaluation

def draw_registration_result(source, target, transformation):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def save_point_cloud(path, pts):
    final_arr = np.asarray(pts[0].points)
    for i in range(1, len(pts)):
        final_arr = np.vstack((np.asarray(pts[i].points), final_arr))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_arr)
    o3d.io.write_point_cloud(f'{path}.ply', pcd)