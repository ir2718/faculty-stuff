import numpy as np
import open3d as o3d
from rec_utils import *

def main():

    PATH = './data/pts'
    SAVE_PATH = './data/zad3_imgs'

    np_pts = read_all_mat_files(PATH)
    cloud_pts = np_to_o3d(np_pts)    

    thresholds = [20, 45, 20, 110]
    transforms = [
        (-40, 0, 0, np.array([0, 70, 30])),
        (-20, -10, 0, np.array([-30, -20, 30])),
        (180, 0, -20, np.array([150, 10, 0])),
        (230, -80, -20, np.array([-150, -60, 80])),
    ]

    #i = 3
    #tf = generate_transform(transforms[i][0], transforms[i][1], transforms[i][2], transforms[i][3])
    #tmp = o3d.geometry.PointCloud()
    #tmp.points = o3d.utility.Vector3dVector(np_pts[i+1] @ tf[:-1, :-1] + tf[:-1, -1])
    #o3d.visualization.draw_geometries([cloud_pts[0], tmp])

    pts = icp_from_multiple_views(cloud_pts, 0, transforms, range(1, 5), thresholds)
    o3d.visualization.draw_geometries(pts)
    save_point_cloud(f'{SAVE_PATH}/reconstruction', pts)

main()