#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import open3d as o3d
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    'mesh-file',
    help='file path of mesh file'
)
parser.add_argument(
    'data-path',
    help='directory path in which color, depth and pose files exist'
)
parser.add_argument(
    '-i', '--iteration',
    help='iteration time for optimazation',
    type=int,
    default=300
)

def main(mesh_file, data_path, iteration=300):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    files = os.listdir(data_path)
    color_files = [filename for filename in files
                   if filename[13:18] == 'color']
    depth_files = [filename for filename in files
                   if filename[13:18] == 'depth']
    pose_files = [filename for filename in files
                  if filename[13:17] == 'pose']
    color_files = sorted(
        color_files,
        key = lambda filename: int(filename[6:12])
    )
    depth_files = sorted(
        depth_files,
        key = lambda filename: int(filename[6:12])
    )
    pose_files = sorted(
        pose_files,
        key = lambda filename: int(filename[6:12])
    )
    assert len(color_files) == len(depth_files) == len(pose_files)
    K = np.loadtxt(os.path.join(data_path, 'camera-intrinsics.txt'))
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    
    rgbd_imgs = []
    camera_params = []
    for color_file, depth_file, pose_file \
        in zip(color_files, depth_files, pose_files):
        sys.stdout.write('\rprocessing: {}'.format(color_file[6:12]))
        sys.stdout.flush()
        color_img = o3d.io.read_image(os.path.join(data_path, color_file))
        depth_img = o3d.io.read_image(os.path.join(data_path, depth_file))
        width, height = color_img.get_max_bound()
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_img, convert_rgb_to_intensity=False
        )
        rgbd_imgs.append(rgbd_img)
        camera_param = o3d.camera.PinholeCameraParameters()
        H = np.loadtxt(os.path.join(data_path, pose_file))
        camera_param.extrinsic = np.linalg.inv(H)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            int(width), int(height),
            fx, fy, cx, cy
        )
        camera_param.intrinsic = intrinsic
        camera_params.append(camera_param)
    print()
    camera_trajectory = o3d.camera.PinholeCameraTrajectory()
    camera_trajectory.parameters = camera_params
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    base_path, ext = os.path.splitext(mesh_file)

    option = o3d.color_map.ColorMapOptimizationOption()
    option.maximum_iteration = 0
    o3d.color_map.color_map_optimization(
        mesh, rgbd_imgs, camera_trajectory, option
    )
    o3d.io.write_triangle_mesh(
        base_path + '_0iter' + ext,
        mesh
    )
    o3d.visualization.draw_geometries([mesh])

    option.maximum_iteration = iteration
    option.non_rigid_camera_coordinate = True
    o3d.color_map.color_map_optimization(
        mesh, rgbd_imgs, camera_trajectory, option
    )
    o3d.io.write_triangle_mesh(
        base_path + '_{}iter'.format(iteration) + ext,
        mesh
    )
    o3d.visualization.draw_geometries([mesh])

if __name__ == '__main__':
    args = parser.parse_args()
    main(getattr(args, 'mesh-file'),
         getattr(args, 'data-path'),
         args.iteration)

    
