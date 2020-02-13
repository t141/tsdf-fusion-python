#!/usr/bin/env python

import argparse
import os
import numpy as np
import cv2
import time
import fusion

parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                    help='data directory in which formed data are contained')
parser.add_argument('--voxel-size', type=float, default=0.02,
                    help='tsdf voxel size. default 0.02')

def main(data_dir, voxel_size):
    # (Optional) sample code to compute 3D bounds (in world coordinates)
    # around convex hull of all camera view frustums in dataset
    print("Estimating voxel volume bounds...")
    files = os.listdir(data_dir)
    color_files = [os.path.join(data_dir, filename)
                 for filename in files if filename[13:18] == 'color']
    depth_files = [os.path.join(data_dir, filename)
                 for filename in files if filename[13:18] == 'depth']
    pose_files = [os.path.join(data_dir, filename)
             for filename in files if filename[13:17] == 'pose']
    assert len(color_files) == len(depth_files) == len(pose_files)
    _, color_ext = os.path.splitext(color_files[0])
    _, depth_ext = os.path.splitext(depth_files[0])
    n_imgs = len(color_files)
    cam_intr = np.loadtxt(os.path.join(data_dir, "camera-intrinsics.txt"),
                          delimiter=' ')
    vol_bnds = np.zeros((3,2))
    
    for i in range(n_imgs):
        # Read depth image and camera pose
        # depth is saved in 16-bit PNG in millimeters
        depth_im = cv2.imread(
            os.path.join(data_dir, "frame-%06d.depth"%(i) + depth_ext), -1
        ).astype(float)/1000.
        # set invalid depth to 0 (specific to 7-scenes dataset)
        depth_im[depth_im == 65.535] = 0
        
        # 4x4 rigid transformation matrix
        cam_pose = np.loadtxt(
            os.path.join(data_dir, "frame-%06d.pose.txt"%(i))) 

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im,cam_intr,cam_pose)
        vol_bnds[:,0] = np.minimum(
            vol_bnds[:,0],np.amin(view_frust_pts,axis=1))
        vol_bnds[:,1] = np.maximum(
            vol_bnds[:,1],np.amax(view_frust_pts,axis=1))
        
    # ---------------------------------------------------------------------

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=voxel_size)

    # Loop through RGB-D images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d"%(i+1,n_imgs))
        
        # Read RGB-D image and camera pose
        color_image = cv2.cvtColor(
            cv2.imread(
                os.path.join(data_dir, "frame-%06d.color"%(i) + color_ext)),
            cv2.COLOR_BGR2RGB
        )
        # depth is saved in 16-bit PNG in millimeters
        depth_im = cv2.imread(
            os.path.join(data_dir, "frame-%06d.depth"%(i) + depth_ext), -1
        ).astype(float)/1000.
        # set invalid depth to 0 (specific to 7-scenes dataset)
        depth_im[depth_im == 65.535] = 0
        
        # 4x4 rigid transformation matrix
        cam_pose = np.loadtxt(
            os.path.join(data_dir, "frame-%06d.pose.txt"%(i)))
        
        # Integrate observation into voxel volume
        # (assume color aligned with depth)
        tsdf_vol.integrate(
            color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)
        
    fps = n_imgs/(time.time()-t0_elapse)
    print("Average FPS: %.2f"%(fps))
        
    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite(os.path.join(data_dir, "mesh.ply"),
                     verts, faces, norms, colors)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.data_dir, args.voxel_size)
