# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import pathlib
import sys
import numpy as np

import torch
import imageio

import util

import nvdiffrast.torch as dr

import cv2 
import math 
import open3d as o3d
import warnings
warnings.filterwarnings('ignore')

#----------------------------------------------------------------------------
# Quaternion math.
#----------------------------------------------------------------------------

# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

# Get a random normalized quaternion.
def q_rnd():
    u, v, w = np.random.uniform(0.0, 1.0, size=[3])
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.asarray([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)], np.float32)

# Get a random quaternion from the octahedral symmetric group S_4.
_r2 = 0.5**0.5
_q_S4 = [[ 1.0, 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0],
         [-0.5, 0.5, 0.5, 0.5], [-0.5,-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5, 0.5],
         [ 0.5, 0.5, 0.5, 0.5], [-0.5, 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5, 0.5], [-0.5,-0.5, 0.5, 0.5],
         [ _r2,-_r2, 0.0, 0.0], [ _r2, _r2, 0.0, 0.0], [ 0.0, 0.0, _r2, _r2], [ 0.0, 0.0,-_r2, _r2],
         [ 0.0, _r2, _r2, 0.0], [ _r2, 0.0, 0.0,-_r2], [ _r2, 0.0, 0.0, _r2], [ 0.0,-_r2, _r2, 0.0],
         [ _r2, 0.0, _r2, 0.0], [ 0.0, _r2, 0.0, _r2], [ _r2, 0.0,-_r2, 0.0], [ 0.0,-_r2, 0.0, _r2]]
def q_rnd_S4():
    return np.asarray(_q_S4[np.random.randint(24)], np.float32)

# Quaternion slerp.
def q_slerp(p, q, t):
    d = np.dot(p, q)
    if d < 0.0:
        q = -q
        d = -d
    if d > 0.999:
        a = p + t * (q-p)
        return a / np.linalg.norm(a)
    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    return s0*p + s1*q

# Quaterion scale (slerp vs. identity quaternion).
def q_scale(q, scl):
    return q_slerp(q_unit(), q, scl)

# Quaternion product.
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# Angular difference between two quaternions in degrees.
def q_angle_deg(p, q):
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

# Quaternion product
def q_mul_torch(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return torch.stack([a, b, c, d])

# Convert quaternion to 4x4 rotation matrix.
def q_to_mtx(q,scales=[1,1,1]):
    r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.tensor([[0], [0], [0]], dtype=torch.float32).cuda()], dim=1) # Pad right column.
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()], dim=0)  # Pad bottom row.
    # print(scales)
    # raise()

    # print(scale.shape)
    # rr = rr * scale
    # rr = torch.matmul(scale,rr)
    # rr = torch.matmul(rr,scale)
    # raise()

    return rr

def scale_to_mtx(scales):
    scale = torch.eye(4).cuda()
    scale[0][0]*=scales[0]
    scale[1][1]*=scales[1]
    scale[2][2]*=scales[2]

    return scale

def translation_to_mtx(trans):
    trans_mtx = torch.eye(4).cuda()
    trans_mtx[0][-1]=trans[0]
    trans_mtx[1][-1]=trans[1]
    trans_mtx[2][-1]=trans[2]

    return trans_mtx


# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def transform_pos_numpy(mtx, pos):
    pos = torch.from_numpy(pos)
    t_mtx = torch.from_numpy(mtx) if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1])], axis=1)
    # print(posw.shape,t_mtx.t().shape)
    return torch.matmul(posw, t_mtx.t()).numpy()

def render(glctx, mtx, pos, pos_idx, col, col_idx, resolution: int):
    # Setup TF graph for reference.
    # print(resolution)
    # print('pos',pos,pos.shape)
    # print('mtx',mtx,mtx.shape)
    pos_clip    = transform_pos(mtx, pos)
    # print('pos_clip',pos_clip,pos_clip.shape)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    # print('pos_idx',pos_idx,pos_idx.shape)
    # print('rast_out',rast_out,rast_out.shape)
    print('rast_out',rast_out,rast_out.shape)

    raise()
    # print('col',col,col.shape)
    # print('col_idx',col_idx,col_idx.shape)

    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    # print('color',color,color.shape)
    # raise()
    return color

#     uv_nonzero = torch.nonzero(rast_out[0,:,:,-1])
#     # print(uv_nonzero.shape)
#     triangle_id = rast_out[0,uv_nonzero[0,0].int(),uv_nonzero[0,1].int(),-1].int()
#     # raise()
#     print('triangle_id',triangle_id)
#     u_value = rast_out[0,uv_nonzero[0,0].int(),uv_nonzero[0,1].int(),0]
#     v_value = rast_out[0,uv_nonzero[0,0].int(),uv_nonzero[0,1].int(),1]

#     print("u_value",u_value)
#     print("v_value",v_value)
#     triangle_idx = pos_idx[triangle_id]
#     print("triangle_idx",triangle_idx)
#     print("pos_in_cam",pos_in_cam.shape)
#     trois_points = torch.cat([pos_in_cam[triangle_idx[0]].unsqueeze(0),
#                               pos_in_cam[triangle_idx[1]].unsqueeze(0),
#                               pos_in_cam[triangle_idx[2]].unsqueeze(0)  
#                             ],dim=0)
#     xyz = u_value * trois_points[0] + v_value * trois_points[1] + (1-u_value-v_value) * trois_points[2]
#     print(trois_points)
#     print(xyz)    


def render(glctx, proj_cam, mtx, pos, pos_idx, col, col_idx, resolution):
    if not type(resolution) == list:
        resolution = [resolution,resolution]
    # Setup TF graph for reference.
    # print(resolution)
    # print('pos',pos,pos.shape)
    # print('mtx',mtx,mtx.shape)
    # print('proj_cam',proj_cam)
    # print('pos','0',torch.min(pos[:,0]),torch.max(pos[:,0]))
    # print('pos','1',torch.min(pos[:,1]),torch.max(pos[:,1]))
    # print('pos','2',torch.min(pos[:,2]),torch.max(pos[:,2]))

    # print('pos_in_cam','0',torch.min(pos_in_cam[:,0]),torch.max(pos_in_cam[:,0]))
    # print('pos_in_cam','1',torch.min(pos_in_cam[:,1]),torch.max(pos_in_cam[:,1]))
    # print('pos_in_cam','2',torch.min(pos_in_cam[:,2]),torch.max(pos_in_cam[:,2]))
    # print('pos_clip','0',torch.min(pos_clip[0,:,0]),torch.max(pos_clip[0,:,0]))
    # print('pos_clip','1',torch.min(pos_clip[0,:,1]),torch.max(pos_clip[0,:,1]))
    # print('pos_clip','2',torch.min(pos_clip[0,:,2]),torch.max(pos_clip[0,:,2]))
    # print('pos_clip',pos_clip,pos_clip.shape)    pos_in_cam    = transform_pos(mtx, pos)
    pos_in_cam = transform_pos(mtx, pos)
    pos_in_cam = pos_in_cam[0,:,:3]
    pos_clip    = transform_pos(proj_cam, pos_in_cam)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=resolution)


    # make a depth buffer
    uv_nonzero = torch.nonzero(rast_out[0,:,:,-1],as_tuple=True)
    triangle_ids =  rast_out[0,uv_nonzero[0],uv_nonzero[1],-1].long() - 1
    # print(torch.unique(triangle_ids))
    # raise()

    u_s = rast_out[0,uv_nonzero[0],uv_nonzero[1],0]
    v_s = rast_out[0,uv_nonzero[0],uv_nonzero[1],1]

    triangle_idx = pos_idx[triangle_ids]
    # print(torch.unique(triangle_idx))
    #Get the 3d points: 
    triangle_idx = triangle_idx.reshape(-1).long()
    troisd_points = pos_in_cam[triangle_idx]#.reshape(triangle_ids.shape[0],3,-1)
    # troisd_points = pos_clip[0,triangle_idx,:3]#.reshape(triangle_ids.shape[0],3,-1)
    # print(torch.unique(troisd_points[:,0]),torch.unique(troisd_points[:,1]),torch.unique(troisd_points[:,2]))
    # raise()


    uv_m1 = (1 - u_s - v_s)
    # print(u_s[0],v_s[0], uv_m1[0])

    all_uvs = torch.cat([
            u_s.unsqueeze(-1),
            v_s.unsqueeze(-1),
            uv_m1.unsqueeze(-1),
        ], dim=1).reshape(-1)
    # print(all_uvs[:3])
    # print('trois',troisd_points.shape,troisd_points[:3])
    # raise()
    step1 = (troisd_points * all_uvs.unsqueeze(1)).reshape(-1,3,3)
    # print('step1',step1.shape,step1[0])
    step2 = step1.sum(dim=1)
    # print('step2',step2.shape,step2[0],torch.min(step2),torch.max(step2))
    # raise()
    depth = torch.ones([rast_out.shape[0],rast_out.shape[1],rast_out.shape[2]]).cuda() 
    depth[0,uv_nonzero[0],uv_nonzero[1]] = step2[:,2]
    # opengl camera
    depth *= -1

    # raise()
    # print('col',col,col.shape)
    # print('col_idx',col_idx,col_idx.shape)

    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    # print('color',color,color.shape)
    # raise()
    return color, depth

def length(v):
    return math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

def normalize(v):
    l = length(v)
    return [v[0]/l, v[1]/l, v[2]/l]

def dot(v0, v1):
    return v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2]

def cross(v0, v1):
    return [
        v0[1]*v1[2]-v1[1]*v0[2],
        v0[2]*v1[0]-v1[2]*v0[0],
        v0[0]*v1[1]-v1[0]*v0[1]]

def lookat(eye, target, up):
    mz = normalize( (eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]) ) # inverse line of sight
    mx = normalize( cross( up, mz ) )
    my = normalize( cross( mz, mx ) )
    tx =  dot( mx, eye )
    ty =  dot( my, eye )
    tz = -dot( mz, eye )   
    return np.array([mx[0], my[0], mz[0], 0, mx[1], my[1], mz[1], 0, mx[2], my[2], mz[2], 0, tx, ty, tz, 1])


def lookat_torch(eye, target, up):
    print(eye)
    print(target)
    mz = normalize( (eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]) ) # inverse line of sight
    mx = normalize( cross( up, mz ) )
    my = normalize( cross( mz, mx ) )
    tx =  dot( mx, eye )
    ty =  dot( my, eye )
    tz = -dot( mz, eye )   
    print(mx)
    return torch.tensor([mx[0], my[0], mz[0], 0, mx[1], my[1], mz[1], 0, mx[2], my[2], mz[2], 0, tx, ty, tz, 1])


#----------------------------------------------------------------------------
# Cube pose fitter.
#----------------------------------------------------------------------------

def fit_pose(max_iter           = 2000,
             repeats            = 10,
             log_interval       = 10,
             display_interval   = None,
             display_res        = 512,
             lr_base            = 0.001,
             lr_falloff         = 1.0,
             nr_base            = 1.0,
             nr_falloff         = 1e-4,
             grad_phase_start   = 0.5,
             resolution         = 256,
             out_dir            = None,
             log_fn             = None,
             mp4save_interval   = None,
             mp4save_fn         = None,
             args = None
             ):

    log_file = None
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(out_dir + '/' + log_fn, 'wt')
        if mp4save_interval != 0:
            writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', mode='I', fps=30, codec='libx264', bitrate='16M')

    else:
        mp4save_interval = None

    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'

    import open3d as o3d



    glctx = dr.RasterizeGLContext()

    # import nvisii

    class CameraTorch(torch.nn.Module):
        def __init__(self, q,trans):
            super().__init__()

            self.qx = torch.nn.Parameter(torch.tensor(q[0]))
            self.qy = torch.nn.Parameter(torch.tensor(q[1]))
            self.qz = torch.nn.Parameter(torch.tensor(q[2]))
            self.qw = torch.nn.Parameter(torch.tensor(q[3]))

            self.x = torch.nn.Parameter(torch.tensor(trans[0]))
            self.y = torch.nn.Parameter(torch.tensor(trans[1]))
            self.z = torch.nn.Parameter(torch.tensor(trans[2]))
            # self.rx = torch.nn.Parameter(torch.tensor(scale[0]))
            # self.ry = torch.nn.Parameter(torch.tensor(scale[1]))
            # self.rz = torch.nn.Parameter(torch.tensor(scale[2]))


        def forward(self):

            return {
                "quat": [self.qx,self.qy,self.qz,self.qw],
                'trans': [self.x,self.y,self.z,],
            }
    # np.random.seed(101012)
    # random.seed(101012)
    import random 
    import pyrr



    for rep in range(repeats):

        pos_idx = None
        pos = None
        col_idx = None
        col = None

        # read the urdf 

        from yourdfpy import URDF
        import yourdfpy
        import random 
        import pyrr

        # urdf_content_path = "/home/jtremblay/code/visii_dr/content/urdf/kinova_description/"
        # robot = URDF.load(f"{urdf_content_path}/urdf/kinova.urdf")

        # /home/jtremblay/code/nvdiffrast/samples/torch/content/kinova_description

        urdf_content_path = "/home/jtremblay/code/nvdiffrast/samples/torch/content/kinova_description/"
        robot = URDF.load(f"{urdf_content_path}/urdf/j2s7s300_standalone.urdf")


        # path_data_2_match = 'data/first_cam_poses_jaco/000/'
        path_data_2_match = f'data/first_cam_poses_jaco/{str(rep).zfill(3)}/'

        frame_img = f"rgb.png"

        # import cv2 
        im = cv2.flip(cv2.imread(path_data_2_match+frame_img),0)[:,:,:3]

        im_gt = torch.tensor(im).cuda().unsqueeze(0).float()/255.0

        im_seg_gt = cv2.flip(cv2.imread(path_data_2_match+'seg.png'),0)[:,:,:3]
        im_seg_gt = torch.tensor(im_seg_gt).cuda().unsqueeze(0).float()

        depth_gt_read = np.flipud(np.load(path_data_2_match+'depth_meters.npy')).copy()
        depth_gt_read = torch.tensor(depth_gt_read).cuda().unsqueeze(0).float()
        print(depth_gt_read.shape,torch.min(depth_gt_read),torch.max(depth_gt_read))

        robot_pose = path_data_2_match + "info.json"

        import json

        with open(f"{robot_pose}", 'r') as f:
            robot_pose = json.load(f)
        print(robot_pose)

        cfg_start = {}

        for i_join, joint_name in enumerate(robot.joint_names):
            print(joint_name)
            if joint_name in robot_pose['robot_state']:
                cfg_start[joint_name] = robot_pose['robot_state'][joint_name]

        robot.update_cfg(cfg_start)
        
        added = -1 

        for link in robot.link_map.keys():    
            link_name = link
            link = robot.link_map[link]
            print(link_name)
            for visual in link.visuals:    
                if not visual.geometry.mesh is None:
                    added += 1
                    path = visual.geometry.mesh.filename.replace("package://kinova_description/",'')
                    data_2_load = os.path.join(urdf_content_path,path)
                    print(" ",data_2_load)

                    # if 'dae' in data_2_load:
                    #     print(data_2_load)
                    #     continue
                    # m = o3d.io.read_triangle_mesh(data_2_load)


                    import trimesh
                    mesh = trimesh.load(data_2_load,force='mesh')
                    print(mesh)

                    pos_3 = np.asarray(mesh.vertices)

                    pos_m = np.ones([pos_3.shape[0],4])
                    pos_m[:,:3] = pos_3
                    
                    # update the vertices 
                    trans = np.array(robot.get_transform(link_name))
                    # pos_m = transform_pos_numpy(pyrr.Matrix44.from_x_rotation(-np.pi/2),
                    #     transform_pos_numpy(trans, pos_m[:,:3])[:,:3])
                    pos_m = transform_pos_numpy(trans, pos_m[:,:3])[:,:3]

                    pos_idx_m = np.asarray(mesh.faces)
                    if not pos_idx is None:

                        print('yeahhhhh')
                        print('pos_idx.shape[0]',pos_idx.shape[0])
                        print('avant',np.min(pos_idx_m),np.max(pos_idx_m))
                        pos_idx_m = pos_idx_m + np.ones(pos_idx_m.shape) * (pos.shape[0])
                        pos_idx_m = pos_idx_m.astype(np.int)
                        print('apres',np.min(pos_idx_m),np.max(pos_idx_m))
                    
                    if pos is None:
                        col_idx_m = np.zeros(pos_idx_m.shape)
                    else:
                        col_idx_m = np.ones(pos_idx_m.shape) * col.shape[0]
                        col_idx_m = col_idx_m.astype(np.int)
                    # col_idx_m = np.zeros(pos_idx_m.shape)

                    # col =  np.ones(pos_3.shape)
                    col_m =  np.random.uniform(0.1,0.9,[1,3])

                    if pos is None:
                        pos = pos_m 
                        pos_idx = pos_idx_m
                        col = col_m
                        col_idx = col_idx_m
                    else:
                        pos = np.concatenate([pos,pos_m], 0)
                        pos_idx = np.concatenate([pos_idx,pos_idx_m], 0)
                        col = np.concatenate([col,col_m], 0)
                        col_idx = np.concatenate([col_idx,col_idx_m], 0)

                    print('added',added)    
                    print("pos_idx_m",pos_idx_m[0],pos_idx_m.shape)
                    print(np.min(pos_idx_m),np.max(pos_idx_m))
                    print("pos_m",pos_m[0],pos_m.shape)
                    print("col_idx_m",col_idx_m[0],col_idx_m.shape)
                    print("col_m",col_m[0],col_m.shape)
                    

                    print('------- ALLLL ------')    
                    print("pos_idx",pos_idx[0],pos_idx.shape)
                    print(np.min(pos_idx),np.max(pos_idx))
                    print("pos",pos[0],pos.shape)
                    print("col_idx",col_idx[0],col_idx.shape)
                    print("col",col[0],col.shape)
                    
            #         break
            # break
            # if added == 1: 
            #     break

        thefile = open('test.obj', 'w')
        for i_pos,position in enumerate(pos):
            thefile.write("v {0} {1} {2}\n".format(position[0],position[1],position[2]))

        # for item in faces:
        for i_pos,item in enumerate(pos_idx):
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0]+1,item[1]+1,item[2]+1))  

        thefile.close()



        col_white = np.ones(col.shape)

        # print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))

        # raise()

        # Some input geometry contains vertex positions in (N, 4) (with v[:,3]==1).  Drop
        # the last column in that case.
        if pos.shape[1] == 4: pos = pos[:, 0:3]

        # Create position/triangle index tensors
        pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
        vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()
        col_idx = torch.from_numpy(col_idx.astype(np.int32)).cuda()
        vtx_col = torch.from_numpy(col.astype(np.float32)).cuda()
        vtx_col_white = torch.from_numpy(col_white.astype(np.float32)).cuda()     

        
        # mtx_cam = lookat(
        #     cam_json['camera_look_at']['eye'],
        #     cam_json['camera_look_at']['at'],
        #     [
        #         0,
        #         0,
        #         1
        #     ]
        # )
        # mtx_cam = torch.tensor(np.array(mtx_cam).reshape((4,4)).T).cuda().float()

        # print(mtx_cam.reshape(4,4))
        # raise()
        # print('pyrr',pyrr.Matrix44.look_at(
        #                 cam_json['camera_look_at']['eye'],
        #     cam_json['camera_look_at']['at'],
        #     [
        #         0,
        #         0,
        #         1
        #     ]
        # ).inverse)
        # print(np.array(cam_json['cam2world']).T)
        # gt_mtx_cam = CameraTorch(
        #     q = pyrr.Quaternion(np.array(cam_json['quaternion_world_xyzw'])).xyzw,
        #     trans = np.array(cam_json['location_world']) 
        #     ).cuda()

        # raise()

        # trans = pyrr.Matrix44(robot_pose["pose_cam_init"]) * pyrr.Matrix44.from_x_rotation(np.pi/2)
        trans = pyrr.Matrix44(robot_pose["pose_cam_init"])
        # print(trans)
        # print(trans.inverse)
        # raise()

        cam_json = {}
        cam_json['location_world'] = [trans[0][-1],trans[1][-1],trans[2][-1]]
        # cam_json['location_world'] = [-trans[2][-1],trans[1][-1],-trans[0][-1]]
        # cam_json['location_world'] = [-trans[2][-1],-trans[1][-1],trans[0][-1]]


        cam_json['quaternion_world_xyzw'] = (pyrr.Matrix33(trans[:3,:3]).quaternion * pyrr.Quaternion.from_x_rotation(np.pi)).xyzw


        cam_json['width'] = 640
        cam_json['height'] = 360
        cam_json['intrinsics'] = {}
        cam_json['intrinsics']['fx'] = 339.2163391113281
        cam_json['intrinsics']['fy'] = 339.2163391113281
        cam_json['intrinsics']['cx'] = 302.383056640625
        cam_json['intrinsics']['cy'] = 181.7169647216797

        trans_gt = torch.tensor(cam_json['location_world']).cuda()
        q_gt = torch.tensor(
                            [
                                cam_json['quaternion_world_xyzw'][0],
                                cam_json['quaternion_world_xyzw'][1],
                                cam_json['quaternion_world_xyzw'][2],
                                cam_json['quaternion_world_xyzw'][3],
                            ]).cuda()

        gt_mtx_cam =  torch.matmul(
                translation_to_mtx(
                    torch.tensor(cam_json['location_world'])).cuda(),
                    torch.nn.functional.normalize(q_to_mtx(
                        torch.tensor(
                            [
                                cam_json['quaternion_world_xyzw'][0],
                                cam_json['quaternion_world_xyzw'][1],
                                cam_json['quaternion_world_xyzw'][2],
                                cam_json['quaternion_world_xyzw'][3],
                            ]).cuda()).float().T)
            ).cuda()
        gt_mtx_cam = torch.inverse(gt_mtx_cam)

        # print(cam_json['location_world'])
        # print()

        # print(gt_mtx_cam)
        # raise()
        # gt_mtx_cam = torch.tensor(np.array(mtx_cam).reshape((4,4)).T).cuda().float()
        
        # print('inverse',gt_mtx_cam)

        best_data = None
        color_best = None
        color_white_best = None
        loss_best = np.inf

        cam_guess = None
        # print(np.eye(4))

        cam_proj = torch.tensor(
            np.matmul(
                util.projection(x=0.4),
                np.eye(4)).astype(np.float32), 
            device='cuda'
        )

        from camera import CameraIntrinsicSettings
        cam_thang = CameraIntrinsicSettings(
            res_width = cam_json['width'], 
            res_height = cam_json['height'],
            fx = cam_json['intrinsics']['fx'], 
            fy = cam_json['intrinsics']['fy'],
            cx = cam_json['intrinsics']['cx'], 
            cy = cam_json['intrinsics']['cy'],
            )
        resolution = [cam_json['height'],cam_json['width']]
        print(cam_proj)
        cam_proj = torch.tensor(cam_thang.get_projection_matrix()).cuda().float()
        print(cam_proj)

        noise_model = torch.distributions.normal.Normal(loc=0, scale=0.3)


        cam_poses_guess = []

        # raise()
        for it in range(max_iter + 1):
            # Set learning rate.
            itf = 1.0 * it / max_iter
            nr = nr_base * nr_falloff**itf


            # Noise input.
            if cam_guess is None:

                error = 0.0001
                degree_error = np.deg2rad(0.001)

                q = pyrr.quaternion.create_from_eulers([
                    random.uniform(-degree_error, degree_error),
                    random.uniform(-degree_error, degree_error),
                    random.uniform(-degree_error, degree_error),
                ])
                q = pyrr.Quaternion(q)
                q_gt = pyrr.Quaternion(cam_json['quaternion_world_xyzw'])
                print(q_gt)
                print(q)
                print(q_gt*q)
                # raise()
                cam_guess = CameraTorch(
                    q = (q_gt*q).xyzw,
                    trans = np.array([
                        cam_json['location_world'][0] + random.uniform(-error, error),
                        cam_json['location_world'][1] + random.uniform(-error, error),
                        cam_json['location_world'][2] + random.uniform(-error, error),
                        # cam_json['location_world'][0],
                        # cam_json['location_world'][1],
                        # cam_json['location_world'][2],
                    ]) 
                    ).cuda()
                optimizer = torch.optim.Adam(cam_guess.parameters(), 
                    betas=(0.9, 0.999), 
                    lr=lr_base
                )
                result = cam_guess()
                print('init cam_pose')
                print(result)

            else:
                result = cam_guess()

            color_white_gt_all = []
            color_gt_all = []
            depth_gt_all = []
            color_white_opt_all = []
            color_white_opt_all = []
            color_opt_all = []
            depth_opt_all = []
            loss_all = None

            # color_white_gt, depth_gt   = render(
            #                     glctx, 
            #                     cam_proj,
            #                     gt_mtx_cam, 
            #                     vtx_pos, 
            #                     pos_idx, 
            #                     vtx_col_white, 
            #                     col_idx, 
            #                     resolution
            #                 )
            color_white_gt = im_seg_gt
            # print(depth_gt.shape)
            # print(depth_gt_read.shape)
            # raise()
            depth_gt = depth_gt_read
            # depth_gt += noise_model.sample(depth_gt.shape).cuda().float()

            color_gt, _   = render(glctx,
                                cam_proj, 
                                gt_mtx_cam, 
                                vtx_pos, 
                                pos_idx, 
                                vtx_col, 
                                col_idx, 
                                resolution
                            )
            color_white_gt_all.append(color_white_gt)
            color_gt_all.append(color_gt)
            depth_gt_all.append(depth_gt)

            # forward
            # print(q_to_mtx(result['quat']).dtype)
            # print(translation_to_mtx(result['trans']).dtype)
            # print(centered_cam.dtype)
            mtx_gu =  torch.matmul(
                    translation_to_mtx(result['trans']),
                    torch.nn.functional.normalize(q_to_mtx(result['quat']).float()).T
                )
            mtx_gu = torch.inverse(mtx_gu)
            # print(mtx_gu)
            # raise()
            # pose_total_opt = q_mul_torch(pose_opt, noise)
            # mtx_total_opt  = torch.matmul(mvp, q_to_mtx(pose_total_opt))
            if rep > 0: 
                mtx_gu = mtx_gu_from_zero
            color_white_opt, depth_opt = render(glctx, cam_proj, mtx_gu, vtx_pos, pos_idx, vtx_col_white, col_idx, resolution)
            color_opt, _     = render(glctx, cam_proj, mtx_gu, vtx_pos, pos_idx, vtx_col, col_idx, resolution)

            color_white_opt_all.append(color_white_opt)
            color_opt_all.append(color_opt)
            depth_opt_all.append(depth_opt)
            # Image-space loss.
            # diff = (color_white_opt - color_white_gt)**2 # L2 norm.
            diff = torch.abs(color_white_opt - color_white_gt) # L2 norm.
            # diff2 = torch.abs(depth_opt[depth_opt>0] - depth_gt[depth_opt>0]) # L2 norm.
            # diff2 = torch.abs(depth_opt - depth_gt) # L2 norm.
            diff2 = torch.abs(depth_opt - depth_gt) # L2 norm.
            # diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
            loss = (torch.mean(diff)*0.5 + torch.mean(diff2)*1)/2
            # loss = torch.mean(diff) 

            # compute ADD
            gt_points = transform_pos(gt_mtx_cam, vtx_pos) 
            gu_points = transform_pos(mtx_gu, vtx_pos)
            


            if loss_all is None:
                loss_all = loss
            else:
                loss_all = (loss_all + loss )/2
            
            # Measure image-space loss and update best found pose.
            loss_val = float(loss_all)

            if (loss_val < loss_best) and (loss_val > 0.0):
                # pose_best = pose_total_opt.detach().clone()
                loss_best = loss_val
                # best_data = result

            # Print/save log.
            if log_interval and (it % log_interval == 0):
                # err = q_angle_deg(pose_opt, pose_target)
                err = 100
                # ebest = q_angle_deg(pose_best, pose_target)
                ebest = 0000
                # s = "rep=%d,iter=%d,err=%f,err_best=%f,loss=%f,loss_best=%f,lr=%f,nr=%f" % (rep, it, err, ebest, loss_val, loss_best, lr, nr)


                # print(f"{it}/{max_iter},'loss',{loss_val},ADD,{add}")
                print(f"{it}/{max_iter},'loss',{loss_val}")
                # if log_file:
                #     log_file.write(s + "\n")

            # Run gradient training step.
            # if itf >= grad_phase_start:

            # TODO
            if rep == 0: 
                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()

            # with torch.no_grad():
            #     pose_opt /= torch.sum(pose_opt**2)**0.5

            # Show/save image.
            display_image = display_interval and (it % display_interval == 0)
            save_mp4      = mp4save_interval and (it % mp4save_interval == 0)

            if display_image or save_mp4:

                add = torch.sqrt(torch.sum((gt_points - gu_points)**2,1))
                add = torch.mean(add)

                gt_points = transform_pos(gt_mtx_cam, torch.tensor([[0,0,0]]).cuda().float()) 
                gu_points = transform_pos(mtx_gu, torch.tensor([[0,0,0]]).cuda().float())
                trans_error = torch.sqrt(torch.sum(( gu_points[0][0][:3] - gt_points[0][0][:3])**2,0))
                # raise()
                print(torch.inverse(gt_mtx_cam))
                gt_pose = torch.tensor([
                            torch.inverse(gt_mtx_cam)[0][-1].item(),
                            torch.inverse(gt_mtx_cam)[1][-1].item(),
                            torch.inverse(gt_mtx_cam)[2][-1].item()
                        ]) 
                gu_pose = torch.tensor([
                            torch.inverse(mtx_gu)[0][-1].item(),
                            torch.inverse(mtx_gu)[1][-1].item(),
                            torch.inverse(mtx_gu)[2][-1].item()
                        ]) 

                trans_error = torch.sqrt(((gt_pose[0]-gu_pose[0])**2)+\
                                         ((gt_pose[1]-gu_pose[1])**2)+\
                                         ((gt_pose[2]-gu_pose[2])**2))
                

                def quaternion_dot(q1, q2):
                    return np.dot(q1, q2)

                def quaternion_angle(q1, q2):
                    dot_product = quaternion_dot(q1, q2)
                    dot_product = np.clip(dot_product, -1, 1)
                    angle = 2 * np.arccos(abs(dot_product))
                    return np.rad2deg(angle)    

                quat_guess = pyrr.Matrix33(
                        torch.inverse(mtx_gu)[:3,:3].cpu().detach().numpy()
                        ).quaternion
                error_rot = quaternion_angle(
                    q_gt,
                    pyrr.Matrix33(
                        torch.inverse(mtx_gu)[:3,:3].cpu().detach().numpy()
                        ).quaternion.xyzw                    
                    )

                cam_poses_guess.append([gu_pose,quat_guess.xyzw])

                def getimg_stack(color_imgs,depth=False):

                    if depth:
                        for i_im in range(len(color_imgs)):

                            color_imgs[i_im] = torch.cat([
                                color_imgs[i_im].unsqueeze(-1),
                                color_imgs[i_im].unsqueeze(-1),
                                color_imgs[i_im].unsqueeze(-1)
                                ],dim=-1)

                            depth_max = 5
                            color_imgs[i_im][color_imgs[i_im]<0] = depth_max
                            
                            color_imgs[i_im]/= depth_max
                            # print(color_imgs[i_im].shape)

                    col_imgs = []
                    for ii in range(1):
                        row_imgs = []
                        for jj in range(1):
                            img_ref  = color_imgs[ii+jj][0].detach().cpu().numpy()
                            row_imgs.append(img_ref)
                        row_all = np.concatenate(row_imgs, axis=1)[::-1]
                        # print(row_all.shape)
                        col_imgs.append(row_all)
                    gt_final = np.concatenate(col_imgs, axis=0)
                    # return cv2.resize(gt_final,(400,400))
                    return gt_final

                img_ref  = getimg_stack([im_gt])
                # img_ref  = getimg_stack(color_gt_all)
                img_opt  = getimg_stack(color_opt_all)
                
                img_white_ref  = getimg_stack(color_white_gt_all)
                img_white_opt  = getimg_stack(color_white_opt_all)
                img_white_opt[:,:,1] = img_white_ref[:,:,1]
                img_gt_depth = getimg_stack(depth_gt_all,depth=True)
                img_opt_depth = getimg_stack(depth_opt_all,depth=True)

                # add text 
                font = cv2.FONT_HERSHEY_SIMPLEX
                # img_opt = cv2.putText(img_opt,
                #     f"ADD: {str(round(add.item(),3)).zfill(2)}", 
                #     (10,30), 
                #     font, 
                #     0.6,
                #     (255,255,255),
                #     1,
                #     2
                #     )
                # img_opt = cv2.putText(img_opt,
                #     f"trans: {str(round(trans_error.item(),3)).zfill(2)}", 
                #     (200,30), 
                #     font, 
                #     0.6,
                #     (255,255,255),
                #     1,
                #     2
                #     )
                # img_opt = cv2.putText(img_opt,
                #     f"rot: {str(round(error_rot,3)).zfill(2)}", 
                #     (10,50), 
                #     font, 
                #     0.6,
                #     (255,255,255),
                #     1,
                #     2
                #     )
                # img_opt = cv2.putText(img_opt,
                #     f"loss: {str(round(loss_val,4 )).zfill(1)}", 
                #     (200,50), 
                #     font, 
                #     0.6,
                #     (255,255,255),
                #     1,
                #     2
                #     )
                
                
                print(img_ref.shape,img_opt.shape)

                result_white_image = np.concatenate([img_white_ref, img_white_opt], axis=1)
                result_depth_image = np.concatenate([img_gt_depth, img_opt_depth], axis=1)
                print(img_white_opt.shape,np.min(img_white_opt),np.max(img_white_opt))

                gray = cv2.cvtColor(getimg_stack(color_white_opt_all), cv2.COLOR_BGR2GRAY)
                gray = gray.astype(np.uint8)
                cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = cnts[0] if len(cnts) == 2 else cnts[1]

                for c in cnts:
                    cv2.drawContours(img_ref, [c], -1, (36, 255, 12), thickness=1,lineType=cv2.LINE_AA)
                result_color_image = np.concatenate([img_ref, img_opt], axis=1)

                result_image = np.concatenate([result_white_image,result_depth_image,result_color_image],axis=0)

                if display_image:

                    # util.display_image(result_image, size=display_res, title='(%d) %d / %d' % (rep, it, max_iter))
                    cv2.imshow("im",result_image)
                    k = cv2.waitKey(33)
                    if k==27:    # Esc key to stop
                        break
                if save_mp4:
                    writer.append_data(np.clip(np.rint(result_image*255.0), 0, 255).astype(np.uint8))
        
                # break
            # break # the for loop iteration 
        if rep == 0: 
            mtx_gu_from_zero = mtx_gu.clone()
        # Draw the cam poses 
        import open3d as o3d
        xyz = []
        for pose in cam_poses_guess:
            xyz.append(pose[0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        
        # add gt and 0,0,0
        tri_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.4)
        cam = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.02,origin=gt_pose.numpy())
        cam.rotate(o3d.geometry.get_rotation_matrix_from_quaternion([q_gt[3],q_gt[0],q_gt[1],q_gt[2]])) 
        cam_f = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.01,origin=pose[0]) 
        cam_f.rotate(o3d.geometry.get_rotation_matrix_from_quaternion([pose[1][3],pose[1][0],pose[1][1],pose[1][2]])) 

        m = o3d.io.read_triangle_mesh('test.obj')

        # o3d.visualization.draw_geometries([pcd,cam_f,cam,tri_origin])
        # o3d.visualization.draw_geometries([pcd,cam_f,cam,m,tri_origin])
        # o3d.visualization.draw_geometries([pcd,cam_f,cam])
        



    # Done.
    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Cube pose fitting example')
    parser.add_argument('--outdir', help='Specify output directory', default='multipleviews')
    parser.add_argument('--display-interval', type=int, default=10)
    parser.add_argument('--mp4save-interval', type=int, default=10)
    parser.add_argument('--max-iter', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--add_noise', action='store_true', help="add noise to the segmentation")
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        out_dir = f'{args.outdir}/pose'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    # Run.
    fit_pose(
        max_iter=args.max_iter,
        repeats=args.repeats,
        log_interval=100,
        display_interval=args.display_interval,
        out_dir=out_dir,
        log_fn='log.txt',
        mp4save_interval=args.mp4save_interval,
        mp4save_fn='cubewhite.mp4',
        args = args,
    )

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
