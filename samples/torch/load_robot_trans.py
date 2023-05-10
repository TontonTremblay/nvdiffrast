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
    # print('col',col,col.shape)
    # print('col_idx',col_idx,col_idx.shape)

    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    # print('color',color,color.shape)
    # raise()
    return color

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

def fit_pose(max_iter           = 1000,
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


    pos_idx = None
    pos = None
    col_idx = None
    col = None

    # read the urdf 

    from yourdfpy import URDF
    import yourdfpy
    import random 
    import pyrr
    np.random.seed(101012)
    random.seed(101012)
    urdf_content_path = "/home/jtremblay/code/visii_dr/content/urdf/kinova_description/"
    robot = URDF.load(f"{urdf_content_path}/urdf/kinova.urdf")

    v = [0.0, -4.95569171455048, -2.774211677484724, 3.1948753258336526, -2.6901821783304887, -3.0518731415223153, 3.8222728696691832, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    cfg_start = {}
    for i_join, joint_name in enumerate(robot.joint_names):
        j = robot.joint_map[joint_name]
        if not j.limit is None: 
            # print(j.limit)
            cfg_start[joint_name] = random.uniform(j.limit.lower,j.limit.upper)
            # cfg_start[joint_name] = v[i_join]
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
                if 'dae' in data_2_load:
                    continue
                m = o3d.io.read_triangle_mesh(data_2_load)
                pos_3 = np.asarray(m.vertices)
                pos_m = np.ones([pos_3.shape[0],4])
                pos_m[:,:3] = pos_3
                
                # update the vertices 
                trans = np.array(robot.get_transform(link_name))
                pos_m = transform_pos_numpy(pyrr.Matrix44.from_x_rotation(-np.pi/2),
                    transform_pos_numpy(trans, pos_m[:,:3])[:,:3])
                pos_idx_m = np.asarray(m.triangles)
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

    # thefile = open('test.obj', 'w')
    # for i_pos,position in enumerate(pos):
    #     thefile.write("v {0} {1} {2}\n".format(position[0],position[1],position[2]))

    # # for item in faces:
    # for i_pos,item in enumerate(pos_idx):
    #     thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0]+1,item[1]+1,item[2]+1))  

    # thefile.close()



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
    for rep in range(repeats):
        mtx_cam = lookat(
             [
                -1.5330392167554536,
                -0.8017582077755034,
                0.7717230792267187
            ],
            [
                0.02902800217270851,
                -0.07310445606708527,
                -0.06623277068138123
            ],
            [
                0,
                0,
                1
            ]
        )
        mtx_cam = np.array(mtx_cam).reshape((4,4)).T

        mvp_gt = torch.tensor(np.matmul(
            util.projection(x=0.4), mtx_cam).astype(np.float32), device='cuda')


        best_data = None
        color_best = None
        color_white_best = None
        loss_best = np.inf

        cam_guess = None
        # print(np.eye(4))
        centered_cam = torch.tensor(np.matmul(
                util.projection(x=0.4), np.eye(4)).astype(np.float32), device='cuda')

        for it in range(max_iter + 1):
            # Set learning rate.
            itf = 1.0 * it / max_iter
            nr = nr_base * nr_falloff**itf


            # Noise input.
            if cam_guess is None:

                mtx_cam = lookat(
                     [
                        -1.5330392167554536 + random.uniform(-0.05, .05),
                        -0.8017582077755034+ random.uniform(-0.05, .05),
                        0.7717230792267187 + random.uniform(-0.05, .05)
                    ],
                    [
                        0.02902800217270851 + random.uniform(-0.05, .05),
                        -0.07305445606708527 + random.uniform(-0.05, .05),
                        -0.06623277068138053 + random.uniform(-0.05, .05)
                    ],
                    [
                        0,
                        0,
                        1
                    ]
                )
                mtx_cam = mtx_cam.reshape((4,4)).T
                cam_guess = CameraTorch(
                    q = pyrr.Matrix44(mtx_cam.T).quaternion.xyzw,
                    trans = [mtx_cam[0,-1],mtx_cam[1,-1],mtx_cam[2,-1]] 
                    ).cuda()
                optimizer = torch.optim.Adam(cam_guess.parameters(), betas=(0.9, 0.999), lr=lr_base)
                result = cam_guess()
            else:
                result = cam_guess()

            color_white_gt_all = []
            color_gt_all = []
            color_white_opt_all = []
            color_white_opt_all = []
            color_opt_all = []
            loss_all = None

            color_white_gt   = render(glctx, 
                                mvp_gt, 
                                vtx_pos, 
                                pos_idx, 
                                vtx_col_white, 
                                col_idx, 
                                resolution
                            )


            color_gt   = render(glctx, 
                                mvp_gt, 
                                vtx_pos, 
                                pos_idx, 
                                vtx_col, 
                                col_idx, 
                                resolution
                            )
            color_white_gt_all.append(color_white_gt)
            color_gt_all.append(color_gt)

            # forward
            # print(q_to_mtx(result['quat']).dtype)
            # print(translation_to_mtx(result['trans']).dtype)
            # print(centered_cam.dtype)
            mtx_gu = torch.matmul(
                centered_cam, 
                torch.matmul(
                    translation_to_mtx(result['trans']),
                    torch.nn.functional.normalize(q_to_mtx(result['quat']).float())
                )
            )


            # pose_total_opt = q_mul_torch(pose_opt, noise)
            # mtx_total_opt  = torch.matmul(mvp, q_to_mtx(pose_total_opt))
            color_white_opt = render(glctx, mtx_gu, vtx_pos, pos_idx, vtx_col_white, col_idx, resolution)
            color_opt       = render(glctx, mtx_gu, vtx_pos, pos_idx, vtx_col, col_idx, resolution)

            color_white_opt_all.append(color_white_opt)
            color_opt_all.append(color_opt)

            # Image-space loss.
            # diff = (color_white_opt - color_white_gt)**2 # L2 norm.
            diff = torch.abs(color_white_opt - color_white_gt) # L2 norm.
            # diff = torch.tanh(5.0 * torch.max(diff, dim=-1)[0])
            loss = torch.mean(diff)

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
                print(loss_val)
                # if log_file:
                #     log_file.write(s + "\n")

            # Run gradient training step.
            # if itf >= grad_phase_start:
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            # with torch.no_grad():
            #     pose_opt /= torch.sum(pose_opt**2)**0.5

            # Show/save image.
            display_image = display_interval and (it % display_interval == 0)
            save_mp4      = mp4save_interval and (it % mp4save_interval == 0)

            if display_image or save_mp4:
                def getimg_stack(color_imgs):
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
                    return cv2.resize(gt_final,(512,512))

                img_ref  = getimg_stack(color_gt_all)
                img_opt  = getimg_stack(color_opt_all)
                # img_best = getimg_stack(color_best_all)
                
                img_white_ref  = getimg_stack(color_white_gt_all)
                img_white_opt  = getimg_stack(color_white_opt_all)
                # img_white_best = getimg_stack(color_white_best_all)

                # print(img_ref.shape)

                # i

                result_white_image = np.concatenate([img_white_ref, img_white_opt], axis=1)
                result_color_image = np.concatenate([img_ref, img_opt], axis=1)
                result_image = np.concatenate([result_white_image,result_color_image],axis=0)

                if display_image:

                    # util.display_image(result_image, size=display_res, title='(%d) %d / %d' % (rep, it, max_iter))
                    cv2.imshow("im",result_image)
                    k = cv2.waitKey(33)
                    if k==27:    # Esc key to stop
                        break
                if save_mp4:
                    writer.append_data(np.clip(np.rint(result_image*255.0), 0, 255).astype(np.uint8))

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
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=1)
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
