# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2018.12.13

# Update: 2025-08-19 by kksix

import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import pickle

np.bool = np.bool_
np.int = np.int_
np.float = np.float64
np.complex = np.complex128
np.object = np.object_
np.unicode = np.str_
np.str = np.str_
import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

from utils.rotation_conversions import axis_angle_to_matrix, rotation_6d_to_axis_angle


class BodyModel(nn.Module):
    def reg_buf(self, name, value):
        self.register_buffer(name, value, persistent=False)

    def __init__(
        self,
        bm_fname,
        num_betas=10,
        num_dmpls=None,
        dmpl_fname=None,
        num_expressions=80,
        dtype=torch.float32,
    ):
        super(BodyModel, self).__init__()

        """
        :param bm_fname: path to a SMPL model as pkl file
        :param num_betas: number of shape parameters to include.
        :param device: default on gpu
        :param dtype: float precision of the computations
        :return: verts, trans, pose, betas
        """

        self.dtype = dtype

        if ".npz" in bm_fname:
            smpl_dict = np.load(bm_fname, encoding="latin1")
        elif ".pkl" in bm_fname:
            smpl_dict = pickle.load(open(bm_fname, "rb"), encoding="latin1")
        else:
            raise ValueError("bm_fname should be either a .pkl nor .npz file")

        self.num_betas = num_betas
        self.num_dmpls = num_dmpls
        self.num_expressions = num_expressions

        njoints = smpl_dict["posedirs"].shape[2] // 3
        try:
            self.model_type = {
                69: "smpl",
                153: "smplh",
                162: "smplx",
                45: "mano",
            }[njoints]
        except Exception:
            raise ValueError(f"model_type should be in smpl/smplh/smplx/mano. njoints: {njoints}")
        self.njoints = njoints // 3 + 1
        self.use_dmpl = False
        if num_dmpls is not None:
            if dmpl_fname is not None:
                self.use_dmpl = True
            else:
                raise (ValueError("dmpl_fname should be provided when using dmpls!"))

        if self.use_dmpl and self.model_type in ["smplx", "mano"]:
            raise (NotImplementedError("DMPLs only work with SMPL/SMPLH models for now."))

        self.reg_buf("v_template", torch.tensor(smpl_dict["v_template"], dtype=dtype))
        self.reg_buf("f", torch.tensor(smpl_dict["f"].astype(np.int32)))

        num_total_betas = smpl_dict["shapedirs"].shape[-1]
        if num_betas < 1:
            num_betas = num_total_betas
        shapedirs = smpl_dict["shapedirs"][:, :, :num_betas]
        self.reg_buf(
            "shapedirs",
            torch.from_numpy(np.array(shapedirs).astype(np.float32)),
        )

        if self.model_type == "smplx":
            if smpl_dict["shapedirs"].shape[-1] > 300:
                begin_shape_id = 300
            else:
                begin_shape_id = 10
                num_expressions = smpl_dict["shapedirs"].shape[-1] - 10

            exprdirs = smpl_dict["shapedirs"][
                :, :, begin_shape_id : (begin_shape_id + num_expressions)
            ]
            self.reg_buf("exprdirs", torch.tensor(exprdirs, dtype=dtype))
            self.reg_buf(
                "init_expression",
                torch.tensor(np.zeros(num_expressions), dtype=dtype),
            )

        if self.use_dmpl:
            dmpldirs = np.load(dmpl_fname)["eigvec"][:, :, :num_dmpls]
            self.reg_buf("dmpldirs", torch.tensor(dmpldirs, dtype=dtype))

        # try:
        if isinstance(smpl_dict["J_regressor"], np.ndarray):
            self.reg_buf("J_regressor", torch.tensor(smpl_dict["J_regressor"], dtype=dtype))
        else:  # chumpy, smpl
            self.reg_buf(
                "J_regressor",
                torch.tensor(smpl_dict["J_regressor"].toarray(), dtype=dtype),
            )

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        posedirs = smpl_dict["posedirs"]
        self.reg_buf("posedirs", torch.tensor(posedirs, dtype=dtype))

        # indices of parents for each joints
        kintree_table = smpl_dict["kintree_table"][0].astype(np.int32).tolist()
        kintree_table[0] = 0
        self.reg_buf("kintree_table", torch.tensor(kintree_table, dtype=torch.int32))

        # LBS weights
        weights = smpl_dict["weights"]
        self.reg_buf("weights", torch.tensor(weights, dtype=dtype))

        # init_trans
        self.reg_buf("init_trans", torch.zeros((1, 3), dtype=dtype))
        # root_orient
        self.reg_buf("init_pose_root", torch.zeros((1, 1, 3), dtype=dtype))
        # pose_body
        if self.model_type in ["smpl", "smplh", "smplx"]:
            self.reg_buf("init_pose_body", torch.zeros((1, 21, 3), dtype=dtype))
        # pose_hand
        if self.model_type in ["smpl"]:
            self.reg_buf("init_pose_hand", torch.zeros((1, 2 * 1, 3), dtype=dtype))
        elif self.model_type in ["smplh", "smplx"]:
            self.reg_buf("init_pose_hand", torch.zeros((1, 2 * 15, 3), dtype=dtype))
        elif self.model_type in ["mano"]:
            self.reg_buf("init_pose_hand", torch.zeros((1, 15, 3), dtype=dtype))
        # face poses
        if self.model_type == "smplx":
            self.reg_buf("init_pose_jaw", torch.zeros((1, 1, 3), dtype=dtype))
            self.reg_buf("init_pose_eye", torch.zeros((1, 2, 3), dtype=dtype))
        self.reg_buf("init_betas", torch.zeros(num_betas, dtype=dtype))
        if self.use_dmpl:
            self.reg_buf("init_dmpls", torch.zeros(num_dmpls, dtype=dtype))

    def get_full_pose(
        self,
        pose_root=None,
        pose_body=None,
        pose_hand=None,
        pose_jaw=None,
        pose_eye=None,
    ):
        batch_size = 1
        for arg in [
            pose_root,
            pose_body,
            pose_hand,
            pose_jaw,
            pose_eye,
        ]:
            if arg is not None:
                batch_size = arg.shape[0]
                break
        if pose_root is None:
            pose_root = self.init_pose_root.expand(batch_size, -1, -1)
        if pose_root.dim() == 2:  # (N, 3) -> (N, 1, 3)
            pose_root = pose_root.unsqueeze(1)
        if self.model_type in ["smplh", "smpl"]:
            if pose_body is None:
                pose_body = self.init_pose_body.expand(batch_size, -1, -1)
            if pose_hand is None:
                pose_hand = self.init_pose_hand.expand(batch_size, -1, -1)
        elif self.model_type == "smplx":
            if pose_body is None:
                pose_body = self.init_pose_body.expand(batch_size, -1, -1)
            if pose_hand is None:
                pose_hand = self.init_pose_hand.expand(batch_size, -1, -1)
            if pose_jaw is None:
                pose_jaw = self.init_pose_jaw.expand(batch_size, -1, -1)
            if pose_eye is None:
                pose_eye = self.init_pose_eye.expand(batch_size, -1, -1)
        elif self.model_type == "mano":
            if pose_hand is None:
                pose_hand = self.init_pose_hand.expand(batch_size, -1, -1)

        if self.model_type in ["smplh", "smpl"]:
            full_pose = torch.cat([pose_root, pose_body, pose_hand], dim=1)
        elif self.model_type == "smplx":
            full_pose = torch.cat([pose_root, pose_body, pose_jaw, pose_eye, pose_hand], dim=1)
        elif self.model_type == "mano":
            full_pose = torch.cat([pose_root, pose_hand], dim=1)

        return full_pose

    def forward(
        self,
        pose_root=None,
        pose_body=None,
        pose_hand=None,
        pose_jaw=None,
        pose_eye=None,
        full_pose=None,
        betas=None,
        trans=None,
        dmpls=None,
        expression=None,
        return_dict=False,
        pose2rot=True,
        root2zero=True,
    ):
        device = self.v_template.device
        n_frames = 1
        for arg in [
            pose_root,
            pose_body,
            pose_hand,
            pose_jaw,
            pose_eye,
            full_pose,
            trans,
            dmpls,
            expression,
        ]:
            if arg is not None:
                n_frames = arg.shape[0]
                break
        if full_pose is None:
            full_pose = self.get_full_pose(pose_root, pose_body, pose_hand, pose_jaw, pose_eye)

        if trans is None:
            trans = self.init_trans.expand(n_frames, -1)
        if betas is None:
            betas = self.init_betas

        if self.use_dmpl:
            if dmpls is None:
                dmpls = self.init_dmpls
            shape_components = torch.cat([betas.squeeze(), dmpls.squeeze()])
            shapedirs = torch.cat([self.shapedirs, self.dmpldirs], dim=-1)
        elif self.model_type == "smplx":
            if expression is None:
                expression = self.init_expression
            shape_components = torch.cat([betas.squeeze(), expression.squeeze()])
            shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)
        else:
            shape_components = betas
            shapedirs = self.shapedirs

        shape_blend = torch.tensordot(shape_components.unsqueeze(0), shapedirs, dims=([1], [2]))
        v_shaped = shape_blend + self.v_template
        J_shaped = torch.matmul(self.J_regressor, v_shaped)
        if root2zero:
            J_shaped, v_shaped = (
                (J_shaped.clone() - J_shaped[:, :1]).squeeze(),
                (v_shaped.clone() - J_shaped[:, :1]).squeeze(),
            )
        if pose2rot:
            pose_rotmat = axis_angle_to_matrix(full_pose.view(n_frames, self.njoints, 3))
        else:
            pose_rotmat = full_pose.view(n_frames, -1, 3, 3)
        I = torch.eye(3, device=device)
        pose_blend = torch.tensordot(
            (pose_rotmat[:, 1:] - I).flatten(1),
            self.posedirs,
            dims=([1], [2]),
        )
        v_posed = v_shaped + pose_blend

        Js, Vs = [_.expand(n_frames, -1, -1) for _ in [J_shaped, v_posed]]
        joints_local = Js.clone()
        joints_local[:, 1:] = joints_local[:, 1:] - Js[:, self.kintree_table[1:]]

        T_local = torch.zeros(n_frames, self.njoints, 4, 4, device=device)
        T_local[..., :3, :3] = pose_rotmat
        T_local[..., :3, 3] = joints_local
        T_local[..., -1, -1] = 1
        T_global = torch.zeros_like(T_local, device=device)
        T_global[:, 0] = T_local[:, 0]
        for i in range(1, len(self.kintree_table)):
            T_global[:, i] = torch.bmm(T_global[:, self.kintree_table[i]].clone(), T_local[:, i])
        poses_global = T_global[..., :3, :3].clone()
        Jtr = T_global[..., :3, 3].clone()
        T_global[..., -1:] = T_global[..., -1:].clone() - torch.matmul(T_global, F.pad(Js, (0, 1)).unsqueeze(-1))
        T_vertex = torch.tensordot(T_global, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
        verts = torch.matmul(T_vertex, F.pad(Vs, (0, 1), value=1).unsqueeze(-1)).squeeze(-1)[
            ..., :3
        ]

        Jtr = Jtr + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)

        res = {}
        res["v"] = verts
        res["f"] = self.f
        res["Jtr"] = Jtr
        res["full_pose"] = full_pose
        res["full_pose_global"] = poses_global

        if not return_dict:

            class result_meta(object):
                pass

            res_class = result_meta()
            for k, v in res.items():
                res_class.__setattr__(k, v)
            res = res_class

        return res

    def forward_motion_feature(self, motion):
        trans = motion[:, :3].reshape(-1, 3)
        # trans = torch.cat([trans, motion[1:, 22*3:22*3+3]], dim=0)
        # trans = torch.cumsum(trans, dim=0)
        # trans[:, 1] = motion[:, 1]
        pose_root = rotation_6d_to_axis_angle(
            motion[:, 22 * 6 : 22 * 6 + 6].reshape(-1, 6)
        ).reshape(-1, 3)
        pose_body = rotation_6d_to_axis_angle(
            motion[:, 22 * 6 + 6 : 22 * 6 + 6 + 21 * 6].reshape(-1, 21, 6)
        ).reshape(-1, 21, 3)
        return self.forward(pose_root=pose_root, pose_body=pose_body, trans=trans), trans
