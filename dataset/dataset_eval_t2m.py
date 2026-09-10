import torch
import copy
from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
from utils.body_model import BodyModel

neutral_bm_path = "dataset/smpl/smplh/neutral/model.npz"
num_betas = 10
bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas)
template = bm()
A = template.Jtr[:, :4].numpy()

def rigid_transform_3D(A_all, B_all):
    assert A_all.shape == B_all.shape
    assert len(A_all.shape) == 3

    R_all = []
    t_all = []
    B2_all = []
    for A, B in zip(A_all, B_all):
        N = A.shape[0]
        mu_A = np.mean(A, axis=0)
        mu_B = np.mean(B, axis=0)

        AA = A - np.tile(mu_A, (N, 1))
        BB = B - np.tile(mu_B, (N, 1))
        H = np.dot(np.transpose(AA), BB)

        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            print("Reflection detected")
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = np.dot(-R, mu_A.T) + mu_B.T

        # Calculate error
        B2 = np.dot(R, A.T) + np.tile(t[:, np.newaxis], (1, N))
        B2 = B2.T
        B2_all.append(B2)
        err = B2 - B
        err = np.multiply(err, err).sum()
        # print("err:", err)

        R_all.append(R)
        t_all.append(t)

    R_all = np.stack(R_all)
    t_all = np.stack(t_all)
    B2_all = np.stack(B2_all)
    return R_all, t_all, B2_all


def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)

def rot_yaw(yaw):
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs, 0, sn], [0, 1, 0], [-sn, 0, cs]])


def interhuman_to_hml3d272(motion):
    root_idx = 0
    # get joint positions
    position_data = motion[:, :22*3].reshape(-1, 22, 3)
    nfrm, njoint, _ = position_data.shape

    # get root rotations
    B = position_data[:, :4]
    root_orient, root_pos, _ = rigid_transform_3D(np.tile(A, (B.shape[0], 1, 1)), B)
    root_orient = root_orient.reshape(nfrm, 1, 3, 3)

    # get smpl rotations
    rotations_6d = motion[:, 22*6:22*6+21*6].reshape(nfrm, 21, 6)
    rotations_matrix = rotation_6d_to_matrix(torch.from_numpy(rotations_6d).float(), interhuman=True).numpy()  # nframe, 21, 3, 3
    rotations_matrix = np.concatenate([root_orient, rotations_matrix], axis=1)

    # put on floor and put root on origin for the first frame
    ori = copy.deepcopy(position_data[0, root_idx]) # first frame root position
    y_min = np.min(position_data[:,:,1])
    ori[1] = y_min
    position_data = position_data - ori
    velocities_root = position_data[1:,root_idx,:] - position_data[:-1,root_idx,:]

    # calculate local position, all frames on xz origin
    position_data[:,:,0] -= position_data[:,0:1,0]
    position_data[:,:,2] -= position_data[:,0:1,2]

    # calculate heading
    global_heading = - np.arctan2(rotations_matrix[:,root_idx,0,2], rotations_matrix[:, root_idx, 2,2])
    global_heading_rot = np.array([rot_yaw(x) for x in global_heading])
    global_heading_diff = global_heading[1:] - global_heading[:-1]
    global_heading_diff_rot = np.array([rot_yaw(x) for x in global_heading_diff])

    # calculate positions no heading
    positions_no_heading = np.matmul(np.repeat(global_heading_rot[:, None,:, :], njoint, axis=1), position_data[...,None]).squeeze(-1)

    # calculate velocity no heading
    velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

    # calculate root velocity_xz_no_heading
    velocities_root_xy_no_heading = np.matmul(global_heading_rot[:-1], velocities_root[:, :, None]).squeeze()[...,[0,2]]

    # calculate rotations no heading
    rotations_matrix[:,0,...] = np.matmul(global_heading_rot, rotations_matrix[:,0,...])

    # concat all
    size_frame = 8+njoint*3+njoint*3+njoint*6
    final_x = np.zeros((nfrm, size_frame))

    # set the first frame of the root rotation to identity
    final_x[0, 2] = 1
    final_x[0, 6] = 1
    final_x[1:,2:8] = matrix_to_rotation_6d(torch.from_numpy(global_heading_diff_rot)).numpy() # take 6D rotation
    final_x[1:,:2] = velocities_root_xy_no_heading
    final_x[:,8:8+3*njoint] = np.reshape(positions_no_heading, (nfrm,-1))
    final_x[1:,8+3*njoint:8+6*njoint] = np.reshape(velocities_no_heading, (nfrm-1,-1))
    final_x[:,8+6*njoint:8+12*njoint] = np.reshape(rotations_matrix[..., :, :2, :], (nfrm,-1)) # take 6D rotation
    return final_x


class Text2MotionDataset(data.Dataset):
    def __init__(self, is_test, max_text_len=20, unit_length=4):
        self.max_length = 20
        self.pointer = 0
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        self.data_root = "dataset/humanml3d_272"
        self.motion_dir = pjoin(self.data_root, "motion_processed")
        self.text_dir = pjoin(self.data_root, "texts")
        self.joints_num = 22
        self.max_motion_length = 300
        fps = 30
        self.meta_dir = "dataset/humanml3d_272/mean_std"
        if is_test:
            split_file = pjoin(self.data_root, "split", "test.txt")
        else:
            split_file = pjoin(self.data_root, "split", "val.txt")

        mean = np.load(pjoin(self.meta_dir, "Mean.npy"))
        std = np.load(pjoin(self.meta_dir, "Std.npy"))

        min_motion_len = 60  # 30 fps

        data_dict = {}
        id_list = []

        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []

        for name in tqdm(id_list):
            motion = np.load(pjoin(self.motion_dir, name + ".npy"))
            if (len(motion)) < min_motion_len or (len(motion) >= self.max_motion_length):
                continue
            motion = interhuman_to_hml3d272(motion)
            text_data = []
            flag = False
            with cs.open(pjoin(self.text_dir, name + ".txt")) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    tokens = line_split[1].split(" ")
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])

                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict["caption"] = caption
                    text_dict["tokens"] = tokens

                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data.append(text_dict)
                    else:
                        n_motion = motion[int(f_tag * fps) : int(to_tag * fps)]
                        if (len(n_motion)) < min_motion_len or (
                            len(n_motion) >= self.max_motion_length
                        ):
                            continue
                        new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                        while new_name in data_dict:
                            new_name = random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" + name
                        data_dict[new_name] = {
                            "motion": n_motion,
                            "length": len(n_motion),
                            "text": [text_dict],
                        }
                        new_name_list.append(new_name)
                        length_list.append(len(n_motion))

            if flag:
                data_dict[name] = {"motion": motion, "length": len(motion), "text": text_data}
                new_name_list.append(name)
                length_list.append(len(motion))

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def forward_transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.name_list[idx]
        data = self.data_dict[name]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        text_data = random.choice(text_list)
        caption = text_data["caption"]

        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]

        # "Motion Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [motion, np.zeros((self.max_motion_length - m_length, motion.shape[1]))], axis=0
            )

        return caption, motion, m_length


def DATALoader(is_test, batch_size, num_workers=64, unit_length=4, drop_last=True):
    val_loader = torch.utils.data.DataLoader(
        Text2MotionDataset(is_test, unit_length=unit_length),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    return val_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
