from functools import lru_cache
from os.path import join

import numpy as np
import torch
from ocamcamera import OcamCamera
from scipy.spatial.transform import Rotation as Rot


def load_poses(pose_file):
    """Calculate pose T cam <- world \in SE(3)"""
    Tcw = []
    with open(pose_file) as f:
        data = f.readlines()

    for it in data:
        it = list(map(float, it.split()))
        T = np.eye(4)  # T world <- cam
        angle = it[:3]
        R = Rot.from_rotvec(angle).as_matrix()
        T[:3, :3] = R
        T[:3, 3] = it[3:]
        T[:3, 3] /= 100  # from cm to m
        Tcw.append(np.linalg.inv(T))
    return Tcw


def spherical_grid(h, w):
    """ Generate meshgrid for equirectangular projection.

    Parameters
    ----------
    h : int
        height of the expected equirectangular image
    w : int
        width of the expected equirectangular image

    Returns
    -------
    phi_xy : numpy array
        phi value (-np.pi < phi < np.pi)
    theta_xy : numpy array
        theta value (-np.pi/2 < theta < np.pi/2)
    """
    p = 2 * np.pi / w
    th = np.pi / h
    phi = [-np.pi + (i + 0.5) * p for i in range(w)]
    theta = [-np.pi / 2 + (i + 0.5) * th for i in range(h)]
    phi_xy, theta_xy = np.meshgrid(phi, theta, sparse=False, indexing='xy')
    return phi_xy, theta_xy


@lru_cache(maxsize=None)
def spherical_grid_3Dpoints(h, w):
    phi_xy, theta_xy = spherical_grid(h, w)
    pts = np.stack([np.sin(phi_xy) * np.cos(theta_xy), np.sin(theta_xy), np.cos(phi_xy) * np.cos(theta_xy)], axis=2)
    return pts


@lru_cache(maxsize=None)
def spherical_sweep_grid(ocam, Tcw_bytes, center_bytes, depth, h=320, w=640):
    # numpy array is not hashable so give array as array.tobytes()
    Tcw = np.frombuffer(Tcw_bytes).reshape(4, 4)
    center = np.frombuffer(center_bytes)

    pts = spherical_grid_3Dpoints(h, w)
    # points from rig center
    pts = pts.reshape(-1, 3) * depth
    pts += center

    # add color from camera
    # pt_c = T_cw*pt_w
    pts_c = Tcw[:3, :3].dot(pts.T) + Tcw[:3, 3:4]
    mapx, mapy = ocam.world2cam(pts_c)

    # for grid_sample
    mapx = mapx.reshape(h, w)
    mapy = mapy.reshape(h, w)
    mapx = 2 * mapx / ocam.width - 1
    mapy = 2 * mapy / ocam.height - 1
    grid = torch.from_numpy(np.stack([mapx, mapy], axis=-1))
    return grid


class SphericalSweeping(object):
    def __init__(self, root_dir, h=320, w=640, fov=220):
        """ SphericalSweeping module.

        Parameters
        ----------
        root_dir : str
            root directory includes poses.txt, ocam{i}.txt where i = [1,2,3,4]
        h : int
            output image height
        w : int
            output image width
        fov : float
            field of view of camera in degree
        """
        self.h = h
        self.w = w
        # load poses T cam <- world
        self.poses_cw = load_poses(join(root_dir, 'poses.txt'))
        self._Tcw_bytes = [T.tobytes() for T in self.poses_cw]

        # estimate rig center
        center = []
        for T in self.poses_cw:
            camera_position = -T[:3, :3].T.dot(T[:3, 3])
            center.append(camera_position)
        center = np.array(center).mean(axis=0)
        self._center = center
        self._center_bytes = center.tobytes()

        # load ocam calibration data
        self._ocams = []
        for i in range(1, 5):
            key = f'cam{i}'
            ocam_file = join(root_dir, f'o{key}.txt')
            self._ocams.append(OcamCamera(ocam_file, fov, show_flag=False))

    def get_grid(self, idx, depth):
        """ Get grid for torch.nn.functional.grid_sample.

        Parameters
        ----------
        idx : int
            camera index
        depth : float
            depth for spherical sweeping

        Returns
        -------
        grid : torch array
            grid normalized for grid_sample
        """
        grid = spherical_sweep_grid(self._ocams[idx], self._Tcw_bytes[idx], self._center_bytes, depth, h=self.h,
                                    w=self.w)
        return grid

    def valid_area(self, idx):
        """ Get valid area of fisheye image based on field of view (fov).

        Parameters
        ----------
        idx : int
            camera index

        Returns
        -------
        valid : numpy array
            2D (height x width) array mask. 255:valid, 0:invalid
        """
        valid = self._ocams[idx].valid_area()
        return valid
