import cv2
import torch
from torchvision import transforms


class Resize(object):
    def __init__(self, image_size, depth_size=None):
        """Resize images and depth

        Parameters
        ----------
        image_size : tuple of int
         (image_width, image_height)

        depth_size : tuple of int or None
         (depth_width, depth_height)
        """
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.depth = 'idepth'
        self.image_size = image_size
        self.depth_size = depth_size

    def __call__(self, sample):
        if self.depth in sample:
            sample[self.depth] = cv2.resize(sample[self.depth], self.depth_size)
        for cam in self.cam_list:
            sample[cam] = cv2.resize(sample[cam], self.image_size)
        return sample


class ToTensor(object):
    def __init__(self):
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
        self.depth = 'idepth'
        self.ToTensor = transforms.ToTensor()

    def __call__(self, sample):
        if self.depth in sample:
            sample[self.depth] = torch.from_numpy(sample[self.depth]).float()
        for cam in self.cam_list:
            sample[cam] = self.ToTensor(sample[cam])
        return sample


class Normalize(object):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std
        self.cam_list = ['cam1', 'cam2', 'cam3', 'cam4']

    def __call__(self, sample):
        for cam in self.cam_list:
            for t, m, s in zip(sample[cam], self.mean, self.std):
                t.sub_(m).div_(s)
        return sample
