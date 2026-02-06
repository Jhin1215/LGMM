import random
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image
from PIL import ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


class MyDataset(Dataset):
    def __init__(self, root_dir, data_set, img_size, label_transform='norm', to_tensor=True):
        self.to_tensor = to_tensor
        self.label_transform = label_transform

        # 路径处理
        file_path = Path(root_dir) / data_set / 'list' / f'{data_set}.txt'
        self.img_name_list = file_path.read_text().splitlines()
        self.pre_img_path_list = ['/'.join([root_dir, data_set, 'A', img_name])
                                  for img_name in self.img_name_list]
        self.last_img_path_list = ['/'.join([root_dir, data_set, 'B', img_name])
                                   for img_name in self.img_name_list]
        self.lable_img_path_list = ['/'.join([root_dir, data_set, 'label', img_name])
                                    for img_name in self.img_name_list]
        # 对图像的变换
        self.augm = CDDataAugmentation(
            img_size=img_size,
            with_random_hflip=True,
            with_random_vflip=True,
            with_scale_random_crop=True,
            with_random_blur=True,
            random_color_tf=True
        ) if data_set == 'train' else \
            CDDataAugmentation(
                img_size=img_size,
                is_evaluation=True
            )

    def __getitem__(self, index):
        pre_imgs = cv2.imread(self.pre_img_path_list[index], cv2.COLOR_BGR2RGB)
        last_imgs = cv2.imread(self.last_img_path_list[index], cv2.COLOR_BGR2RGB)
        # 只读取第一个维度，因为是 label 是黑白图
        # bit 经过网络之后，最终的输出是 C=2, 分别表示变化的概率和没变化的概率，因此需要将元素值映射到[0, 1]
        lable_imgs = cv2.imread(self.lable_img_path_list[index])[:, :, 0]
        # if you are getting error because of dim mismatch ad [:,:,0] at the end
        # Note: label should be grayscale (single channel image)
        if self.label_transform == 'norm':
            lable_imgs = lable_imgs // 255

        [img, img_B], [label] = self.augm.transform([pre_imgs, last_imgs], [lable_imgs], to_tensor=self.to_tensor)

        return {'pre_imgs': img,
                'post_imgs': img_B,
                'labels': label,
                'img_name': self.img_name_list[index]}

    def __len__(self):
        return len(self.pre_img_path_list)


def get_dataloader(args, data_set='train', seed=None):
    # 根据数据集选则具体路径
    if args.dataset_name == 'LEVIR':
        print(1)
        args.root_dir = '/home/fangc/datasets/LEVIR-CD'
        # args.root_dir = '/root/LEVIR-CD/'
    elif args.dataset_name == 'BCDD':
        args.root_dir = '/home/fangc/datasets/BCDD256'
    elif args.dataset_name == 'CDD':
        args.root_dir = '/home/fangc/datasets/CDD'
    elif args.dataset_name == 'SYSU':
        args.root_dir = '/home/fangc/datasets/SYSU-CD'
    elif args.dataset_name == 'GZCD':
        args.root_dir = '/home/fangc/datasets/GZ-CD'
    elif args.dataset_name == 's2looking':
        args.root_dir = '/home/fangc/datasets/s2looking256'
    elif args.dataset_name == 'UAV':
        args.root_dir = '/home/fangc/datasets/UAV-CD256'
    elif args.dataset_name == 'LEVIR-CD1024':
        args.root_dir = '/home/fangc/datasets/LEVIR-CD1024'
    elif args.dataset_name == 'LEVIR-CD512':
        args.root_dir = '/home/fangc/datasets/LEVIR-CD512'
    elif args.dataset_name == 'LEVIR-CD128':
        args.root_dir = '/home/fangc/datasets/LEVIR-CD128'
    elif args.dataset_name == 'debug':
        args.root_dir = './samples'
    dataset = MyDataset(args.root_dir, data_set, args.input_img_size, label_transform='norm', to_tensor=True)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle if data_set == 'train' else False,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )


class CDDataAugmentation:
    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
            random_color_tf=False,
            is_evaluation=False
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur
        self.random_color_tf = random_color_tf
        self.is_evaluation = is_evaluation

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,] (list of images)
        :param labels: [ndarray,] (list of label images)
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        labels = [TF.to_pil_image(img) for img in labels]
        if not self.is_evaluation:
            if len(labels) != 0:
                if labels[0].size != (self.img_size, self.img_size):
                    labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                              for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [45, 90, 120, 180, 270]
            index = random.randint(0, 4)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.8, 1.2), ratio=(1, 1))

            imgs = [TF.resized_crop(img, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC)
                    for img in imgs]

            labels = [TF.resized_crop(img, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                      for img in labels]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if self.random_color_tf:
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
            imgs_tf = []
            for img in imgs:
                tf = transforms.ColorJitter(
                    color_jitter.brightness,
                    color_jitter.contrast,
                    color_jitter.saturation,
                    color_jitter.hue)
                imgs_tf.append(tf(img))
            imgs = imgs_tf

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                      for img in labels]

            imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    for img in imgs]

        return imgs, labels


def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]
    return imgs, labels


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)
