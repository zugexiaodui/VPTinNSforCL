import os
from os.path import join, exists, realpath, dirname
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from typing import Union, Callable
from copy import deepcopy
from functools import partial
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class ImagePathDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.class_imgpath_dict: dict[int, str] = {}
        self.class_int_str_map: dict[int, str] = {}
        self.split_path = ""

    def __getitem__(self, index):
        return self.class_imgpath_dict[index]

    def __len__(self):
        return len(self.class_imgpath_dict)

    @property
    def class_list(self):
        return sorted(list(self.class_imgpath_dict.keys()))

    @property
    def num_classes(self):
        return len(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: image_dir=\"{self.split_path}\", full_dataset_classes={len(self)}"


class CIFAR100Path(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        name_map = {}
        if exists(_rf := join(f"{dirname(dirname(realpath(__file__)))}/tools/cifar100_classnames.txt")):
            with open(_rf, 'r') as f:
                for _i, line in enumerate(f.readlines()):
                    _dn = line.strip('\n')
                    name_map[_i] = _dn
        else:
            raise FileExistsError(f"{_rf}")
        assert len(name_map) == 100

        for cls_dir in cls_dir_list:
            cls_int = int(cls_dir)
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = name_map[cls_int]

        assert len(self.class_imgpath_dict) == 100


class ImageNetRPath(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        name_map = {}
        if exists(_rf := join(f"{dirname(dirname(realpath(__file__)))}/tools/imagenet_r_classnames.txt")):
            with open(_rf, 'r') as f:
                for line in f.readlines():
                    _sn = line.split(' ')[0]
                    _dn = line.strip('\n')[len(_sn) + 1:]
                    name_map[_sn] = _dn
        else:
            raise FileExistsError(f"{_rf}")
        assert len(name_map) == 200

        for cls_int, cls_dir in enumerate(cls_dir_list):
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = name_map[cls_dir]

        assert len(self.class_imgpath_dict) == 200


class SDomainNetPath(ImagePathDataset):
    def __init__(self, root_dir: str, train: bool):
        super().__init__()
        cls_dir_list = sorted(os.listdir(split_path := join(root_dir, 'train' if train else 'val')))
        assert exists(split_path), split_path
        self.split_path = split_path

        for cls_int, cls_dir in enumerate(cls_dir_list):
            assert not cls_int in self.class_imgpath_dict
            cls_path = realpath(join(split_path, cls_dir))
            self.class_imgpath_dict[cls_int] = [join(cls_path, img_file) for img_file in os.listdir(cls_path)]
            self.class_int_str_map[cls_int] = cls_int

        assert len(self.class_imgpath_dict) == 200


class ImagePathDatasetClassManager():
    def __init__(self, **kwargs):
        self.dataset_dict = {
            'cifar100': partial(CIFAR100Path, root_dir="../datasets/data.CIFAR100" if not (v := kwargs.get('cifar100')) else v),
            'imagenet_r': partial(ImageNetRPath, root_dir="../datasets/data.ImageNet-R" if not (v := kwargs.get('imagenet_r')) else v),
            'sdomainet': partial(SDomainNetPath, root_dir="../datasets/data.DomainNet" if not (v := kwargs.get('sdomainet')) else v),
        }

    def __getitem__(self, dataset: str) -> ImageNetRPath | CIFAR100Path:
        dataset = dataset.lower()
        if dataset not in (_valid_names := self.dataset_dict.keys()):
            raise NameError(f"{dataset} is not in {_valid_names}")
        return self.dataset_dict[dataset]


class ClassIncremantalDataset(Dataset):
    def __init__(self, path_dataset: ImagePathDataset, task_class_list: list[int], transforms: T.Compose = None, target_transforms: Callable = None, expand_times: int = 1, return_index: bool = False, sample_type='path'):
        super().__init__()
        self.path_dataset = path_dataset
        self.task_class_list = tuple(deepcopy(task_class_list))
        assert isinstance(expand_times, int) and expand_times >= 1
        self.expand_times = expand_times
        self.transforms = transforms
        self.target_transforms = target_transforms
        assert sample_type in ('path', 'image')
        self.sample_type = sample_type

        self.samples, self.labels = self.get_all_samples(sample_type=self.sample_type)
        self.return_index = return_index
        self.num_samples = len(self.labels)

        self.cache_dict = {}

    def get_all_samples(self, sample_type: str = 'path') -> tuple[list[Image.Image | str], list[int]]:
        assert sample_type in ('path', 'image'), f"{sample_type}"
        smp_list = []
        lbl_list = []
        for cls_int in self.task_class_list:
            assert cls_int in self.path_dataset.class_list
            assert len(self.path_dataset[cls_int]) > 0
            for img_path in sorted(self.path_dataset[cls_int]):
                if sample_type == 'image':
                    sample: Image.Image = Image.open(img_path).convert('RGB')
                elif sample_type == 'path':
                    sample: str = img_path
                smp_list.append(sample)

                label = cls_int
                if self.target_transforms is not None:
                    label = self.target_transforms(label)
                lbl_list.append(label)

        return smp_list, lbl_list

    def read_one_image_label(self, index: int) -> tuple[Image.Image, int]:
        if self.sample_type == 'path':
            if self.expand_times == 1:
                img = Image.open(self.samples[index]).convert('RGB')
            elif self.expand_times > 1:
                img = Image.open(self.samples[index]).convert('RGB')
            else:
                raise ValueError(f"{self.expand_times}")

            lbl = self.labels[index]
            return img, lbl
        elif self.sample_type == 'image':
            return self.samples[index], self.labels[index]
        else:
            raise NameError(f"{self.sample_type}")

    def __getitem__(self, index: int) -> tuple[Union[Image.Image, Tensor], int]:
        index %= self.num_samples
        img, lbl = self.read_one_image_label(index)

        if self.transforms is not None:
            if self.return_index:
                return self.transforms(img), lbl, index
            else:
                return self.transforms(img), lbl
        if self.return_index:
            return img, lbl, index
        else:
            return img, lbl

    def __len__(self):
        return self.num_samples * self.expand_times

    def __repr__(self) -> str:
        _repr = f"{self.__class__.__name__} for {self.path_dataset.__repr__()}: task_class_list({len(self.task_class_list)})={self.task_class_list}, num_samples={len(self.samples)}, expand_times={self.expand_times}"
        return _repr


class Mixup:
    """ Mixup/Cutmix that applies different params to each element or whole batch
        REF: https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/mixup.py
    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def cutmix_bbox_and_lam(self, img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
        """ Generate bbox and apply lambda correction.
        """
        if ratio_minmax is not None:
            yl, yu, xl, xu = self.rand_bbox_minmax(img_shape, ratio_minmax, count=count)
        else:
            yl, yu, xl, xu = self.rand_bbox(img_shape, lam, count=count)
        if correct_lam or ratio_minmax is not None:
            bbox_area = (yu - yl) * (xu - xl)
            lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
        return (yl, yu, xl, xu), lam

    def mixup_target(self, target, num_classes, lam=1., smoothing=0.0):
        off_value = smoothing / num_classes
        on_value = 1. - smoothing + off_value
        y1 = self.one_hot(target, num_classes, on_value=on_value, off_value=off_value)
        y2 = self.one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value)
        return y1 * lam + y2 * (1. - lam)

    @staticmethod
    def rand_bbox(img_shape, lam, margin=0., count=None):
        """ Standard CutMix bounding-box
        Generates a random square bbox based on lambda value. This impl includes
        support for enforcing a border margin as percent of bbox dimensions.

        Args:
            img_shape (tuple): Image shape as tuple
            lam (float): Cutmix lambda value
            margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
            count (int): Number of bbox to generate
        """
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape[-2:]
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
        cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
        cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yh = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xh = np.clip(cx + cut_w // 2, 0, img_w)
        return yl, yh, xl, xh

    @staticmethod
    def rand_bbox_minmax(img_shape, minmax, count=None):
        """ Min-Max CutMix bounding-box
        Inspired by Darknet cutmix impl, generates a random rectangular bbox
        based on min/max percent values applied to each dimension of the input image.

        Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

        Args:
            img_shape (tuple): Image shape as tuple
            minmax (tuple or list): Min and max bbox ratios (as percent of image size)
            count (int): Number of bbox to generate
        """
        assert len(minmax) == 2
        img_h, img_w = img_shape[-2:]
        cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
        cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
        yl = np.random.randint(0, img_h - cut_h, size=count)
        xl = np.random.randint(0, img_w - cut_w, size=count)
        yu = yl + cut_h
        xu = xl + cut_w
        return yl, yu, xl, xu

    @staticmethod
    def one_hot(x, num_classes, on_value=1., off_value=0.):
        x = x.long().view(-1, 1)
        return torch.full((x.size()[0], num_classes), off_value, device=x.device).scatter_(1, x, on_value)

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = self.cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = self.cutmix_bbox_and_lam(
                        x[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = self.cutmix_bbox_and_lam(
                x.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target=None):
        assert len(x) % 2 == 0, f'Batch size ({x.shape}) should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        if target is not None:
            target = self.mixup_target(target, self.num_classes, lam, self.label_smoothing)
            return x, target
        return x


def define_dataset(GVM, task_classes: list[int], training: bool, transform_type: str = 'timm', target_map_to_local: bool = True,
                   use_eval_transform: bool = False, expand_times: int = 1, **kwargs) -> ClassIncremantalDataset:
    _current_dataset = GVM.args.dataset
    match _current_dataset:
        case 'imagenet_r' | 'sdomainet':
            interp_mode = 'bilinear'
        case 'cifar100':
            interp_mode = 'bicubic'
    match interp_mode:
        case 'bilinear':
            interp_mode = T.InterpolationMode.BILINEAR
        case 'bicubic':
            interp_mode = T.InterpolationMode.BICUBIC
        case _:
            raise ValueError(interp_mode)
    bilinear = T.InterpolationMode.BILINEAR

    match transform_type:
        case 'timm':
            transforms = create_transform(**resolve_data_config(GVM.cache_dict['pretrained_cfg']), is_training=training if not use_eval_transform else False)
        case 'autoaug':
            dmean: tuple[float] = GVM.cache_dict['pretrained_cfg']['mean']
            dstd: tuple[float] = GVM.cache_dict['pretrained_cfg']['std']
            if training and not use_eval_transform:
                match _current_dataset:
                    case 'cifar100':
                        transforms = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.CIFAR10, bilinear), T.RandomResizedCrop((224, 224), interpolation=interp_mode, antialias=True), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case 'imagenet_r' | 'sdomainet':
                        transforms = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, bilinear), T.RandomResizedCrop((224, 224), interpolation=interp_mode, antialias=True), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case _:
                        raise NotImplementedError(_current_dataset)
            else:
                match _current_dataset:
                    case 'cifar100':
                        transforms = T.Compose([T.Resize((224, 224), antialias=True, interpolation=interp_mode), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case 'imagenet_r' | 'sdomainet':
                        transforms = T.Compose([T.Resize((256, 256), antialias=True, interpolation=interp_mode), T.CenterCrop(224), T.ToTensor(), T.Normalize(dmean, dstd)])
                    case _:
                        raise NotImplementedError(_current_dataset)
        case 'prototype':
            assert not training or use_eval_transform, "Only used for extracting prototypes"
            match _current_dataset:
                case 'cifar100':
                    transforms = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True)])
                case 'imagenet_r' | 'sdomainet':
                    transforms = T.Compose([T.Resize((256, 256), antialias=True), T.CenterCrop((224, 224)), T.ToTensor()])
                case _:
                    raise NotImplementedError(_current_dataset)
        case 'clip':
            _preprocess = GVM.cache_dict['clip_preprocess']
            if training and not use_eval_transform:
                match _current_dataset:
                    case 'cifar100':
                        transforms = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.CIFAR10, bilinear), T.RandomResizedCrop((224, 224), interpolation=interp_mode, antialias=True), _preprocess])
                    case 'imagenet_r' | 'sdomainet':
                        transforms = T.Compose([T.AutoAugment(T.AutoAugmentPolicy.IMAGENET, bilinear), T.RandomResizedCrop((224, 224), interpolation=interp_mode, antialias=True), _preprocess])
                    case _:
                        raise NotImplementedError(_current_dataset)
            else:
                match _current_dataset:
                    case 'cifar100' | 'imagenet_r' | 'sdomainet':
                        transforms = _preprocess
                    case _:
                        raise NotImplementedError(_current_dataset)
        case _:
            raise NotImplementedError(f"{transform_type}")

    class TargetTransform():
        def __init__(self, label_map_g2l: dict[int, tuple[int, int, int]], target_map_to_local: bool) -> None:
            self.label_map_g2l = deepcopy(label_map_g2l)  # {original_label: (taskid, local_label, global_label)}
            self.target_map_to_local = target_map_to_local

        def __call__(self, target: int):
            if self.target_map_to_local:
                return self.label_map_g2l[target][1]
            else:
                return self.label_map_g2l[target][2]

        def __repr__(self) -> str:
            label_map = {k: v[1] if self.target_map_to_local else v[2] for k, v in self.label_map_g2l.items()}
            _repr = str(label_map)
            return _repr

    target_transforms = TargetTransform(GVM.label_map_g2l, target_map_to_local)

    _mode = 'train' if training else 'eval'
    dataset = ClassIncremantalDataset(GVM.path_data_dict[_mode], task_classes, transforms, target_transforms, expand_times=expand_times, return_index=False, sample_type=GVM.args.sample_type)

    return dataset
