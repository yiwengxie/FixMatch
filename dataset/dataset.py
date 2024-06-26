import math
from datasets import load_dataset
import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch_cifar(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    valid_dataset = test_dataset

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch_cifar(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)
    
    valid_dataset = test_dataset

    return train_labeled_dataset, train_unlabeled_dataset, valid_dataset, test_dataset


def get_flowers102(args, root):
    # 128*4 256*1
    size = 256
    transform_labeled = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=size,
                              padding=int(size*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    if args.one:
        folder = "/vhome/xieyiweng/flowers-102-FixMatch-one"
    elif args.three:
        folder = "/vhome/xieyiweng/flowers-102-FixMatch-three"
    else:
        folder = "/vhome/xieyiweng/flowers-102-FixMatch"
    base_dataset_labeled = load_dataset("imagefolder", 
                                        data_dir=f"{folder}/flowers-102-FixMatch-labeled",
                                        drop_labels=False)
    base_dataset_unlabeled = load_dataset("imagefolder",
                                        data_dir=f"{folder}/flowers-102-FixMatch-unlabeled",
                                        drop_labels=False)
    base_dataset_generated = load_dataset("imagefolder",
                                        data_dir=f"{folder}/flowers-102-FixMatch-generated",
                                        drop_labels=False)
    base_dataset_train_labeled = base_dataset_labeled['train']
    base_dataset_train_unlabeled = base_dataset_unlabeled['train']
    base_dataset_train_generated = base_dataset_generated['train']
    base_dataset_valid = base_dataset_labeled['validation']
    base_dataset_test = base_dataset_labeled['test']
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset_train['label'])
    train_labeled_dataset = HFDataset(
        root, 
        # train_labeled_idxs,
        transform=transform_labeled,
        dataset=base_dataset_train_labeled)

    train_unlabeled_dataset = HFDataset(
        root,
        # train_unlabeled_idxs,
        transform=TransformFixMatch_flowers(mean=normal_mean, std=normal_std, size=size),
        dataset=base_dataset_train_unlabeled)
    
    train_generated_dataset = HFDataset(
        root, 
        # train_labeled_idxs,
        transform=transform_labeled,
        dataset=base_dataset_train_generated)

    valid_dataset = HFDataset(
        root, 
        transform=transform_val, 
        dataset=base_dataset_valid)

    test_dataset = HFDataset(
        root, 
        transform=transform_val, 
        dataset=base_dataset_test)

    return train_labeled_dataset, train_unlabeled_dataset, train_generated_dataset, valid_dataset, test_dataset


def get_birds200(args, root):
    # 128*4 256*1
    size = 256
    transform_labeled = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=size,
                              padding=int(size*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    folder = "/vhome/xieyiweng/birds-200-FixMatch-test"
    base_dataset_labeled = load_dataset("imagefolder", 
                                        data_dir=f"{folder}/birds-200-FixMatch-labeled",
                                        drop_labels=False)
    base_dataset_unlabeled = load_dataset("imagefolder",
                                        data_dir=f"{folder}/birds-200-FixMatch-unlabeled",
                                        drop_labels=False)
    base_dataset_generated = load_dataset("imagefolder",
                                        data_dir=f"{folder}/birds-200-FixMatch-generated",
                                        drop_labels=False)
    base_dataset_train_labeled = base_dataset_labeled['train']
    base_dataset_train_unlabeled = base_dataset_unlabeled['train']
    base_dataset_train_generated = base_dataset_generated['train']
    base_dataset_valid = base_dataset_labeled['validation']
    base_dataset_test = base_dataset_labeled['test']
    print("len(base_dataset_train_labeled):", len(base_dataset_train_labeled))
    print("len(base_dataset_train_unlabeled):", len(base_dataset_train_unlabeled))
    print("len(base_dataset_train_generated):", len(base_dataset_train_generated))
    print("len(base_dataset_valid):", len(base_dataset_valid))
    print("len(base_dataset_test):", len(base_dataset_test))
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset_train['label'])
    train_labeled_dataset = HFDataset(
        root, 
        # train_labeled_idxs,
        transform=transform_labeled,
        dataset=base_dataset_train_labeled)

    train_unlabeled_dataset = HFDataset(
        root,
        # train_unlabeled_idxs,
        transform=TransformFixMatch_flowers(mean=normal_mean, std=normal_std, size=size),
        dataset=base_dataset_train_unlabeled)
    
    train_generated_dataset = HFDataset(
        root, 
        # train_labeled_idxs,
        transform=transform_labeled,
        dataset=base_dataset_train_generated)
    
    train_generated_dataset = HFDataset(
        root, 
        # train_labeled_idxs,
        transform=transform_labeled,
        dataset=base_dataset_train_generated)

    valid_dataset = HFDataset(
        root, 
        transform=transform_val, 
        dataset=base_dataset_valid)

    test_dataset = HFDataset(
        root, 
        transform=transform_val, 
        dataset=base_dataset_test)

    return train_labeled_dataset, train_unlabeled_dataset, train_generated_dataset, valid_dataset, test_dataset


def get_cars196(args, root):
    # 128*4 256*1
    size = 256
    transform_labeled = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=size,
                              padding=int(size*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((size, size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    folder = "/vhome/xieyiweng/cars-196-FixMatch-test"
    base_dataset_labeled = load_dataset("imagefolder", 
                                        data_dir=f"{folder}/cars-196-FixMatch-labeled",
                                        drop_labels=False)
    base_dataset_unlabeled = load_dataset("imagefolder",
                                        data_dir=f"{folder}/cars-196-FixMatch-unlabeled",
                                        drop_labels=False)
    base_dataset_generated = load_dataset("imagefolder",
                                        data_dir=f"{folder}/cars-196-FixMatch-generated",
                                        drop_labels=False)
    base_dataset_train_labeled = base_dataset_labeled['train']
    base_dataset_train_unlabeled = base_dataset_unlabeled['train']
    base_dataset_train_generated = base_dataset_generated['train']
    base_dataset_valid = base_dataset_labeled['validation']
    base_dataset_test = base_dataset_labeled['test']
    print("len(base_dataset_train_labeled):", len(base_dataset_train_labeled))
    print("len(base_dataset_train_unlabeled):", len(base_dataset_train_unlabeled))
    print("len(base_dataset_train_generated):", len(base_dataset_train_generated))
    print("len(base_dataset_valid):", len(base_dataset_valid))
    print("len(base_dataset_test):", len(base_dataset_test))
    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(args, base_dataset_train['label'])
    train_labeled_dataset = HFDataset(
        root, 
        # train_labeled_idxs,
        transform=transform_labeled,
        dataset=base_dataset_train_labeled)

    train_unlabeled_dataset = HFDataset(
        root,
        # train_unlabeled_idxs,
        transform=TransformFixMatch_flowers(mean=normal_mean, std=normal_std, size=size),
        dataset=base_dataset_train_unlabeled)
    
    train_generated_dataset = HFDataset(
        root, 
        # train_labeled_idxs,
        transform=transform_labeled,
        dataset=base_dataset_train_generated)
    
    train_generated_dataset = HFDataset(
        root, 
        # train_labeled_idxs,
        transform=transform_labeled,
        dataset=base_dataset_train_generated)

    valid_dataset = HFDataset(
        root, 
        transform=transform_val, 
        dataset=base_dataset_valid)

    test_dataset = HFDataset(
        root, 
        transform=transform_val, 
        dataset=base_dataset_test)

    return train_labeled_dataset, train_unlabeled_dataset, train_generated_dataset, valid_dataset, test_dataset



def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch_cifar(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
    
class TransformFixMatch_flowers(object):
    def __init__(self, mean, std, size=512):
        self.weak = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
class HFDataset(datasets.VisionDataset):
    def __init__(self, root, indexs=None, 
                 transform=None, target_transform=None, dataset=None):
        super().__init__(root, 
                         transform=transform,
                         target_transform=target_transform)

        self.data = dataset['image']
        self.targets = dataset['label']

        if indexs is not None:
            self.data = [self.data[i] for i in indexs]
            self.targets = [self.targets[i] for i in indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(img)

        # 灰度图像！！！！！！！！！！！！！！！！！！！！
        # 坑死爹了
        if img.mode == 'L':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
    def __len__(self):
        return len(self.data)


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'flowers102': get_flowers102,
                   'birds200': get_birds200,
                   'cars196': get_cars196}
