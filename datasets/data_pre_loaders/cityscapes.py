import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from loguru import logger

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class CityscapesDataset(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None, augment=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.augment = augment

        self.images_dir = os.path.join(root, "leftImg8bit", split)
        self.labels_dir = os.path.join(root, "gtFine", split)

        self.images = []
        self.labels = []

        for city in sorted(os.listdir(self.images_dir)):
            img_dir = os.path.join(self.images_dir, city)
            label_dir = os.path.join(self.labels_dir, city)

            for file_name in sorted(os.listdir(img_dir)):
                if file_name.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(img_dir, file_name)
                    label_file = file_name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    label_path = os.path.join(label_dir, label_file)

                    if os.path.exists(label_path):
                        self.images.append(img_path)
                        self.labels.append(label_path)

        # Define paired Albumentations augmentations
        if self.augment:
            logger.info("Applying augmentations")
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.RandomGamma(p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2)
            ], additional_targets={'mask': 'mask'})
        else:
            self.aug = None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.labels[idx]

        # Keep image and label as PIL initially
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        if self.aug:
            # Convert to NumPy only for Albumentations
            image_np = np.array(image)
            label_np = np.array(label)

            augmented = self.aug(image=image_np, mask=label_np)
            # Convert back to PIL for torchvision.transforms
            image = Image.fromarray(augmented['image'])
            label = Image.fromarray(augmented['mask'])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


def get_dataset(root, split='train', resize_size=(256, 256), augment=False):
    logger.info(f"Loading Cityscapes dataset from {root} for {split} split")

    image_transform = T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    label_transform = T.Compose([
        T.Resize(resize_size, interpolation=InterpolationMode.NEAREST),
        T.PILToTensor(),
        T.Lambda(lambda x: x.squeeze(0).long())
    ])

    return CityscapesDataset(root=root, split=split,
                             transform=image_transform,
                             target_transform=label_transform,
                             augment=augment if split == 'train' else False)
