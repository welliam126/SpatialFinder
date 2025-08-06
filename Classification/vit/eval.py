import random
from PIL import Image
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from segmentation_models_pytorch import utils
from torch.utils.data import Dataset as BaseDataset, DataLoader
import cv2
import os
import numpy as np
from skimage.color import rgb2hed,hed2rgb
from skimage.exposure import rescale_intensity
import torchvision.transforms as transforms
import torch
import albumentations as albu
import pandas as pd
import matplotlib.pyplot as plt
from unimlp import UNIMLP
from univit import UNIViT
from torch import nn
from tqdm import tqdm

class MyDataset(BaseDataset):
    def __init__(self, root_dir, mode='train', train_ratio=, augmentation=None, preprocessing=None):
        self.root_dir = root_dir
        self.mode = mode
        self.train_ratio = train_ratio
        self.transform = augmentation
        self.preprocessing = preprocessing

        self.image_paths, self.labels = self._get_image_paths_and_labels()
        self._split_dataset()

    def _get_image_paths_and_labels(self):
        image_paths = []
        labels = []
        class_index = 0

        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    if image_name.endswith('.png'):
                        image_path = os.path.join(folder_path, image_name)
                        image_paths.append(image_path)
                        labels.append(class_index)
                class_index += 1

        return image_paths, labels

    def _split_dataset(self):
        combined = list(zip(self.image_paths, self.labels))
        random.seed(2025)
        random.shuffle(combined)
        self.image_paths, self.labels = zip(*combined)

        split_index = int(len(self.image_paths) * self.train_ratio)
        if self.mode == 'train':
            self.image_paths = self.image_paths[:split_index]
            self.labels = self.labels[:split_index]
        elif self.mode == 'eval':
            self.image_paths = self.image_paths[split_index:]
            self.labels = self.labels[split_index:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = np.array(Image.open(image_path).convert('RGB'))

        if self.transform:
            image = self.transform(image=image)['image']
        if self.preprocessing:
            image = self.preprocessing(image=image)['image']

        return image, label

def get_training_augmentation():
    train_transform = [
        albu.RandomRotate90(p=),
        albu.HorizontalFlip(p=),
        albu.VerticalFlip(p=),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.GaussNoise(p=),
        albu.Perspective(p=),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(0.2, 0.2),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
        albu.HueSaturationValue(p=1),
        albu.Resize(height=224, width=224),
        albu.Normalize(),
        albu.pytorch.ToTensorV2(),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        albu.Resize(height=224, width=224),
        albu.Normalize(),
        albu.pytorch.ToTensorV2(),
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

def main(args):
    log_dir = "path/to/log/directory"
    weight_dir = args.weight_dir
    log_file = os.path.join(log_dir, f"log_{args.model}.csv")

    if not os.path.isfile(log_file):
        log_data = {
            "Epoch": [],
            "IoU": [],
            "Precision": [],
            "dice_loss + bce_loss": [],
            "Accuracy": [],
            "Recall": [],
            "Fscore": [],
        }
        log_df = pd.DataFrame(log_data)
    else:
        log_df = pd.read_csv(log_file)

    data_dir = 'path/to/dataset'

    train_dataset = MyDataset(data_dir, augmentation=get_training_augmentation())
    valid_dataset = MyDataset(data_dir, augmentation=get_validation_augmentation(), mode='eval')
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    if args.model == "UNIMLP":
        model = UNIMLP()
    elif args.model == "UNIViT":
        model = UNIViT()
    else:
        raise RuntimeError("Invalid model specified")
    
    device_ids = list(map(int, args.gpu.split(',')))
    main_device = device_ids[0]

    if not torch.cuda.is_available() or main_device >= torch.cuda.device_count():
        raise RuntimeError(f"Invalid device ID: {main_device}")

    model = model.to(f'cuda:{main_device}')
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    model.load_state_dict(torch.load(weight_dir))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader):
            inputs = inputs.to(f'cuda:{main_device}')
            labels = labels.to(f'cuda:{main_device}')
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    fscore = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {fscore:.4f}")

    evaluation_results = {
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [fscore],
    }
    pd.DataFrame(evaluation_results).to_csv(os.path.join(log_dir, 'evaluation_results.csv'), index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="Comma-separated list of GPU IDs to use.")
    parser.add_argument("--model", type=str, default="UNIMLP")
    parser.add_argument("--batch_size", type=int, default=)
    parser.add_argument("--backbone", type=str, default="")
    parser.add_argument("--weight_dir", type=str, required=True, help="Path to pretrained weights")  

    args = parser.parse_args()
    main(args)