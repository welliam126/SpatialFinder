import random
from PIL import Image
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset as BaseDataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
import torch
import albumentations as albu
import pandas as pd
from torch import nn
from tqdm import tqdm
import argparse

class CustomDataset(BaseDataset):
    """Custom dataset for image classification"""
    def __init__(self, root_dir, mode='train', augmentation=None, preprocessing=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = augmentation
        self.preprocessing = preprocessing
        self.image_paths, self.labels = self._get_image_paths_and_labels()

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
    """Create training augmentation pipeline"""
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.3),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.Resize(height=224, width=224),
        albu.Normalize(),
        albu.pytorch.ToTensorV2(),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Create validation augmentation pipeline"""
    test_transform = [
        albu.Resize(height=224, width=224),
        albu.Normalize(),
        albu.pytorch.ToTensorV2(),
    ]
    return albu.Compose(test_transform)

def evaluate_model(model, valid_loader, device):
    """Evaluate model performance on validation set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    fscore = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {fscore:.4f}")
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": fscore
    }

def main(args):
    # Load dataset
    train_dataset = CustomDataset(
        args.data_dir,
        augmentation=get_training_augmentation(),
    )
    
    valid_dataset = CustomDataset(
        args.data_dir,
        augmentation=get_validation_augmentation(),
        mode='eval',
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # Initialize model
    if args.model == "UNIMLP":
        model = UNIMLP()
    elif args.model == "UNIViT":
        model = UNIViT()
    else:
        raise ValueError("Invalid model name")
    
    # Load weights
    model.load_state_dict(torch.load(args.weight_path))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Evaluate model
    results = evaluate_model(model, valid_loader, device)
    
    # Save results
    pd.DataFrame([results]).to_csv(args.output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--model", type=str, required=True, choices=["UNIMLP", "UNIViT"], help="Model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--weight_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--output_csv", type=str, default="evaluation_results.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    main(args)