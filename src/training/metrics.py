from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
import torch
from torchvision import transforms 
import random
from torchvision.transforms import functional as tf
class RandomCenterCrop:
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        size = random.randint(self.min_size, self.max_size)
        return tf.center_crop(img, size)

def make_metrics(num_classes:int, device: torch.device) -> dict:
    return {
        "accuracy":           Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "f1_micro":           F1Score(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "f1_macro":           F1Score(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "f1_classwise":       F1Score(task="multiclass", average="none", num_classes=num_classes).to(device),
        "precision_micro":    Precision(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "precision_macro":    Precision(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "precision_classwise":Precision(task="multiclass", average="none", num_classes=num_classes).to(device),
        "recall_micro":       Recall(task="multiclass", average="micro", num_classes=num_classes).to(device),
        "recall_macro":       Recall(task="multiclass", average="macro", num_classes=num_classes).to(device),
        "recall_classwise":   Recall(task="multiclass", average="none", num_classes=num_classes).to(device),
        "roc_auc":            AUROC(task="multiclass", num_classes=num_classes).to(device),
    }
# Hardcoded as there will be less editing of these transforms, but can be moved to config if needed
raw_transforms = transforms.Compose([
    RandomCenterCrop(170, 240), 
    transforms.Resize(160),                            
    transforms.RandomCrop(128),                        
    transforms.RandomRotation(degrees = [0, 360]),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1376, 0.1276, 0.1187],
                         std= [0.1728, 0.1586, 0.1488])    
])

val_transforms = transforms.Compose([
    transforms.CenterCrop(207),  
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1376, 0.1276, 0.1187],
                         std= [0.1728, 0.1586, 0.1488])    
])
