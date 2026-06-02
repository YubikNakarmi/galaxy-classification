from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import loguru as logging
import torch
import pandas as pd
import os
from PIL import Image


class GalaxyDataset(Dataset):
    ''' Custom dataset for galaxy classification weith weighted samplers. 
    Assume the mappping and label datasets are already merged'''

    CLASS_NAMES = {                  #default class values
        0: "completely_round_smooth",
        1: "in_between_smooth",
        2: "cigar_shaped_smooth",
        3: "edge_on",
        4: "spiral"}

    mean:float
    std:float

    def __init__(self, img_dir,labels_df, transform=None,class_names=CLASS_NAMES):

        self.img_dir = img_dir
        self.labels =  labels_df
        self.transform = transform
        self.CLASS_NAMES = class_names

    def __len__(self):        
        return len(self.labels)

    def __getitem__(self, idx):

        row = self.labels.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{str(row['asset_id'])}.jpg")# load img based on img name

        if not os.path.exists(img_path):
            logging.error(f"Image not found: {img_path}")
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])  

        if self.transform:
            image = self.transform(image)
        return image, label
    
    def make_weighted_sampler(self)-> WeightedRandomSampler:
        ''' Create a weighted random sampler to address class imbalance.'''

        class_counts = self.labels["label"].value_counts().sort_index()
        class_counts = torch.tensor(class_counts.values, dtype=torch.float)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[self.labels["label"]]

        return WeightedRandomSampler(
            weights     = sample_weights,
            num_samples = len(sample_weights),
            replacement = True
        )
    
    def random_image_sample(self, num_samples=10):
        indices = torch.randperm(len(self.labels))[:num_samples]
        return [self[i] for i in indices]
    


    def find_mean_std(self, sample_size:int = 1000):
        ''' Compute the mean and standard deviation of the dataset for normalization. 
        Args:
            sample_size (int): The number of random samples to use for estimation. Default is 1000.
        Returns:
            mean (torch.Tensor): The mean pixel values for each channel.
            std (torch.Tensor): The standard deviation of pixel values for each channel.
        '''    

        for img,label in self.random_image_sample(sample_size):
                img = img.float() / 255.0

                if 'mean' not in locals():
                    mean = torch.zeros(3)
                    std = torch.zeros(3)
                self.mean += img.mean(dim=[1, 2])
                self.std += img.std(dim=[1, 2])
        self.mean /= sample_size
        self.std /= sample_size

class GalaxyDatasetDirichlet(Dataset):

    def Dirichlet_nll_loss(self, input, target, alpha):
        # Compute the negative log-likelihood loss for Dirichlet distribution
        log_prob = torch.log(input)
        loss = -torch.sum(target * log_prob, dim=1) / torch.sum(alpha)
        return loss.mean()