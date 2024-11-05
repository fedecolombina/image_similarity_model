import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random

class GeometricShapesDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, pairs_per_class=5000): # The dataset provides 10k images per class
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.pairs = []
        self.pairs_per_class = pairs_per_class
        self._create_pairs()

    def _create_pairs(self): # For class balance 
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            label_to_indices.setdefault(label, []).append(idx)
        
        for label in label_to_indices:
            indices = label_to_indices[label]
            n = len(indices)
            if n < 2:
                continue
            for _ in range(self.pairs_per_class):
                i, j = random.sample(indices, 2)
                self.pairs.append((i, j, 1))  # Similar pair
        
        all_labels = list(set(self.labels))
        for _ in range(self.pairs_per_class * len(all_labels)):
            label1, label2 = random.sample(all_labels, 2) # Take two different labels
            idx1 = random.choice([i for i, l in enumerate(self.labels) if l == label1])
            idx2 = random.choice([i for i, l in enumerate(self.labels) if l == label2])
            self.pairs.append((idx1, idx2, 0))  # Dissimilar pair

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        img1 = Image.open(self.image_paths[idx1]).convert('L')
        img2 = Image.open(self.image_paths[idx2]).convert('L')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), label

def getLabelFromFilename(filename):

    shape = os.path.basename(filename).split('_')[0] # Images start with shape (label)
    return shape

def loadData(data_dir, test_size=0.2, batch_size=32, max_samples=None):

    image_paths = glob.glob(os.path.join(data_dir, '*.png'))
    labels = [getLabelFromFilename(path) for path in image_paths]

    label_set = sorted(set(labels)) 
    label_map = {label: idx for idx, label in enumerate(label_set)}
    numeric_labels = [label_map[label] for label in labels]

    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, numeric_labels, test_size=test_size, random_state=42)

    if max_samples is not None:
        train_paths = train_paths[:max_samples]
        train_labels = train_labels[:max_samples]
        test_paths = test_paths[:max_samples]
        test_labels = test_labels[:max_samples]

    #import pdb
    #pdb.set_trace()

    # Transfromations for training dataset (with data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomAffine(degrees=30, scale=(0.8, 1.2)),
        transforms.ToTensor()        ])

    # Transformations for test dataset (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
        ])

    train_dataset = GeometricShapesDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = GeometricShapesDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
