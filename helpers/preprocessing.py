import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class GeometricShapesDataset(Dataset):

    def __init__(self, image_paths, numeric_labels, transform=None):
        self.image_paths = image_paths
        self.numeric_labels = numeric_labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) 

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.numeric_labels[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, label

def getLabelFromFilename(filename):

    shape = os.path.basename(filename).split('_')[0] # Images start with shape (label)
    return shape

def loadData(data_dir, test_size=0.2, batch_size=32):

    image_paths = glob.glob(os.path.join(data_dir, '*.png'))
    labels = [getLabelFromFilename(path) for path in image_paths]

    label_set = sorted(set(labels)) # Sort to ensure numbers are the same in train and test datasets
    label_map = {label: idx for idx, label in enumerate(label_set)}
    numeric_labels = [label_map[label] for label in labels]

    #import pdb
    #pdb.set_trace()

    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, numeric_labels, test_size=test_size, random_state=42)

    # Transfromations for training dataset (with data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Transformations for test dataset (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = GeometricShapesDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = GeometricShapesDataset(test_paths, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader
