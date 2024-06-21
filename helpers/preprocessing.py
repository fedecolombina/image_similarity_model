import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class GeometricShapesDataset(Dataset):
    
    #initialize the dataset
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        #convert labels to numerical format
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.numeric_labels = [self.label_map[label] for label in self.labels]

    def __len__(self):
        return len(self.image_paths) #number of images in the dataset 

    #__getitem__ method to load data
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.numeric_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def extract_label_from_filename(filename):
    shape = os.path.basename(filename).split('_')[0] #images start with shape (label)
    return shape

def load_data(data_dir, test_size=0.2, batch_size=32, subset_size=1):
    image_paths = glob.glob(os.path.join(data_dir, '*.png'))
    labels = [extract_label_from_filename(path) for path in image_paths]
    
    #put subset_size !=1 for test and debugging 
    subset_size = int(len(image_paths) * subset_size)
    image_paths = image_paths[:subset_size]
    labels = labels[:subset_size]
    
    #import pdb
    #pdb.set_trace()

    train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=test_size, random_state=42)

    #resize PIL images and convert to tensors 
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])

    train_dataset = GeometricShapesDataset(train_paths, train_labels, transform=transform)
    test_dataset = GeometricShapesDataset(test_paths, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader