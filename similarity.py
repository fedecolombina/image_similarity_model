import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from helpers.model import myCNN

def load_model(model_path):
    model = myCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  #add batch dimension, shape [1, 3, 200, 200]S
    return image

def compute_similarity(model, image1, image2):

    with torch.no_grad():
        
        output1 = model(image1)
        output2 = model(image2)
        
        dist = F.pairwise_distance(output1, output2)

        #return distance as a scalar
        return dist.item()

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='Compute similarity between two images.')
    parser.add_argument('image1', type=str, help='Path to the first image.')
    parser.add_argument('image2', type=str, help='Path to the second image.')
    parser.add_argument('--model-path', type=str, default='models/model.pth', help='Path to the trained model.')

    args = parser.parse_args()

    model = load_model(args.model_path)
    image1 = preprocess_image(args.image1)
    image2 = preprocess_image(args.image2)

    similarity = compute_similarity(model, image1, image2)
    print(f'Similarity between images: {similarity:.3f}')
