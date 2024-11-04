import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from helpers.model import SimilarityCNN

def loadModel(model_path):

    model = SimilarityCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def preprocessImage(image_path):

    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension, shape [1, 1, 200, 200]
    
    return image

def computeSimilarity(model, image1, image2):

    with torch.no_grad():

        output1 = model(image1)
        output2 = model(image2)

        cosine_similarity = F.cosine_similarity(output1, output2).item()
        similarity = ((cosine_similarity + 1) / 2 )*100  # Scale cosine similarity to be between 0 and 1

        return similarity    

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Compute similarity between two images.')
    parser.add_argument('image1', type=str, help='Path to the first image.')
    parser.add_argument('image2', type=str, help='Path to the second image.')
    parser.add_argument('--model-path', type=str, default='models/model.pth', help='Path to the trained model.')

    args = parser.parse_args()

    model = loadModel(args.model_path)
    image1 = preprocessImage(args.image1)
    image2 = preprocessImage(args.image2)
    similarity = computeSimilarity(model, image1, image2)
    print(f'Similarity between images: {similarity:.3f}%')
