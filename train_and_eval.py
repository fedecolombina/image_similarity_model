import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from helpers.preprocessing import loadData
from helpers.model import SimilarityCNN
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import StepLR
from collections import Counter

# Define contrastive loss for learning similar or dissimilar pairs
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, weight_sim=1.0, weight_dissim=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.weight_sim = weight_sim # Define weights for class balance 
        self.weight_dissim = weight_dissim

    def forward(self, output1, output2, label):

        dist = F.pairwise_distance(output1, output2)
        loss_sim = self.weight_sim * label * torch.pow(dist, 2)
        loss_dissim = self.weight_dissim * (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.mean(loss_sim + loss_dissim)
        
        return loss

def computeClassWeights(train_loader):

    pair_labels = []
    for _, labels in train_loader:

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        else:
            labels = np.array(labels)
        
        if len(labels) % 2 != 0:
            labels = labels[:-1]
        
        half_batch_size = len(labels) // 2
        labels1 = labels[:half_batch_size]
        labels2 = labels[half_batch_size:]
        
        pairs_similar = (labels1 == labels2).astype(int)
        pair_labels.extend(pairs_similar)
    
    count = Counter(pair_labels) # count[0]: number of dissimilar pairs; count[1]: number of similar pairs
    total = len(pair_labels)
    
    weight_dissim = total / (2.0 * count[0]) if count[0] > 0 else 1.0
    weight_sim = total / (2.0 * count[1]) if count[1] > 0 else 1.0
    
    class_weights = {
        0: weight_dissim,
        1: weight_sim
    }
    return class_weights

def trainModel(train_loader, model, criterion, optimizer, scheduler, num_epochs=45):

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0
        batch_count = 0

        similarity_similar_sum = 0.0
        similarity_dissimilar_sum = 0.0
        similar_pair_count = 0
        dissimilar_pair_count = 0

        for i, ((data1, data2), labels) in enumerate(train_loader, 0):
            data1 = data1.to(device)
            data2 = data2.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()

            outputs1 = model(data1)
            outputs2 = model(data2)

            loss = criterion(outputs1, outputs2, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
   
            euclidean_distance = F.pairwise_distance(outputs1, outputs2)
            similarity = 1 / (1 + euclidean_distance) 

            similarity_similar_sum += similarity[labels == 1].sum().item()
            similarity_dissimilar_sum += similarity[labels == 0].sum().item()
            similar_pair_count += (labels == 1).sum().item()
            dissimilar_pair_count += (labels == 0).sum().item()

        scheduler.step()

        epoch_loss = running_loss / batch_count
        average_similarity_similar = similarity_similar_sum / similar_pair_count if similar_pair_count > 0 else 0
        average_similarity_dissimilar = similarity_dissimilar_sum / dissimilar_pair_count if dissimilar_pair_count > 0 else 0
        
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.3f}')
        print(f'Average Similarity (Similar Pairs): {average_similarity_similar:.3f}')
        print(f'Average Similarity (Dissimilar Pairs): {average_similarity_dissimilar:.3f}')

    print('Training finished.')

    model_save_path = 'models/model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

def evaluateModel(test_loader, model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_outputs = []
    all_labels = []

    # Disable gradient calculation to evaluate model
    with torch.no_grad():

        for ((data1, data2), labels) in test_loader:
            
            data1 = data1.to(device)
            data2 = data2.to(device)

            labels = labels.to(device).float()

            outputs1 = model(data1)
            outputs2 = model(data2)

            # Use euclidean distance to estimate similarity
            euclidean_distance = F.pairwise_distance(outputs1, outputs2).cpu().numpy()
            similarities = 1 / (1 + euclidean_distance)  # Scale similarity to be between 0 and 1

            all_outputs.append(similarities)
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    # Find optimal threshold before making predictions
    thresholds = np.linspace(0, 1, 100)
    optimal_threshold = 0.5
    best_f1 = 0
    final_precision = 0
    final_recall = 0

    for threshold in thresholds:
        predictions_tmp = (all_outputs >= threshold).astype(int)
        precision_tmp, recall_tmp, f1_tmp, _ = precision_recall_fscore_support(all_labels, predictions_tmp, average='binary')
        
        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            optimal_threshold = threshold
            final_precision = precision_tmp
            final_recall = recall_tmp
            predictions = predictions_tmp

    print(f'Optimal Threshold (Max F1): {optimal_threshold:.3f}, F1 Score: {best_f1:.3f}')

    accuracy = accuracy_score(all_labels, predictions)

    print(f'Accuracy: {accuracy:.3f}')
    print(f'Precision: {final_precision:.3f}')
    print(f'Recall: {final_recall:.3f}')

if __name__ == "__main__":

    data_dir = 'dataset/output'
    train_loader, test_loader = loadData(data_dir, batch_size=64)

    class_weight_dict = computeClassWeights(train_loader)
    print(f'Class Weights: {class_weight_dict}')

    model = SimilarityCNN()
    criterion = ContrastiveLoss(margin=1.0, weight_sim=class_weight_dict[1], weight_dissim=class_weight_dict[0])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1) # Reduce LR during training

    trainModel(train_loader, model, criterion, optimizer, scheduler)
    evaluateModel(test_loader, model)