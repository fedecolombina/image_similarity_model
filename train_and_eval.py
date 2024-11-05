import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from helpers.preprocessing import loadData
from helpers.model import SimilarityCNN
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from torch.optim.lr_scheduler import StepLR
from sklearn.utils.class_weight import compute_class_weight
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

def trainModel(train_loader, model, criterion, optimizer, scheduler, class_weights, num_epochs=20):

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0
        batch_count = 0

        for i, (data, labels) in enumerate(train_loader, 0):
            data = data.to(device)
            labels = torch.tensor(labels, dtype=torch.float).to(device)

            # Reset gradients from previous batch
            optimizer.zero_grad()

            # Ensure batch size is even..
            if len(data) % 2 != 0:
                data = data[:-1]
                labels = labels[:-1]

            # ..and split batches into pairs
            half_batch_size = len(data) // 2
            data1 = data[:half_batch_size]
            data2 = data[half_batch_size:]

            # Create binary labels for the pairs 
            pair_labels = (labels[:half_batch_size] == labels[half_batch_size:]).float().to(device)
            outputs1 = model(data1)
            outputs2 = model(data2)

            loss = criterion(outputs1, outputs2, pair_labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        scheduler.step()

        epoch_loss = running_loss / batch_count
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.3f}')

    print('Training finished.')

    model_save_path = 'models/model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

def evaluateModel(test_loader, model, criterion):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loss = 0.0

    all_outputs = []
    all_labels = []

    # Disable gradient calculation to evaluate model
    with torch.no_grad():

        for data, labels in test_loader:
            data = data.to(device)
            labels = torch.tensor(labels, dtype=torch.float).to(device)

            if len(data) % 2 != 0:
                data = data[:-1]
                labels = labels[:-1]

            half_batch_size = len(data) // 2
            data1 = data[:half_batch_size]
            data2 = data[half_batch_size:]

            pair_labels = (labels[:half_batch_size] == labels[half_batch_size:]).float().to(device)

            outputs1 = model(data1)
            outputs2 = model(data2)

            loss = criterion(outputs1, outputs2, pair_labels)
            test_loss += loss.item()

            # Use euclidean distance to estimate similarity
            euclidean_distance = F.pairwise_distance(outputs1, outputs2).cpu().numpy()
            similarities = 1 / (1 + euclidean_distance)  # Scale similarity to be between 0 and 1

            all_outputs.append(similarities)
            all_labels.append(pair_labels.cpu().numpy())

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.3f}')

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    # Find optimal threshold before making predictions
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    roc_auc = auc(fpr, tpr) 

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    predictions = (all_outputs >= optimal_threshold).astype(int)

    accuracy = accuracy_score(all_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')

    print(f'AUC: {roc_auc:.3f}')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')

if __name__ == "__main__":

    data_dir = 'dataset/output'
    train_loader, test_loader = loadData(data_dir, batch_size=64)

    class_weight_dict = computeClassWeights(train_loader)
    print(f'Class Weights: {class_weight_dict}')

    model = SimilarityCNN()
    criterion = ContrastiveLoss(margin=2.0, weight_sim=class_weight_dict[1], weight_dissim=class_weight_dict[0])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1) # Reduce LR during training

    trainModel(train_loader, model, criterion, optimizer, scheduler, class_weight_dict)
    evaluateModel(test_loader, model, criterion)