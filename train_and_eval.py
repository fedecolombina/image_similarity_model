import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from helpers.preprocessing import loadData
from helpers.model import SimilarityCNN
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from torch.optim.lr_scheduler import StepLR

# Define contrastive loss for learning similar or dissimilar pairs
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2)
        loss = label * torch.pow(dist, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        loss = torch.mean(loss)
        return loss

def trainModel(train_loader, model, criterion, optimizer, scheduler, num_epochs=20):

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

            distances = F.pairwise_distance(outputs1, outputs2)
                        
            all_outputs.append(distances.cpu().numpy())
            all_labels.append(pair_labels.cpu().numpy())


    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.3f}')

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    similarities = 1 / (1 + np.exp(-all_outputs))

    fpr, tpr, thresholds = roc_curve(all_labels, similarities)
    roc_auc = auc(fpr, tpr) 

    # Optimal threshold
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]

    #import pdb 
    #pdb.set_trace()

    predictions = (similarities >= optimal_threshold).astype(int)

    accuracy = accuracy_score(all_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')


    print(f'AUC: {roc_auc:.3f}')
    print(f'Optimal Threshold: {optimal_threshold:.3f}')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')

if __name__ == "__main__":

    data_dir = 'dataset/output'
    train_loader, test_loader = loadData(data_dir, batch_size=32)

    model = SimilarityCNN()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    trainModel(train_loader, model, criterion, optimizer, scheduler)
    evaluateModel(test_loader, model, criterion)
