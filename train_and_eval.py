import torch
import torch.optim as optim
import torch.nn.functional as F
from helpers.preprocessing import loadData
from helpers.model import SimilarityCNN
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import numpy as np

# Define contrastive loss for learning similar or dissimilar pairs
class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2)

        loss = (1 - label) * torch.pow(dist, 2) \
               + label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)

        loss = torch.mean(loss)

        return loss

def trainModel(train_loader, model, criterion, optimizer, num_epochs=5):

    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0
        batch_count = 0

        for i, (data, labels) in enumerate(train_loader, 0):
            data = data.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

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
            pair_labels = (labels[:half_batch_size] != labels[half_batch_size:]).long().to(device)

            outputs1 = model(data1)
            outputs2 = model(data2)

            loss = criterion(outputs1, outputs2, pair_labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            #print outputs for debugging
            #print(f'Epoch {epoch+1}, Batch {i+1}, Output1: {outputs1.detach().cpu().numpy()}, Output2: {outputs2.detach().cpu().numpy()}')

            running_loss += loss.item()
            batch_count += 1

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

    all_outputs1 = []
    all_outputs2 = []
    all_labels = []

    # Disable gradient calculation to evaluate model
    with torch.no_grad():

        for data, labels in test_loader:
            data = data.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            if len(data) % 2 != 0:
                data = data[:-1]
                labels = labels[:-1]

            half_batch_size = len(data) // 2
            data1 = data[:half_batch_size]
            data2 = data[half_batch_size:]

            pair_labels = (labels[:half_batch_size] != labels[half_batch_size:]).long().to(device)

            outputs1 = model(data1)
            outputs2 = model(data2)

            loss = criterion(outputs1, outputs2, pair_labels)

            test_loss += loss.item()

            all_outputs1.append(outputs1.cpu().numpy())
            all_outputs2.append(outputs2.cpu().numpy())
            all_labels.append(pair_labels.cpu().numpy())

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.3f}')

    all_outputs1 = np.concatenate(all_outputs1)
    all_outputs2 = np.concatenate(all_outputs2)
    all_labels = np.concatenate(all_labels)

    # Evaluate distances and similarity based on a threshold 
    distances = F.pairwise_distance(torch.tensor(all_outputs1), torch.tensor(all_outputs2)).numpy()
    fpr, tpr, thresholds = roc_curve(all_labels, -distances)  # Minus sign for similarity
    #roc_auc = auc(fpr, tpr)

    # Find the threshold that maximizes the efficiency
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    #import pdb 
    #pdb.set_trace()

    predictions = (distances < -optimal_threshold).astype(int)

    accuracy = accuracy_score(all_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, predictions, average='binary')

    print(f'Accuracy: {accuracy:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')

if __name__ == "__main__":

    data_dir = 'dataset/output'
    train_loader, test_loader = loadData(data_dir, batch_size=32)

    model = SimilarityCNN()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trainModel(train_loader, model, criterion, optimizer)
    evaluateModel(test_loader, model, criterion)
