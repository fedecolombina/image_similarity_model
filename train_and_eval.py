import torch
import torch.optim as optim
import torch.nn.functional as F
from helpers.preprocessing import load_data
from helpers.model import myCNN
import torch.nn as nn

#define contrastive loss for learning similar or dissimilar pairs
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

def train_model(train_loader, model, criterion, optimizer, num_epochs=20):

    #GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()

    for epoch in range(num_epochs):

        running_loss = 0.0
        batch_count = 0

        #load batches of data and labels
        for i, (data, labels) in enumerate(train_loader, 0):
            data = data.to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)

            #reset gradients from previous batch
            optimizer.zero_grad()

            #ensure batch size is even
            if len(data) % 2 != 0:
                data = data[:-1]
                labels = labels[:-1]

            #split batches into pairs
            half_batch_size = len(data) // 2
            data1 = data[:half_batch_size]
            data2 = data[half_batch_size:]

            #create binary labels for the pairs (1 for same class, 0 for different class)
            pair_labels = torch.tensor([1 if labels[j] == labels[j + half_batch_size] else 0 for j in range(half_batch_size)], dtype=torch.float32).to(device)

            #create outputs and compute loss
            outputs1 = model(data1)
            outputs2 = model(data2)

            loss = criterion(outputs1, outputs2, pair_labels)

            #backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        #print average loss after each epoch
        epoch_loss = running_loss / batch_count
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.3f}')

    print('Finished Training')

    #save the model
    model_save_path = 'models/model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

def evaluate_model(test_loader, model, criterion):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loss = 0.0

    #disable gradient calculation to evaluate model
    with torch.no_grad():

        for data, labels in test_loader:
            data = data.to(device)
            labels = torch.tensor(labels, dtype=torch.float32).to(device)

            if len(data) % 2 != 0:
                data = data[:-1]
                labels = labels[:-1]

            half_batch_size = len(data) // 2
            data1 = data[:half_batch_size]
            data2 = data[half_batch_size:]

            pair_labels = torch.tensor([1 if labels[j] == labels[j + half_batch_size] else 0 for j in range(half_batch_size)], dtype=torch.float32).to(device)

            outputs1 = model(data1)
            outputs2 = model(data2)

            loss = criterion(outputs1, outputs2, pair_labels)

            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.3f}')

if __name__ == "__main__":

    data_dir = 'dataset/output'
    train_loader, test_loader = load_data(data_dir, subset_size=1, batch_size=50)

    model = myCNN()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, model, criterion, optimizer)
    evaluate_model(test_loader, model, criterion)
