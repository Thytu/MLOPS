import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, inS, outS):
        super().__init__()
        self.input_size = inS
        self.fc1 = nn.Linear(in_features=inS, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=outS)
    
    def forward(self, t):
        t = t.reshape(-1, self.input_size)
        t = self.fc1(t)
        t = F.relu(t)
    
        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        t = F.softmax(t, dim=1)

        return t


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def train(network, train_loader, optimizer):
    network.train()
    correct_in_episode = 0
    episode_loss = 0
  
    for batch in train_loader:
        images, labels = batch

        predictions = network(images)
        loss = F.cross_entropy(predictions, labels)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        episode_loss += loss.item()
        correct_in_episode += get_num_correct(predictions, labels)

    return correct_in_episode, episode_loss


def test(network, test_loader):
    network.eval()
    episode_loss = 0
    correct_in_episode = 0
  
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch

            predictions = network(images)
            loss = F.cross_entropy(predictions, labels)

            episode_loss = loss.item()
            correct_in_episode += get_num_correct(predictions, labels)

    return correct_in_episode, episode_loss
