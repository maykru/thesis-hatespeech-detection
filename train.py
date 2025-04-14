from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torch

def train(model, loader_train: DataLoader):       
   #set learning rate and select optimiser
    learning_rate = 0.001
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
   
    # Starts training phase
    for epoch in range(100):
        # Set model in training model
        model.train()
        predictions, gold = [], []
        # Starts batch training
        for tweet_batch, features_batch, label_batch in loader_train:
      
            label_batch = label_batch.type(torch.FloatTensor)
         
            # Feed the model
            label_pred = model(tweet_batch, features_batch)
         
            # Loss calculation
            loss = F.binary_cross_entropy(label_pred, label_batch)
         
            optimizer.zero_grad()
         
            # backwards pass
            loss.backward()
         
            # Gradients update
            optimizer.step()
         
            # Save predictions
            predictions += list(label_pred.detach().numpy())
            gold += list(label_batch.detach().numpy())
      
        # Metrics calculation
        train_accuracy = accuracy_score(gold, np.around(predictions))

        if (epoch+1) % 10 == 0:
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f" % (epoch+1, loss.item(), train_accuracy))