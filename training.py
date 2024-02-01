import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from collections import OrderedDict
import itertools
from tst import Transformer
from src.dataset import LSDataset
from src.utils import fit, visualise_result
import csv
import matplotlib.pyplot as plt

# Dataset name
name = 'bsh'

# Fixed parameters
d_input = 3
d_output = 1 
h = 1 
N = 2
chunk_mode = None 
NUM_WORKERS = 0
pe = None 
BATCH_SIZE = 12
EPOCHS = 200
dropout = 0.2  
LR = 0.002
opt = 'Adam'


# ===== user set params ====
param_grid = OrderedDict({
    "d_model": [16, 32, 48],
    "q": [1, 3, 5],
    "k": [1, 3, 5],
    "v": [1, 3, 5],
    "attention_size": [6, 9, 12]
})


# Generate all possible combinations of parameter values
param_combinations = list(itertools.product(*param_grid.values()))

for i, params in enumerate(param_combinations):
    print(f"Training model {i+1}/{len(param_combinations)} with params {params}")

    # Set the parameters
    d_model, q, k, v, attention_size  = params

    # Config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # Load dataset
    lsDataset = LSDataset(name)

    # Split the dataset into train and test sets
    dataset_test = Subset(lsDataset, range(len(lsDataset) - 12, len(lsDataset)))
    dataset_valid = Subset(lsDataset, range(len(lsDataset) - 24, len(lsDataset) - 12))
    dataset_train = Subset(lsDataset, range(0, len(lsDataset) - 24))

    dataloader_train = DataLoader(dataset_train,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=NUM_WORKERS,
                                pin_memory=False
                                )

    dataloader_val = DataLoader(dataset_valid,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=NUM_WORKERS
                                )

    dataloader_test = DataLoader(dataset_test,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS
                                )

    # Load transformer with Adam optimizer and MSE loss function
    if pe == None:
        net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                        dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
    else:
        pe_period = 12
        net = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size,
                        dropout=dropout, chunk_mode=chunk_mode, pe=pe, pe_period=pe_period).to(device)  
            

    # Create the optimizer with the initial learning rate
    optimizer = optim.Adam(net.parameters(), lr=LR)
    loss_function = nn.MSELoss()   

    # Fit model
    with tqdm(total=EPOCHS) as pbar:
        train_loss, valid_loss = fit(net, optimizer, loss_function, dataloader_train,
                dataloader_val, epochs=EPOCHS, pbar=pbar, device=device)

    # loss visualisation
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

    # Switch to evaluation
    _ = net.eval()

    # Select target values in test split
    y_true = lsDataset._y[dataloader_test.dataset.indices]
    train_y = lsDataset._y[dataloader_train.dataset.indices]

    # Compute predictions (test)
    predictions = torch.empty(len(dataloader_test.dataset), 12, 1)  
    idx_prediction = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader_test, total=len(dataloader_test)):
            netout = net(x.to(device)).cpu()
            predictions[idx_prediction:idx_prediction+x.shape[0]] = netout
            idx_prediction += x.shape[0]

    # Save model
    torch.save(net.state_dict(), 'models/model.pth')