import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torchvision.transforms import transforms

from src.dataset import get_train_dataloader, get_test_dataloader
from src.models import FCNet

EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 5e-5

# Load data
train_loader = get_train_dataloader(batch_size=BATCH_SIZE)
test_loader = get_test_dataloader(batch_size=BATCH_SIZE)

# Model creation and configuration
model = FCNet()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
compute_loss = torch.nn.CrossEntropyLoss(reduction='mean')

# Check to use cuda
use_cuda: bool = torch.cuda.is_available()
if use_cuda:
    model = model.cuda()

# Main loop
start_n_iter = 0
start_epoch = 0

n_iter = start_n_iter
for epoch in range(EPOCHS):
    model.train()
    
    # use prefetch_generator and tqdm for iterating through data
    pbar = tqdm(enumerate(train_loader),
                total=len(train_loader))
    start_time = time.time()

    # for loop going through dataset
    for i, data in pbar:
        # data preparation
        image, label = data
        if use_cuda:
            image = image.cuda()
            label = label.cuda()

        # keep track of preparation time
        prepare_time = start_time - time.time()

        # forward and backward pass
        out = model(image)
        loss = compute_loss(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute computation time and *compute_efficiency*
        process_time = start_time-time.time()-prepare_time
        compute_efficiency = process_time/(process_time+prepare_time)
        pbar.set_description(
            f'Compute efficiency: {compute_efficiency:.2f}, ' 
            f'loss: {loss.item():.2f},  epoch: {epoch}/{EPOCHS}')
        start_time = time.time()

    # test data every N=1 epochs
    if epoch % 1 == 0:
        model.eval()

        correct = 0
        total = 0
        

        pbar = tqdm(enumerate(test_loader),
                total=len(test_loader)) 
        with torch.no_grad():
            for i, data in pbar:
                # data preparation
                image, label = data
                if use_cuda:
                    image = image.cuda()
                    label = label.cuda()
                
                out = model(image)
                _, predicted = torch.max(out.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        print(f'Accuracy on test set: {100*correct/total:.2f}')

