import numpy as np
import torch 
import torchvision 
import pandas as pd
import math
import os
import torch.nn as nn
from typing import Dict, Callable, List
from torch import FloatTensor, LongTensor
from Embedding import word_embedding

def train_rnn(input_data: FloatTensor,target_data:FloatTensor,  optim, criterion , embedding,num_itter, num_episode, alpha, lr):
    
    # init with zero optimizers
    optim.zero_grad()
    loss = 0
    
    for i in range(input_data.size(0)):
        pass
    

    #optim = optim.Adam(model.parameters(), lr)
    


def embedding_train(embedding: object, num_epoch: int,X:FloatTensor, X_weighted: FloatTensor,lr: FloatTensor , optimizer:object):
    glove=embedding
    losses = []

    opt = optimizer(glove.parameters(),lr)
    opt.zero_grad()
    
    for i in range(num_epoch):
        # embedding return loss_(X_weighted, W, Wc, b, bc, X)
        loss = glove(X_weighted)
        # step back with gradients
        loss.backward()
        # update the gradients. 
        opt.step() # compute the gradients
        
        opt.zero_grad()
        # appends
        losses.append(loss)
        
    # return model , losses, 
    return(glove, losses) 
        
    
    
