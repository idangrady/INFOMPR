import numpy as np
import torch 
import torchvision 
import pandas as pd
import math
import os
import torch.nn as nn
from typing import Dict, Callable, List
from torch import FloatTensor, LongTensor


def train(input_data: Floattensor,target_data:FloatTensor,  optim, criterion , embedding,num_itter, num_episode, alpha, lr):
    
    # init with zero optimizers
    optim.zero_grad()
    loss = 0
    
    for i in range(input_data.size(0)):
        output, hidden = rnn()
        loss+= criterion (prediction, )
        pass
    

    optim = optim.Adam(model.parameters(), lr)
    
    loss = model.