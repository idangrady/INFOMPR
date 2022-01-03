import numpy as np
import torch 
import torchvision 
import pandas as pd
import math
import os
import torch.nn as nn
from typing import Dict, Callable, List
from torch import FloatTensor, LongTensor


def clean_data()->List:
    """ 
    return a clean input data
    
    """
    pass

def word_to_idx()-> dict:
    """ 
    Function that maps the data and return a dictionary of words corresponding to their index
    
    return: 
        dict 1 idx to word
        dict 2 word to idx
    """
    pass

def X_weighted()->FloatTensor:
    """ 
    Weighte the input X
    1 if X ==X.max
    else: X/ X.max
    """
    pass


def loss_(
        X_weighted: FloatTensor,
        W: FloatTensor,
        Wc: FloatTensor,
        b: FloatTensor,
        bc: FloatTensor,
        X: FloatTensor,
        )-> FloatTensor:
        return((torch.sum( X_weighted* (torch.mul(W, Wc.T) + b + bc.T - torch.log(X))**2)))

class word_embedding(nn.Module):
    """ 
    function that accepts a weighted co-occurrence matrix , the co-occurrence matrix , 
    then finds the embeddings of all words
    """
    
    def __init__(self,vocab: Dict[str, int], 
                 num_emb = 15, 
                 device: str="cpu")-> None :
        super(word_embedding, self).__init__()
        
        self.device = device
        self.num_emb = num_emb
        self.vocab= vocab
        self.len_vocab = len(self.vocab)
      
    
        self.embedding_w = nn.Embedding(num_embeddings = self.len_vocab, embedding_dim = self.num_emb).to(self.device)
        self.embedding_wc = nn.Embedding(num_embeddings = self.len_vocab, embedding_dim = self.num_emb).to(self.device)
        self.embedding_bw = nn.Embedding(num_embeddings = self.len_vocab, embedding_dim= 1)
        self.embedding_bc = nn.Embedding(num_embeddings = self.len_vocab, embedding_dim= 1)


    def forward(self, X_weighted: FloatTensor,X: FloatTensor)-> FloatTensor:
        #create an array in the length of the input (self.len_vocab) 
        em_input = torch.arange(self.len_vocab).to(self.device)
        
        #pass the values through the embedding
        W = self.embedding_w(em_input)
        Wc = self.embedding_wc(em_input)
        b = self.embedding_bw(em_input)
        bc = self.embedding_bc (em_input)
        # return the loss for the training
        return loss_(X_weighted, W, Wc, b, bc, X)
    
    def get_vector(self):
        """ 
        function that receives no input and produces the word vectors and 
        context word vectors of all words
        """
        pass
    
    
