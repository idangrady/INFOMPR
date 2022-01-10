import numpy as np
import torch 
import torchvision 
import pandas as pd
import math
import os
import torch.nn as nn
from typing import Dict, Callable, List
from torch import FloatTensor, LongTensor
import nltk
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup
from utils import contraction_mapping
from nltk.corpus import stopwords
import pickle
print("Loaded packages")



stop_words = set(stopwords.words('english')) 

def clean_data(text,num)->List:
    """ 
    return a clean input data
    
    """
    newString = text.lower()  #converts all uppercase characters in the string into lowercase characters and returns it
    newString = BeautifulSoup(newString, "lxml").text #parses the string into an lxml.html 
    newString = re.sub(r'\([^)]*\)', '', newString) #used to replace a string that matches a regular expression instead of perfect match
    newString = re.sub('"','', newString)           
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")]) #for expanding contractions using the contraction_mapping dictionary    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    if(num==0): 
      tokens = [w for w in newString.split() if not w in stop_words]  #converting the strings into tokens
    else :
      tokens = newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:                  #removing short words
            long_words.append(i)   
    return (" ".join(long_words)).strip()

def word_to_idx(data)-> dict:
    """ 
    Function that maps the data and return a dictionary of words corresponding to their index
    
    return: 
        dict 1 idx to word
        dict 2 word to idx
    """
    set_words  = set(data) # eliminate duplicates in the text
    w_2_i = {set_words[i]:i for i in range(len(set_words))}
    i_2_w = {i:set_words[i] for i in range(len(set_words))}
    
    return (w_2_i, i_2_w)

def X_weighted(X: FloatTensor)->FloatTensor:
    """ 
    Weighte the input X
    1 if X ==X.max
    else: X/ X.max
    """
    max_value = X.max()
    return (torch.where(X==max_value,X, X/max_value))

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
                 num_emb : 15,
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
        # return the entire sequence
        embedding_input = torch.arange(self.vocab_len).to(self.device)
        return(self.embedding_w,self.embedding_wc)
    
    


