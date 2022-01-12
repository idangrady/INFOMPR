import tensorflow as tf
import keras
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
print("Libraires imported Successfuly")



download = False

if download:
    spacy.cli.download("en_core_web_lg")
    spacy.cli.download("gn_core_web_lg")
print("Do not need to download en_core_web_lg. if so ==> download = True")


spacy_ger = spacy.load('gn_core_web_lg')
spacy_eng = spacy.load('en_core_web_lg')



class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size,  hidden_size, num_layers, p_dropout):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_Size = embedding_size
        self.num_layers = num_layers

        self.dropout= nn.Dropout(p_dropout)
        self.embedding = nn.Embedding(input_size,embedding_size)  # input size maped to embedding size
        self.lstm = nn.LSTM(embedding_size,hidden_size, num_layers = num_layers, dropout=p_dropout)

    def forward(self, x, h):
        """
        we send to the forward the indexes of the word in the vocabulary

        shape x: sequence length , N ==> ( batch size)
        (sequence length, batch_size)
        """
        embedding = self.dropout(self.embedding_Size(x))
        output, (hidden, cell) = self.lstm(embedding)
        """ 
        in the encoder, the only thing that we can about is the input and the cell. because this is what we feed forward. 
        however, in the decoder layer, we would care more in the output.
        The idea is that we do not care from the prediction along the encoder, yet only in the decoder 
        
        shape: 
        shape of the embedding:
        x comes as (sequence length, batch size)
        
        but because we add embedding dimention for each word, we could look at it as we concat and train for each word
        its embedding representation 
        """
        return (hidden, cell)




class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size,num_layers , p_dropout):
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.outsize =output_size


        """"  
        only the outout interests you
        the input is the embedding from the encoder
        
        the architecture could vary for different tasks
        
        shapes: 
        is is important to note that the input size and the output size have to be the same is that case
        if the vocab size is :
        5000 x d the output size has to also be 5000 as it distributes the probability of each word to appear.
        """
        #TODO: to ask: why do we need to feed here the embedding as well? as we get the input for the anbedding?
        #TODO: To ask: why the hidden size of the encoder and the decoder need to be the same?

        self.dropout = nn.Dropout(p_dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers= num_layers, dropout= p_dropout)
        self.FConnected = nn.Linear(hidden_size, output_size)
        self.hidden_act = nn.Tanh()
        self.output_act = nn.Sigmoid()

    def forward(self,x,hidden, cell):
        """
        the decoder would predict one word at a time. given the preivous hidden and cell state. the preivous predicted word

        shapes:
        It depends on the way we take the input
        if we do word by word:
        x: (batch_size,) ==> we should x.expend_dims()

        if we take as an input more than one word
        x: (sequence_length, num_words)
        """
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # shapes: (1, N, embedding_size)

        # lstm
        output, (cur_hidden,cur_cell) = self.lstm(embedding, (hidden, cell))
        """ 
        the output : what we think the next word should be
        hid,cell ==> the hidden and cell for the next layer in the sequence. 
        
        shapes:
        output: (1,batch, hidden_size)
        """

        fully_connected = self.FConnected(output)
        # shape of fully connected: (1, batch_size, length_vocab)
        """ 
        why do we want to remove the first dim?
        when we send it to the loss function:?
        we need to make sure that they are in the right shape. 
        adapt it until it works.
        """
        #TODO to ask: why dont we need to pass it through a sigmoid function. Does it do it automatically for us?

        fully_connected = fully_connected.squeeze(0)

        return(fully_connected, (cur_hidden,cur_cell))

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
    def forward(self, source, target, vocab, teacher_Ratio = 0.5):
        """
        In the forward path we pass the data through th encoder . recieve the output vector. and send it to the decoder.
        there are a couple of parameters that we could play with them

        - the one, the amount of time we feed in the model the prediction, to the real target variable. This is done in order to avoid a situation where we are
        train the models on falsy prediction. : Teacher Ratio

        output: a sequence of prediction based on the model prediction length.
        """

        batch_size = source[1]
        # source shape: (seq length , batchsize)
        target_len = target[0]
        # target len( seq_lengh, batchsize)

        target_vocab_length = len(vocab)
        # amout of words in the vocabulary

        prediction = np.zeros([target_len, batch_size, target_vocab_length])
        """ 
        the target length is a constant number
        batch size is a constant as well
        the target is used during the training for the fit. So we basically use that value as a restriction , so the model could learn based on that. 
        """

        """ 
        the architechture is as follow:
        we send the input to the encoder. 
            - Embedding
            - LSTM
                output: Vector/ Matrix
        the take vector the encoder produces, and feed to the decoder. 
        
        Now here we have room to play : To answer for ourself!!!!!!!
            - Do we send word by word to the decoder
            - Do we send the entire sequence
            - do we output word by words prediction? or start in a later phase. 
        """

        # send to the encoder:
        # the encoder outputs the hidden & the cell
        (hidden, cell) = self.decoder(source)


        for i in range(target_len):

            current_target = target[i]

            curreent_prediction = self.encoder(source, hidden, cell)

            if np.random.random()< teacher_Ratio:
                # we take the real value
                prediction[i] = current_target
            else:
                prediction[i] = curreent_prediction

        return  prediction

