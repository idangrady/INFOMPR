# -*- coding: utf-8 -*-
"""In-House_Abstractive_Text_Summarization(17102135).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17dCQU4fi-n1458_KFK_ZYlyj2bZIHOFq
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
import numpy as np   #Package for scientific computing and dealing with arrays
import pandas as pd  #Package providing fast, flexible and expressive data structures
import re            #re stands for RegularExpression providing full support for Perl-like Regular Expressions in Python
from bs4 import BeautifulSoup   #Package for pulling data out of HTML and XML files
from keras.preprocessing.text import Tokenizer  #For tokenizing the input sequences
from keras.preprocessing.sequence import pad_sequences  #For Padding the seqences to same length
from nltk.corpus import stopwords   #For removing filler words
from tensorflow.keras.layers import Input, LSTM, Attention, Embedding, Dense, Concatenate, TimeDistributed   #Layers required to implement the model
from tensorflow.keras.models import Model  #Helps in grouping the layers into an object with training and inference features
from tensorflow.keras.callbacks import EarlyStopping  #Allows training the model on large no. of training epochs & stop once the performance stops improving on validation dataset
import warnings  #shows warning message that may arise 
from data import get_data

SIZE_OF_DATASET_IN_ARTICLES = 3000
MAX_ARTICLE_LEN = 400
MAX_ABSTRACT_LEN = 50

pd.set_option("display.max_colwidth", 200) #Setting the data sructure display length
warnings.filterwarnings("ignore")

# TODO: discuss if we only use the train_*.bin files and separate them into train, val, and test because those are already 287,113 articles
# if we only use train, we should remove mode to make it less misleading
news_data=pd.DataFrame(get_data(n = SIZE_OF_DATASET_IN_ARTICLES, mode = "train"))
news_data = news_data.transpose()
print(news_data.shape) #Analyzing the shape of the dataset
news_data.head(n=10)

DATASET_COLUMNS = ["Article", "Abstract"]
news_data.columns = DATASET_COLUMNS
news_data.head(n=10)

#Reducing the length of dataset for better training and performance
news_data.dropna(axis=0,inplace=True) #Dropping the rows with Missing values

news_data.info() #Getting more info on datatypes and shape of Dataset

#Preprocessing

#This the dictionary used for expanding contractions
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",

                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",

                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}

#Text Cleaning
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
def text_cleaner(text,num):
    newString = text.lower()  #converts all uppercase characters in the string into lowercase characters and returns it
    newString = BeautifulSoup(newString, "lxml").text #parses the string into an lxml.html 
    newString = re.sub(r'\([^)]*\)', '', newString) #used to replace a string that matches a regular expression instead of perfect match
    newString = re.sub('"','', newString)           
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")]) #for expanding contractions using the contraction_mapping dictionary    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    if(num==0): 
      tokens = [w for w in newString.split() if not w in stop_words]  #converting the strings into tokens; removing stopwords
    else :
      tokens = newString.split() # converting the strings into tokens, leaving the stopwords in
    long_words=[]
    for i in tokens:
        if len(i)>1:                  #removing short words
            long_words.append(i)   
    return (" ".join(long_words)).strip()

#Calling the function
cleaned_article = []
for t in news_data['Article']:
    cleaned_article.append(text_cleaner(t,0))

news_data['Article'][:10] #Looking at the 'Article' column of the dataset

cleaned_article[:10] #Looking at the article after removing stop words, special characters , punctuations etc.

#Abstract Cleaning 
cleaned_abstract = []    #Using the text_cleaner function for cleaning summary too
for t in news_data['Abstract']:
    cleaned_abstract.append(text_cleaner(t,1))

news_data['Abstract'][:10]

cleaned_abstract[:10]

news_data['Cleaned_Article'] = cleaned_article  #Adding cleaned article to the dataset
news_data['Cleaned_Abstract'] = cleaned_abstract  #Adding cleaned abstract to the dataset
#Dropping Empty Rows
news_data['Cleaned_Abstract'].replace('', np.nan, inplace=True)
#Dropping rows with Missing values
news_data.dropna(axis=0,inplace=True)

#Before Cleaning
print("Before Preprocessing:\n")
for i in range(5):
    print("Article:",news_data['Article'][i])
    print("Abstract:",news_data['Abstract'][i])
    print("\n")

#Printing the Cleaned text and summary which will work as input to the model 
print("After Preprocessing:\n")
for i in range(5):
    print("Article:",news_data['Cleaned_Article'][i])
    print("Abstract:",news_data['Cleaned_Abstract'][i])
    print("\n")

# TODO: this whole part can go, not interesting
#Data Visualization
import matplotlib.pyplot as plt
article_word_count = []
abstract_word_count = []

#Populating the lists with sentence lengths
for i in news_data['Cleaned_Article']:
      article_word_count.append(len(i.split()))

for i in news_data['Cleaned_Abstract']:
      abstract_word_count.append(len(i.split()))

length_df = pd.DataFrame({'article':article_word_count, 'abstract':abstract_word_count})
length_df.hist(bins = 30)
plt.show()

#Function for gauging the actual article and abstract lengths
count=0 
for i in news_data['Cleaned_Article']:
    if(len(i.split())<=MAX_ARTICLE_LEN):
        count=count+1
print(f"% of articles that are <= {MAX_ARTICLE_LEN} tokens: {(count/len(news_data['Cleaned_Article'])) * 100}")

#Function for getting the Maximum Abstract length
count=0
for i in news_data['Cleaned_Abstract']:
    if(len(i.split())<=MAX_ABSTRACT_LEN):
        count=count+1
print(f"% of articles that are <= {MAX_ABSTRACT_LEN} tokens: {(count/len(news_data['Cleaned_Abstract'])) * 100}")


cleaned_article =np.array(news_data['Cleaned_Article'])
cleaned_abstract=np.array(news_data['Cleaned_Abstract'])

short_article=[]
short_abstract=[]

# only keep the articles and the abstract when they are smaller than the max lengths
for i in range(len(cleaned_article)):
    if(len(cleaned_abstract[i].split())<=MAX_ABSTRACT_LEN and len(cleaned_article[i].split())<=MAX_ARTICLE_LEN):
        short_article.append(cleaned_article[i])
        short_abstract.append(cleaned_abstract[i])
        
df=pd.DataFrame({'article':short_article,'abstract':short_abstract})

#Adding START and END tags/tokens to abstract for better decoding
df['abstract'] = df['abstract'].apply(lambda x : 'sostok '+ x + ' eostok')

#Splitting the Dataset twice to get 80% training data, 10% of validation data and 10% of test data
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(np.array(df['article']),np.array(df['abstract']),test_size=0.2,random_state=0,shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5, random_state = 0, shuffle = True)

#Preparing Tokenizer

#Text Tokenizer
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

#preparing a tokenizer on training data
X_tokenizer = Tokenizer() 
X_tokenizer.fit_on_texts(list(X_train))

#Rarewords and their coverage in the article
thresh = 4  #If a word whose count is less than threshold i.e 4, then it's considered as rare word 

cnt = 0      #denotes no. of rare words whose count falls below threshold
tot_cnt = 0  #denotes size of unique words in the text
freq = 0
tot_freq = 0

for key,value in X_tokenizer.word_counts.items():
    tot_cnt=tot_cnt+1
    tot_freq=tot_freq+value
    if(value<thresh):
        cnt=cnt+1
        freq=freq+value
    
print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
print("Total Coverage of rare words:",(freq/tot_freq)*100)

#Defining the Tokenizer with top most common words for articles
# TODO: include not only training data but also val and test into the vocabulary

#Preparing a Tokenizer for articles on training data
X_tokenizer = Tokenizer(num_words=tot_cnt - cnt)   #provides top most common words
X_tokenizer.fit_on_texts(list(X_train))

#Converting text sequences into integer sequences; note that we use the vocabulary that was generated on the
# training data to generate the validation and test sequences as well
X_train_seq    =   X_tokenizer.texts_to_sequences(X_train) 
X_val_seq   =   X_tokenizer.texts_to_sequences(X_val)
X_test_seq = X_tokenizer.texts_to_sequences(X_train)

#Padding zero upto maximum length; we want one length so we can make the encoder max_article_len
X_train    =   pad_sequences(X_train_seq,  maxlen = MAX_ARTICLE_LEN, padding = 'post')
X_val   =   pad_sequences(X_val_seq, maxlen = MAX_ARTICLE_LEN, padding = 'post')
X_test = pad_sequences(X_test_seq, maxlen = MAX_ARTICLE_LEN, padding = 'post')


#Size of vocabulary (+1 for padding token)
X_voc   =  X_tokenizer.num_words + 1

#Abstract Tokenizer

#Preparing a Tokenizer for abstracts on training data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_train))

#Rarewords and their coverage in summary

thresh = 4  ##If a word whose count is less than threshold i.e 6, then it's considered as rare word 

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key,value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt+1
    tot_freq = tot_freq+value
    if(value<thresh):
        cnt = cnt+1
        freq = freq+value
    
print("% of rare words in vocabulary:",(cnt/tot_cnt)*100)
print("Total Coverage of rare words:",(freq/tot_freq)*100)

#Defining Tokenizer with the most common words in abstract

#Preparing a tokenizer for abstracts on training data
# TODO: be aware that this exludes for example names of people that appear in just one article < thresh times!
y_tokenizer = Tokenizer(num_words=tot_cnt - cnt)  #provides top most common words
# TODO: if we do it that way, we will also have sos as a token? why would we want this in our vocabulary
y_tokenizer.fit_on_texts(list(y_train))

#Converting text sequences into integer sequences
y_train_seq    =   y_tokenizer.texts_to_sequences(y_train) 
y_val_seq   =   y_tokenizer.texts_to_sequences(y_val) 
y_test_seq   =   y_tokenizer.texts_to_sequences(y_test) 

#Padding zero upto maximum length
y_train    =   pad_sequences(y_train_seq, maxlen=MAX_ABSTRACT_LEN, padding='post')
y_val   =   pad_sequences(y_val_seq, maxlen=MAX_ABSTRACT_LEN, padding='post')
y_test = pad_sequences(y_test_seq, maxlen=MAX_ABSTRACT_LEN, padding='post')


#size of vocabulary based on the vocabulary of the abstracts of the training set
y_voc  =   y_tokenizer.num_words +1

#Checking the length of training data
# TODO: what does this line do? those two statements should deliver the same results right?
y_tokenizer.word_counts['sostok'],len(y_train)

#Deleting rows containing only START and END tokens
#For Training set
ind=[]
for i in range(len(y_train)):
    cnt=0
    for j in y_train[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_train=np.delete(y_train,ind, axis=0)
X_train=np.delete(X_train,ind, axis=0)


#For Validation set
ind=[]
for i in range(len(y_val)):
    cnt=0
    for j in y_val[i]:
        if j!=0:
            cnt=cnt+1
    if(cnt==2):
        ind.append(i)

y_val=np.delete(y_val,ind, axis=0)
X_val=np.delete(X_val,ind, axis=0)

# TODO: maybe do this for test as well or remove this part, seems unnecessary

#Model Building

# TODO: skip attention for now
#Adding Custom Attention layer 

import tensorflow as tf
import os
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]

from keras import backend as K 
K.clear_session()  #Resets all state generated by Keras

latent_dim = 256
embedding_dim = 256

# Encoder
encoder_inputs = Input(shape=(MAX_ARTICLE_LEN,))

# TODO: understand how embedding works here; is this our own trained embedding? maybe we should just use word2vec
#embedding layer
enc_emb =  Embedding(X_voc, embedding_dim,trainable=True, mask_zero = True)(encoder_inputs)

#encoder lstm
# TODO: why are we not using an activation function? default is none, only for recurrent activation the default is sigmoid
# TODO: why do we need the encoder_outputs? only for attention probably
encoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)

#Setting up the Decoder using 'encoder_states' as initial state
# TODO: figure out why the shape is None? because we also use it later for our decoding when we only give one
decoder_inputs = Input(shape=(None,))

#Embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim,trainable=True, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
# TODO: is the LSTM bidirectional? why do we even need those two states?
# TODO: why do the graphs say (None, 256) where 256 is the latent_dim (the length of the state vectors); shouldn't it be clear that it is (1, 256)?
decoder_outputs ,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# #Attention layer; removed for now
# attn_layer = AttentionLayer(name='attention_layer')
# attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# #Concating Attention input and Decoder LSTM output
# decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

#Dense layer
# TODO: figure out what TimeDistributed does
decoder_dense =  TimeDistributed(Dense(units = y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

#Defining the model 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

#Visualize the Model
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='training_model_plot.png', show_shapes=True, show_layer_names=True)

# TODO: understand this
#Adding Metrics
model.compile(optimizer='rmsprop' , loss='sparse_categorical_crossentropy' , metrics=['accuracy'])

#Adding Callback
# TODO: how exactly does this work? is there an internal mapping of the string "val_loss" to running a test on the specified validation_data?
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# Commented out IPython magic to ensure Python compatibility.
#Training the Model
# %tensorflow_version 1.x
# indexing is clear, removing the eos token for the decoder inputs and removing the sos token for the decoder outputs
# TODO: think about how exactly this is working with calculating the loss etc., maybe that's the problem
# TODO: why is y 3-dimensional
# TODO: fit only on one example and check if model actually works
history = model.fit(x = [X_train,y_train[:,:-1]], y = y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,epochs=20,callbacks=[es],batch_size = int(y_train.shape[0] * 0.1), validation_data=([X_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

#Visualizing Accuracy 
from matplotlib import pyplot
pyplot.plot(history.history['accuracy'], label='train') 
pyplot.plot(history.history['val_accuracy'], label='val') 
pyplot.legend() 
pyplot.show()

#Visualizing Loss 
pyplot.plot(history.history['loss'], label='train') 
pyplot.plot(history.history['val_loss'], label='val') 
pyplot.legend() 
pyplot.show()


#Building Dictionary for Source Vocabulary
# TODO: what is happening here? we are just saving the vocabularies of the tokenizers (index -> word or word -> index maps)
target_index_word=y_tokenizer.index_word 
source_index_word=X_tokenizer.index_word 
target_word_index=y_tokenizer.word_index

#Testing phase
#Encoding the input sequence to get the feature vector
# TODO: is this what links this model to the trained model? that we use the same encoder_inputs Input object?
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

#Decoder setup
#These tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
# TODO: remove this, probably only used for attention
# decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

#Getting the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs) 

#Setting the initial states to the states from the previous time step for better prediction
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

# TODO: remove this
# #Attention inference
# attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
# decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

#Adding Dense softmax layer to generate proability distribution over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_outputs2) 

#Final Decoder model
# TODO: how is this all linked to what we trained before? where does the parameter sharing happen?
# TODO: why are we appending the lists instead of just writing it in one list
decoder_model = Model(
    [decoder_inputs] + [decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

#Function defining the implementation of inference process
def decode_sequence(input_seq):
    #Encoding the input as state vectors
    # TODO: plotting to see if this is actually the trained encoder and it is, but why?
    plot_model(encoder_model, to_file='encoder_model_plot.png', show_shapes=True, show_layer_names=True)
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    
    #Generating empty target sequence of length 1
    target_seq = np.zeros((1,1))
    
    #Populating the first word of target sequence with the start word
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_h, e_c])

        #Sampling a token
        # TODO: understand the indexing: why -1 instead of 0 as well? should the output tokens be of shape (1, 1, num_words)?
        # TODO: I added the plus one because I think the neurons in the dense layer start at 0 but our dictionary starts at 1, despite a one-on-one mapping from
        # dictionary index numbers to neurons if I understand correctly
        sampled_token_index = np.argmax(output_tokens[0, -1, :]) + 1
        sampled_token = target_index_word[sampled_token_index]
        
        if(sampled_token!='eostok'):
            decoded_sentence += ' '+sampled_token

        #Exit condition: either hit max length or find stop word
        # TODO: it used to be >= (max_abstract_len - 1); why? doesn't make sense to me
        if (sampled_token == 'eostok'  or len(decoded_sentence.split()) == MAX_ABSTRACT_LEN):
            stop_condition = True

        #Updating the target sequence (of length 1)
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        #Updating internal states
        e_h, e_c = h, c

    return decoded_sentence

#Functions to convert an integer sequence to a word sequence for summary as well as reviews 
def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sostok']) and i!=target_word_index['eostok']):
            newString=newString+target_index_word[i]+' '
    return newString

def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+source_index_word[i]+' '
    return newString

#Summaries generated by the model

# TODO: change this, will crash when there is less than 20
for i in range(0,20):
    print("Article:",seq2text(X_train[i]))
    print("Original summary:",seq2summary(y_train[i]))
    print("article to be decoded:")
    print(X_train[i].reshape(1,MAX_ARTICLE_LEN))
    print("Predicted summary:",decode_sequence(X_train[i].reshape(1,MAX_ARTICLE_LEN)))
    print("\n")

# #BLEU Score of Training set
# #n-gram individual BLEU
# from nltk.translate.bleu_score import sentence_bleu
# for i in range(0,1000):
#   reference = seq2summary(y_test[i])
#   candidate = decode_sequence(X_test[i].reshape(1, max_text_len))

# print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
# print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
# print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))

# #4-gram cumulative BLEU
# from nltk.translate.bleu_score import sentence_bleu
# for i in range(0,1000):
#   reference = seq2summary(y_test[i])
#   candidate = decode_sequence(X_test[i].reshape(1, max_text_len))

# score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
# print(score)

# #cumulative BLEU scores
# from nltk.translate.bleu_score import sentence_bleu
# for i in range(0,1000):
#   reference = seq2summary(y_test[i])
#   candidate = decode_sequence(X_test[i].reshape(1, max_text_len))

# print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
# print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
# print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

# #BLEU Score of Test/Validation set
# #n-gram individual BLEU
# from nltk.translate.bleu_score import sentence_bleu
# for i in range(0,1000):
#   reference = seq2summary(y_test[i])
#   candidate = decode_sequence(X_test[i].reshape(1, max_text_len))
# print("Test/Validation Set :")
# print('Individual 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# print('Individual 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 1, 0, 0)))
# print('Individual 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 1, 0)))
# print('Individual 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0, 0, 0, 1)))

# #4-gram cumulative BLEU
# from nltk.translate.bleu_score import sentence_bleu
# for i in range(0,1000):
#   reference = seq2summary(y_test[i])
#   candidate = decode_sequence(X_test[i].reshape(1, max_text_len))

# score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
# print(score)

# #cumulative BLEU scores
# from nltk.translate.bleu_score import sentence_bleu
# for i in range(0,1000):
#   reference = seq2summary(y_test[i])
#   candidate = decode_sequence(X_test[i].reshape(1, max_text_len))

# print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
# print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
# print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
# print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))