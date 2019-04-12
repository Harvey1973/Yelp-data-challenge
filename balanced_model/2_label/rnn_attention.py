import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten,Conv2D,Conv1D,MaxPooling1D, Dropout,RepeatVector,Permute,merge,Lambda,multiply
from keras.layers import concatenate
from keras.layers import Bidirectional, GlobalMaxPool1D,BatchNormalization
from keras.models import Model, Sequential
from keras.layers import Convolution1D,GlobalMaxPooling1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.utils import to_categorical
import numpy as np
import sys
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras import backend as K

# Get the number of cores assigned to this job.
def get_n_cores():
    # On a login node run Python with:
    # export NSLOTS=4
    # python mycode.py
    #
    nslots = os.getenv('NSLOTS')
    if nslots is not None:
      return int(nslots)
    raise ValueError('Environment variable NSLOTS is not defined.')

# Get the Tensorflow backend session.
def get_session():
    try:
        nthreads = get_n_cores() - 1
        if nthreads >= 1:
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=nthreads,
                inter_op_parallelism_threads=1,
                allow_soft_placement=True)
            return tf.Session(config=session_conf)
    except: 
        sys.stderr.write('NSLOTS is not set, using default Tensorflow session.\n')
        sys.stderr.flush()
    return ktf.get_session()

# Assign the configured Tensorflow session to keras
ktf.set_session(get_session()) 
# Rest of your Keras script starts here....

#df = pd.read_csv("C:/Users/Harvey/Desktop/Yelp_data_set/restuarant_review_balanced.csv"
#df = pd.read_csv("/home/ec2-user/Data/restuarant_review_5_label_unbalanced.csv")
#df = pd.read_csv("/usr4/cs542sp/zzjiang/Data/restuarant_review_5_label_unbalanced.csv")
train= pd.read_csv("/usr4/cs542sp/zzjiang/Data/restuarant_balanced_2_train.csv",lineterminator='\n')
test = pd.read_csv("/usr4/cs542sp/zzjiang/Data/restuarant_balanced_2_test.csv",lineterminator='\n')
print(train.shape)
print(test.shape)
print(np.unique(train['stars']))

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(train['Processed_Reviews'])


maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = train['stars']
#####################
# Test data
tokenizer.fit_on_texts(test['Processed_Reviews'])
list_tokenized_test = tokenizer.texts_to_sequences(test['Processed_Reviews'])


maxlen = 130
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
y_test = test['stars']
print(np.unique(test['stars']))


#####################################################################
# Using pretrained glove vector
#####################################################################
#GLOVE_DIR = "/home/ec2-user/Data/"
#GLOVE_DIR = "/Users/harvey/Desktop/Data/"
GLOVE_DIR = "/usr4/cs542sp/zzjiang/Data/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'),encoding = 'utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))


word_index = tokenizer.word_index
embed_size = 100
embedding_matrix = np.random.random((len(word_index) + 1, embed_size))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            embed_size,
                            weights=[embedding_matrix],
                            input_length=maxlen,
                            trainable=True)
'''
#Randomly initialized 
embedding_layer = Embedding(len(word_index) + 1,
                            embed_size,
                            input_length=maxlen,
                            trainable=True)

'''  



print("start building model") 
sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

units = 64
activations = Bidirectional(LSTM(64, return_sequences = True,kernel_regularizer=regularizers.l2(0.1)))(embedded_sequences)

# compute importance for each step
attention = Dense(1, activation='tanh')(activations)
attention = Flatten()(attention)
attention = Activation('softmax')(attention)
attention = RepeatVector(units*2)(attention)
attention = Permute([2, 1])(attention)


sent_representation = multiply([activations, attention])
sent_representation = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(units*2,))(sent_representation)

out = Dense(1, activation='sigmoid')(sent_representation)

model = Model(inputs=sequence_input, outputs=out)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()



batch_size = 512
epochs = 50
history = model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)


y_test = test["stars"]
list_sentences_test = test['Processed_Reviews']
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
print('accuracy :{0}'.format(accuracy_score(y_pred, y_test)))
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))

#################################################################
#Save train history as dict 
#################################################################

with open(r"/usr4/cs542sp/zzjiang/History/2_label/rnn_attention_pre", "wb") as output_file:
    pickle.dump(history.history, output_file)

