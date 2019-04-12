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


df = pd.read_csv("/usr4/cs542sp/zzjiang/Data/restuarant_review_balanced.csv")

train = df.sample(frac = 0.8,random_state = 200)
test = df.drop(train.index)



max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(train['Processed_Reviews'])

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = train['sentiment']
word_index = tokenizer.word_index



embed_size = 128
model = Sequential()
model.add(Embedding(len(word_index)+1, embed_size,input_length = maxlen))
model.add(LSTM(32, return_sequences = True))
#model.add(Bidirectional(LSTM(32, return_sequences = True)))
#model.add(GlobalMaxPool1D())
model.add(MaxPooling1D(pool_size = 4))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 512
epochs = 100
history = model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

#################################################################
#Save train history as dict 
#################################################################

with open(r"/usr4/cs542sp/zzjiang/History/rnn_vanilla_two_label", "wb") as output_file:
    pickle.dump(history.history, output_file)






y_test = test["sentiment"]
list_sentences_test = test['Processed_Reviews']
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
prediction = model.predict(X_te)
y_pred = (prediction > 0.5)
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
print('accuracy :{0}'.format(accuracy_score(y_pred, y_test)))
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')