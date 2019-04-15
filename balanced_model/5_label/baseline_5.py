import pandas as pd
import pickle
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten,Conv2D,Conv1D,MaxPooling1D, Dropout
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
print("finished importing")



#######################################################################
# Read in Data and tokenize , prepare training data and test data
#df = pd.read_csv("C:/Users/Harvey/Desktop/Yelp_data_set/restuarant_review_5_label_unbalanced.csv")
#df = pd.read_csv("/home/ec2-user/Data/restuarant_review_5_label_unbalanced.csv")

train = pd.read_csv("/usr4/cs542sp/zzjiang/Data/restuarant_balance_5_train.csv")
test = pd.read_csv("/usr4/cs542sp/zzjiang/Data/restuarant_balance_5_test.csv")

train.loc[:,'stars'] -= 1
test.loc[:,'stars'] -=1
reviews_train = train['Processed_Reviews']
reviews_test = test['Processed_Reviews']
print(np.unique(train['stars']))
print(np.unique(test['stars']))

whole_data = pd.concat([reviews_train,reviews_test])

maxlen = 130
max_features = 15000
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(whole_data)
list_tokenized_train = tokenizer.texts_to_sequences(reviews_train)
x_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
y_train =to_categorical(train['stars'])
#####################
# Test data
list_tokenized_test = tokenizer.texts_to_sequences(reviews_test)
x_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
y_test = to_categorical(test['stars'])

#######################################################################




#####################################################################
# Using pretrained glove vector
#####################################################################
GLOVE_DIR = "/usr4/cs542sp/zzjiang/Data/"
#GLOVE_DIR ="/home/ec2-user/Data/"
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
                            trainable=False)


'''
#Randomly initialized 
embedding_layer = Embedding(len(word_index) + 1,
                            embed_size,
                            input_length=maxlen,
                            trainable=True)

'''

l2_penalty = 0.5
# create the model
model = Sequential()
model.add(embedding_layer)
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',kernel_regularizer=regularizers.l2(l2_penalty)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(32, activation='relu',,kernel_regularizer=regularizers.l2(l2_penalty)))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


batch_size = 512
epochs = 100
history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_data = [x_test,y_test])

score, acc = model.evaluate(x_test,y_test,batch_size = batch_size)
print("Test acc: " , acc)
print("Test score: " , score)

#################################################################
#Save train history as dict 
#################################################################

with open(r"/usr4/cs542sp/zzjiang/History/2_label/baseline_5", "wb") as output_file:
    pickle.dump(history.history, output_file)