import pandas as pd
import pickle
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




#######################################################################
# Read in Data and tokenize , prepare training data and test data
#df = pd.read_csv("C:/Users/Harvey/Desktop/Yelp_data_set/restuarant_review_5_label_unbalanced.csv")
#df = pd.read_csv("/home/ec2-user/Data/restuarant_review_5_label_unbalanced.csv")



#df = pd.read_csv("C:/Users/Harvey/Desktop/Yelp_data_set/restuarant_review_balanced.csv"
#df = pd.read_csv("/home/ec2-user/Data/restuarant_review_5_label_unbalanced.csv")
#df = pd.read_csv("/usr4/cs542sp/zzjiang/Data/restuarant_review_5_label_unbalanced.csv")
train= pd.read_csv("/usr4/cs542sp/zzjiang/Data/restuarant_balanced_2_train.csv",lineterminator='\n')
test = pd.read_csv("/usr4/cs542sp/zzjiang/Data/restuarant_balanced_2_test.csv",lineterminator='\n')
reviews_train = train['Processed_Reviews\r']
reviews_test = test['Processed_Reviews\r']
print(train.shape)
print(test.shape)
whole_data = pd.concat([reviews_train,reviews_test])

maxlen = 100
max_features = 15000
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(whole_data)
list_tokenized_train = tokenizer.texts_to_sequences(reviews_train)
x_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
y_train =train['stars']
#####################
# Test data
#tokenizer.fit_on_texts(reviews_test)
list_tokenized_test = tokenizer.texts_to_sequences(reviews_test)
x_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
y_test = test['stars']




#####################################################################
# Using pretrained glove vector
#####################################################################
GLOVE_DIR = "/usr4/cs542sp/zzjiang/Data/"
#GLOVE_DIR ="/home/ec2-user/Data/"
#GLOVE_DIR = "/Users/harvey/Desktop/Data/"
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

#Randomly initialized 
#embedding_layer = Embedding(len(word_index) + 1,
#                            embed_size,
#                            input_length=maxlen,
#                            trainable=True)


##############################
# Orignal Yoon Kim 
##############################

conv_filters = 32
sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# Specify each convolution layer and their kernel siz i.e. n-grams 
conv1_1 = Conv1D(filters=conv_filters, kernel_size=3)(embedded_sequences)
actv1_1 = Activation('relu')(conv1_1)
glmp1_1 = GlobalMaxPooling1D()(actv1_1)

conv1_2 = Conv1D(filters=conv_filters, kernel_size=4)(embedded_sequences)
actv1_2 = Activation('relu')(conv1_2)
glmp1_2 = GlobalMaxPooling1D()(actv1_2)

conv1_3 = Conv1D(filters=conv_filters, kernel_size=5)(embedded_sequences)
actv1_3 = Activation('relu')(conv1_3)
glmp1_3 = GlobalMaxPooling1D()(actv1_3)

conv1_4 = Conv1D(filters=conv_filters, kernel_size=6)(embedded_sequences)
actv1_4 = Activation('relu')(conv1_4)
glmp1_4 = GlobalMaxPooling1D()(actv1_4)

# Gather all convolution layers
cnct = concatenate([glmp1_1, glmp1_2, glmp1_3, glmp1_4], axis=1)
#cnct = concatenate([glmp1_1, glmp1_2, glmp1_3], axis=1)
#cnct = concatenate([glmp1_1, glmp1_2], axis=1)
drp1  = Dropout(0.1)(cnct)
dns1  = Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.1))(drp1)
#drp2  = Dropout(0.5)(dns1)
#dns2 = 
out = Dense(1, activation='sigmoid')(dns1)


model = Model(inputs=sequence_input, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
###############################################################################


batch_size = 512
epochs = 100
history = model.fit(x_train,y_train, batch_size=batch_size, epochs=epochs, validation_data = [x_test,y_test])

score, acc = model.evaluate(x_test,y_test,batch_size = batch_size)
print("Test acc: " , acc)
print("Test score: " , score)

#################################################################
#Save train history as dict 
#################################################################

with open(r"/usr4/cs542sp/zzjiang/History/2_label/yoon_kim_ori_2", "wb") as output_file:
    pickle.dump(history.history, output_file)

