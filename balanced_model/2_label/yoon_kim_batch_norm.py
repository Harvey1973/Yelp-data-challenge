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
from keras.optimizers import Adam
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
print(train.shape)
print(test.shape)

print(np.unique(train['stars']))

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train['Processed_Reviews\r'])
list_tokenized_train = tokenizer.texts_to_sequences(train['Processed_Reviews\r'])


maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y =train['stars']
#####################
# Test data
tokenizer.fit_on_texts(test['Processed_Reviews\r'])
list_tokenized_test = tokenizer.texts_to_sequences(test['Processed_Reviews\r'])


maxlen = 130
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
y_test = test['stars']
print(np.unique(test['stars']))

#######################################################################


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
                            trainable=True)

#Randomly initialized 
#embedding_layer = Embedding(len(word_index) + 1,
#                            embed_size,
#                            input_length=maxlen,
#                            trainable=False)

#############################################
# Original yoon kim with batch norm and drop out0
#############################################
conv_filters = 128
drop_out_rate = 0.1 + np.random.rand()*0.25
sequence_input = Input(shape=(maxlen,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# Specify each convolution layer and their kernel siz i.e. n-grams 
conv1_1 = Conv1D(filters=conv_filters, kernel_size=2,kernel_regularizer=regularizers.l2(0.01))(embedded_sequences)
#conv1_1 = Conv1D(filters=conv_filters, kernel_size=2)(embedded_sequences)
btch1_1 = BatchNormalization()(conv1_1)
actv1_1 = Activation('relu')(btch1_1)
conv1_2 = Conv1D(filters=conv_filters, kernel_size=2,kernel_regularizer=regularizers.l2(0.01))(actv1_1)
btch1_2 = BatchNormalization()(conv1_2)
actv1_2 = Activation('relu')(btch1_2)
glmp1_1 = MaxPooling1D(pool_size = 4)(actv1_2)

conv2_1 = Conv1D(filters=conv_filters, kernel_size=3,kernel_regularizer=regularizers.l2(0.01))(embedded_sequences)
#conv2_1 = Conv1D(filters=conv_filters, kernel_size=2)(embedded_sequences)
btch2_1 = BatchNormalization()(conv2_1)
actv2_1 = Activation('relu')(btch2_1)
conv2_2 = Conv1D(filters=conv_filters, kernel_size=3,kernel_regularizer=regularizers.l2(0.01))(actv2_1)
btch2_2 = BatchNormalization()(conv2_2)
actv2_2 = Activation('relu')(btch2_2)
glmp2_1 = MaxPooling1D(pool_size = 4)(actv2_2)

conv3_1 = Conv1D(filters=conv_filters, kernel_size=4,kernel_regularizer=regularizers.l2(0.01))(embedded_sequences)
#conv3_1 = Conv1D(filters=conv_filters, kernel_size=2)(embedded_sequences)
btch3_1 = BatchNormalization()(conv3_1)
actv3_1 = Activation('relu')(btch3_1)
conv3_2 = Conv1D(filters=conv_filters, kernel_size=4,kernel_regularizer=regularizers.l2(0.01))(actv3_1)
btch3_2 = BatchNormalization()(conv3_2)
actv3_2 = Activation('relu')(btch3_2)
glmp3_1 = MaxPooling1D(pool_size = 4)(actv3_2)

conv4_1 = Conv1D(filters=conv_filters, kernel_size=5,kernel_regularizer=regularizers.l2(0.01))(embedded_sequences)
#conv4_1 = Conv1D(filters=conv_filters, kernel_size=2)(embedded_sequences)
btch4_1 = BatchNormalization()(conv4_1)
actv4_1 = Activation('relu')(btch4_1)
conv4_2 = Conv1D(filters=conv_filters, kernel_size=5,kernel_regularizer=regularizers.l2(0.01))(actv4_1)
btch4_2 = BatchNormalization()(conv4_2)
actv4_2 = Activation('relu')(btch4_2)
glmp4_1 = MaxPooling1D(pool_size = 4)(actv4_2)

# Gather all convolution layers
cnct = concatenate([glmp1_1, glmp2_1, glmp3_1, glmp4_1], axis=1)
drp = Dropout(drop_out_rate)(cnct)

dns1  = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.01))(drp)
btch1 = BatchNormalization()(dns1)
drp1  = Dropout(drop_out_rate)(btch1)
dns2  = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01))(drp1)
btch2 = BatchNormalization()(dns2)
drp2 = Dropout(drop_out_rate)(btch2)
flat = Flatten()(drp2)
out = Dense(1, activation='sigmoid')(flat)

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model = Model(inputs = sequence_input, outputs=out)
model.compile(optimizer = adam, loss='binary_crossentropy', metrics=['acc'])
model.summary()





batch_size = 512
epochs = 100
history = model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

prediction = model.predict(X_test)
y_pred = (prediction > 0.5).astype(int).reshape(-1,)
#print(y_pred[-200:])
y_test = np.array(y_test)
#print(y_test[-200:])
#print(y_test.shape)
#print(y_pred.shape)
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
print('accuracy :{0}'.format(accuracy_score(y_pred, y_test)))

#################################################################
#Save train history as dict 
#################################################################

with open(r"/usr4/cs542sp/zzjiang/History/2_label/yoon_kim_batch_pre", "wb") as output_file:
    pickle.dump(history.history, output_file)
