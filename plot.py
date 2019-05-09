
'''
import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

text1='some thing to eat maybe'
text2='some thing to drink'
texts=[text1,text2]
 
print (T.text_to_word_sequence(text1))  #以空格区分，中文也不例外 ['some', 'thing', 'to', 'eat']
print (T.one_hot(text1,10))  #[7, 9, 3, 4] -- （10表示数字化向量为10以内的数字）
print (T.one_hot(text2,10))  #[7, 9, 3, 1]
 
tokenizer = Tokenizer(num_words=None) #num_words:None或整数,处理的最大单词数量。少于此数的单词丢掉
tokenizer.fit_on_texts(texts)
print (( tokenizer.word_counts)) #[('some', 2), ('thing', 2), ('to', 2), ('eat', 1), ('drink', 1)]
print( tokenizer.word_index) #{'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
print( tokenizer.word_docs) #{'some': 2, 'thing': 2, 'to': 2, 'drink': 1,  'eat': 1}
print( tokenizer.index_docs) #{1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
 
# num_words=多少会影响下面的结果，行数=num_words
print( tokenizer.texts_to_sequences(texts)) #得到词索引[[1, 2, 3, 4], [1, 2, 3, 5]]
print( tokenizer.texts_to_matrix(texts))  # 矩阵化=one_hot

maxlen = 4
X_t = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=maxlen)
print(X_t)
'''
import numpy as np
import pickle
import matplotlib.pyplot as plt
#file = open("/Users/harvey/Desktop/report/2_label/cnn/yoon_kim_ori_2_overfit",'rb')
#history = pickle.load(file)
file = open("/Users/harvey/Desktop/report/2_label/cnn/yoon_kim_ori_2",'rb')
history_2 = pickle.load(file)



#history_2['loss'][0] = history_2['loss'][1]
#history['loss'] = np.log(history['loss'])
#history['val_loss'] = np.log(history['val_loss'])

acc_1 = history_2["acc"] # Training accuracy
val_acc = history_2["val_acc"] # Validation accuracy
loss = history_2["loss"] # Training loss
val_loss = history_2["val_loss"] # Validation loss
#acc_2 = history_2["acc"]
epochs = range(1, len(acc_1) + 1) #plots every epoch, here 10
#epochs = range(1,31)
plt.plot(epochs, acc_1, "b", label = "Training accuracy") # "bo" gives dot plot
#plt.plot(epochs, acc_2, "r", label = "Training acc with batch norm") # "bo" gives dot plot
plt.plot(epochs, val_acc, "r", label = "Validation accuracy") # "b" gives line plot
plt.title("Train and validation accuracy")
plt.ylim((0.0,1.0))
plt.legend()

plt.show()

'''
plt.plot(epochs, loss, "b", label = "Training loss")
plt.plot(epochs, val_loss, "r", label = "Validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()
'''