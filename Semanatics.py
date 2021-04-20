from sklearn.utils import shuffle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix



df = pd.read_csv('dinesh_proj/assorted_train.txt', sep=' 	')
# df2 = pd.read_csv('neg_train.txt', sep=' 	')
df.columns = ['review', 'rating']
# df1 = pd.read_csv('pos_train.txt', sep=' 	')
# df2 = pd.read_csv('neg_train.txt', sep=' 	')
# df1.columns = ['review', 'rating']
# df2.columns = ['review', 'rating']
# df = pd.concat([df1, df2])
# df = shuffle(df)
X = df['review']
y = df['rating'].values
# print(X.shape, y.shape)

# df_test = pd.read_csv('test_labelled.txt', sep=' 	')
# df_test.columns = ['review', 'rating']
# X_test = df_test['review']
# y_test = df_test['rating']
# X = tfidf.transform(X_test)
# full_test_pred = adaboost.predict(X_test)
# print(classification_report(full_test_pred, y_test))
# print(confusion_matrix(full_test_pred, y_test))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
tfidf = TfidfVectorizer(decode_error='ignore')
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

print("Adaboost Classifier\n")
adaboost = AdaBoostClassifier(n_estimators=50, learning_rate=1,random_state=0)
adaboost.fit(X_train, y_train)
y_pred_ada = adaboost.predict(X_test)
print(classification_report(y_pred_ada, y_test))
print(confusion_matrix(y_pred_ada, y_test))
y_pred_individual=['This is one of the best movie i have ever seen worth watching and superb action with amazing direction']
y_pred_individual = tfidf.transform(y_pred_individual)
print(tfidf.get_feature_names())
y_pred_individual = adaboost.predict(y_pred_individual)
print(y_pred_individual)
# testr = []
# pred=[]
# with open('textfile.txt','r') as source:
#     for line in source:
#       testr.append(line)


# for i in testr:
#   y_pred_individual=[str(i)]
#   y_pred_individual = tfidf.transform(y_pred_individual)
#   y_pred_individual = adaboost.predict(y_pred_individual)
#   pred.append(y_pred_individual)
  # print(y_pred_individual)

# f = open("test.txt", "w")
# for i in range(0,len(testr)):
#   x=str(int(pred[i]))+"\n"
#   f.write(x)
# f.close()
from sklearn.neighbors import  KNeighborsClassifier
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Bernoulli Naive Bayes\n')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
y_pred_individual = ['''This is the most disastarous movie i have ever seen. the direction of this movie is worst  whereas the acting is just crap''']
y_pred_individual = tfidf.transform(y_pred_individual)
y_pred_individual = classifier.predict(y_pred_individual)
print(y_pred_individual)


#####################################################
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras.callbacks import ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

model_checkpoint = ModelCheckpoint(filepath='dinesh_proj/model.h5', monitor='val_loss', save_best_only=True)
checkpoint = [model_checkpoint]

max_features = 6000
tokenizer = Tokenizer(num_words=max_features, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(df.review.values)
post_seq = tokenizer.texts_to_sequences(df.review.values)
post_seq_padded = pad_sequences(post_seq, maxlen=130)
y = df['rating']
X_train, X_test, y_train, y_test = train_test_split(post_seq_padded, y, test_size=0.2, random_state=404, shuffle=True)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True, activation='tanh')))
model.add(GlobalMaxPool1D())
model.add(Dense(10, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 128
epochs = 3
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=checkpoint)

y_pred = model.predict(X_test)
y_pred = y_pred > 0.5
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))

import  numpy as np
y_pred_test = np.array(['Excellent Phone and excellent service', 'I am a business user who heavily depends on mobile service',
               'Good Phone but bad service', 'One of the worst phones i have ever had',
               'I bought this phone a week ago', 'I am very disappointed with this device',
               'This is a good phone with many good features'])

# y_pred_test = ['Excellent Phone and excellent service', 'I am a business user who heavily depends on mobile service',
#                'Good Phone but bad service', 'One of the worst phones i have ever had',
#                'I bought this phone a week ago', 'I am very disappointed with this device',
#                'This is a good phone with many good features']
y_pred_test = y_pred_test.reshape(-1,1)
y_pred_test = pd.DataFrame(y_pred_test)
y_pred_test.shape
y_pred_test = tfidf.transform(y_pred_test)
y_pred_test = classifier.predict(y_pred_test)
print(y_pred_test)
