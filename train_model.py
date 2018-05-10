import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import GlobalMaxPool1D
from keras.models import Model
from keras.callbacks import TensorBoard
import time

now = time.strftime("%c")

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
tensorboard_callback = TensorBoard(log_dir="./logs/train_" + now, histogram_freq=0, write_graph=True,
                                   write_images=False)

train.head()
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]

inp = Input(shape=(maxlen,))  # maxlen=200 as defined earlier

embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(50, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.25)(x)
x = Dense(40, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=1,
          callbacks=[tensorboard_callback])

model.summary()
