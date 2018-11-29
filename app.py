import string
import re
import numpy as np
from tensorflow.contrib import learn

def clean_string(s):
	translator = str.maketrans('', '', string.punctuation)
	s = s.translate(translator)
	s = s.replace('\t', ' ')
	s = s.replace('\n', ' ')
	s = re.sub('[^a-zA-Z0-9 \n\.]', ' ', s)
	s = s.strip()
	s = s.lower()
	s = ' '.join(s.split())
	return s

def load_data_and_labels(positive_data_file, negative_data_file):
    # Load data from files
    positive_examples = list(
        open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(
        open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]


    X = positive_examples + negative_examples
    X = [clean_string(s) for s in X]

    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [X, y]

def data_process():
    X, y = load_data_and_labels("./data/rt-polaritydata/rt-polarity.pos", "./data/rt-polaritydata/rt-polarity.neg")
    max_document_length = max([len(x.split(" ")) for x in X])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    temp_X = np.array(list(vocab_processor.fit_transform(X)))

    X = np.zeros((temp_X.shape[0], temp_X.shape[1], temp_X.shape[1], 1))


    for i in range(len(temp_X)):
        for j in range(len(temp_X[i])):
            X[i, j] = temp_X[i].reshape((temp_X.shape[1], 1))

    print(X.shape)

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = X[shuffle_indices]
    y_shuffled = y[shuffle_indices] 

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.1 * float(len(y)))
    x_train, x_val = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_val = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_val)))
    return x_train, y_train, vocab_processor, x_val, y_val


x_train, y_train, vocab_processor, x_val, y_val = data_process()

print(x_train[1])
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Conv2D, Embedding, Dropout, MaxPooling2D, Flatten
from keras.models import Sequential

def bias_init(shape, dtype='float'):
    return K.constant(value=0.1, shape=shape, dtype=dtype)


def CNN_baseline(x_train, filter_size, input_dim):
    model=Sequential()
    '''
    model.add(Embedding(input_dim=input_dim, output_dim=128,
                        input_length=x_train.shape[0],
                        embeddings_initializer='uniform'))
    '''
    model.add(Conv2D(filters=128, kernel_size=(filter_size,filter_size),strides=(1,1), padding='valid',
                     activation='relu', input_shape=(51,51, 1)))

    model.add(MaxPooling2D(pool_size=(x_train.shape[1]-filter_size+1, 1), strides=(1, 1),padding='valid'))

    model.add(Flatten())

    model.add(Dense(units=2))

    model.add(Dropout(0.5))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    return model

model = CNN_baseline(x_train, 3, len(vocab_processor.vocabulary_))
model.fit(x_train, y_train, batch_size=128, epochs=200, validation_data=(x_val,
                                                                         y_val))

