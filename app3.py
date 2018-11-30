import string
import re
from scipy.sparse import csr_matrix
import numpy as np
from tensorflow.contrib import learn
import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense, Conv2D, Embedding, Dropout, MaxPooling2D, Flatten, Lambda, Input, concatenate
from keras.models import Sequential, Model
from keras.initializers import Constant
from keras import regularizers
from keras.utils.np_utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

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
        open(positive_data_file, "r", encoding='latin-1').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(
        open(negative_data_file, "r", encoding='latin-1').readlines())
    negative_examples = [s.strip() for s in negative_examples]


    X = positive_examples + negative_examples
    X = [clean_string(s) for s in X]

    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [X, y]

def data_process():
	X, y = load_data_and_labels(
	    "./data/rt-polaritydata/rt-polarity.pos", "./data/rt-polaritydata/rt-polarity.neg")
	max_document_length = max([len(x.split(" ")) for x in X])
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	X = np.array(list(vocab_processor.fit_transform(X)))

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
	return x_train, y_train, x_val, y_val, vocab_processor


x_train, y_train, x_val, y_val, vocab_processor= data_process()
# x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_val.shape[1]))
# x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1], x_val.shape[1]))
# t_train=to_categorical(y_train)
#
# t_val=to_categorical(y_val)
input_dim=np.amax(x_train)
# print(x_train.shape)

#
# def bias_init(shape, dtype='float'):
#     return K.constant(value=0.1, shape=shape, dtype=dtype)
#
#
# #


# def CNN_baseline(x_train, filter_size, input_dim):
#     model=Sequential()
#     model.add(Embedding(input_dim=input_dim+1, output_dim=128, input_length=51,
#                         embeddings_initializer='uniform'))
#     #
#     model.add(Lambda(lambda x: K.expand_dims(x, 3)))
#     # #
#     model.add(Conv2D(filters=128, kernel_size=(filter_size[0],filter_size[0]),strides=(1,1), padding='valid',
#                      activation='relu', bias_initializer=Constant(0.1)))
#     model.add(MaxPooling2D(pool_size=(1, x_train.shape[1] - filter_size[0] + 1), strides=(1, 1), padding='valid'))
#
#     model.add(Conv2D(filters=128, kernel_size=(filter_size[1],filter_size[1]),strides=(1,1), padding='valid',
#                      activation='relu', bias_initializer=Constant(0.1)))
#     model.add(MaxPooling2D(pool_size=(1, x_train.shape[1] - filter_size[1] + 1), strides=(1, 1), padding='valid'))
#
#     # model.add(Conv2D(filters=128, kernel_size=(filter_size[2],filter_size[2]),strides=(1,1), padding='valid',
#     #                  activation='relu', bias_initializer=Constant(0.1)))
#     # model.add(MaxPooling2D(pool_size=(1, x_train.shape[1] - filter_size[2] + 1), strides=(1, 1), padding='valid'))
#
#
#     model.add(Flatten())
#
#     model.add(Dense(units=2, kernel_regularizer=regularizers.l2(0.01)))
#
#     model.add(Dropout(0.5))
#
#     model.compile(loss=keras.losses.categorical_crossentropy,
#                   optimizer=keras.optimizers.Adam(lr=0.001),
#                   metrics=['accuracy'])
#     return model
# #

def CNN_baseline(x_train, filter_size):
	input_shape=Input(shape=[x_train.shape[1]])
	embed1 = Embedding(input_dim=input_dim+1, output_dim=128, input_length=51,
                        embeddings_initializer='uniform')(input_shape)

	input = Lambda(lambda x: K.expand_dims(x, 3))(embed1)
	# filte size 3
	conv1 = Conv2D(filters=128, kernel_size=(filter_size[0],filter_size[0]),strides=(1,1), padding='valid',
                     activation='relu', bias_initializer=Constant(0.1))(input)
	pool1 = MaxPooling2D(pool_size=(1, x_train.shape[1] - filter_size[0] + 1), strides=(1, 1), padding='valid')(conv1)

	# filte size 4
	conv2 = Conv2D(filters=128, kernel_size=(filter_size[1], filter_size[1]), strides=(1, 1), padding='valid',
				   activation='relu', bias_initializer=Constant(0.1))(input)
	pool2 = MaxPooling2D(pool_size=(1, x_train.shape[1] - filter_size[1] + 1), strides=(1, 1), padding='valid')(conv2)

	# filte size 5
	conv3 = Conv2D(filters=128, kernel_size=(filter_size[2], filter_size[2]), strides=(1, 1), padding='valid',
				   activation='relu', bias_initializer=Constant(0.1))(input)
	pool3 = MaxPooling2D(pool_size=(1, x_train.shape[1] - filter_size[2] + 1), strides=(1, 1), padding='valid')(conv3)

	# concatenate pooled results
	merged = keras.layers.concatenate([pool1, pool2, pool3], axis=1)
	merged = Flatten()(merged)

	output = Dense(units=2, kernel_regularizer=regularizers.l2(0.01))(merged)

	model = Model(input_shape, output)
	model.compile(loss=keras.losses.categorical_crossentropy,
	                  optimizer=keras.optimizers.Adam(lr=0.001),
	                  metrics=['accuracy'])
	return model


filter_size=[3, 4, 5]
model = CNN_baseline(x_train, filter_size)

# model.summary()
model.fit(x_train, y_train, batch_size=55, epochs=200, validation_data=(x_val,y_val))

