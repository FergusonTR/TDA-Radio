# import libraries
from IPython.display import display, clear_output
import numpy as np
import os

"""
path = 'data/signal_dataset/'

# load training data
print('Loading training data ...')
x_train = np.load(path + 'train/signals.npy')
y_train = np.load(path + 'train/labels.npy')
snr_train = np.load(path + 'train/snrs.npy')
print('Load complete!')
print('\n')

# load validation data
print('Loading validation data ...')
x_val = np.load(path + 'validation/signals.npy')
y_val = np.load(path + 'validation/labels.npy')
snr_val = np.load(path + 'validation/snrs.npy')
print('Load complete!')
print('\n')

# load testing data
print('Loading testing data ...')
x_test = np.load(path + 'test/signals.npy')
y_test = np.load(path + 'test/labels.npy')
snr_test = np.load(path + 'test/snrs.npy')
print('Load complete!')
print('\n')
"""


# import deep learning libraries
import os
import keras
from keras import layers
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.layers import Input, Dropout, Add, Dense
from keras.layers import MaxPooling1D, Reshape, Activation
from keras.layers import BatchNormalization, Flatten, Conv1D
from keras.utils.vis_utils import plot_model

# 1d conv resnet
def residual_stack(x, f):
    # 1x1 conv linear
    #x = Conv1D(f, 1, strides=1, padding='same', data_format='channels_last', activation='linear')(x)
    x = Conv1D(f, 1, strides=1, padding='same', data_format='channels_last')(x)
    x = Activation('linear')(x)

    # residual unit 1
    x_shortcut = x
    #x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last', activation='relu')(x)
    x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(x)
    x = Activation('relu')(x)
    
    #x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last', activation='linear')(x)
    x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(x)
    x = Activation('linear')(x)
    
    # add skip connection
    if x.shape[1:] == x_shortcut.shape[1:]:
        x = Add()([x, x_shortcut])
    else:
        raise Exception('Skip Connection Failure!')

    # residual unit 2
    x_shortcut = x
    #x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last', activation='relu')(x)
    x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(x)
    x = Activation('relu')(x)
    
    #x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last', activation='linear')(x)
    x = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(x)
    x = Activation('linear')(x)
    
    # add skip connection
    if x.shape[1:] == x_shortcut.shape[1:]:
        x = Add()([x, x_shortcut])
    else:
        raise Exception('Skip Connection Failure!')

    # max pooling layer
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x)
    return x

# define resnet model
def ResNet(input_shape, classes):
    # create input tensor
    x_input = Input(input_shape)
    x = x_input

    # residual stack
    num_filters = 32
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)   
    #x = residual_stack(x, num_filters) 
    # x = residual_stack(x, num_filters)

    # output layer
    x = Flatten()(x)
    x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
    x = Dropout(.5)(x)
    x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
    x = Dropout(.5)(x)
    x = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(x)

    # Create model
    model = Model(inputs=x_input, outputs=x)

    return model

# option to save model weights and model history
save_model = True
save_history = False

# create directory for model weights
if save_model is True:
    #weights_path = input("Name model weights directory: ")
    weights_path = "test"
    weights_path = "c:/data/weights/" + weights_path
    isFolder = os.path.isdir(weights_path)
    
    if isFolder == 0:
        try:    
            os.makedirs(weights_path)
        except OSError:
            print("Creation of the directory %s failed" % weights_path)
        else:
            print("Successfully created the directory %s " % weights_path)
        print('\n')
    else:
        print("Folder already exists.")
    
# create directory for model history
if save_history is True:
    #history_path = input("Name model history directory: ")
    history_path= "hist"
    history_path = "c:/data/model_history/" + history_path
    isFolder = os.path.isdir(history_path)
    
    if isFolder == 0:
        try:
            os.makedirs(history_path)
        except OSError:
            print("Creation of the directory %s failed" % history_path)
        else:
            print("Successfully created the directory %s " % history_path)
        print('\n')
    else:
        print("Folder already exists")
        
# reshape input data
#x_train = x_train.reshape([-1, 1024, 2])
#x_val = x_val.reshape([-1, 1024, 2])
#x_test = x_test.reshape([-1, 1024, 2])

# initialize optimizer
adm = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# set number of epochs
#num_epochs = input('Enter number of epochs: ')
num_epochs = "2"
num_epochs = int(num_epochs)

# set batch size
batch = 32

# configure weights save
# filepath= weights_path + "/{epoch}.hdf5"
filepath= weights_path + "/weights.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode="auto")
if save_model is True:
    callbacks_list = [checkpoint]
else:
    callbacks_list = []


# initialize and train model
model = ResNet((1024, 2), 24)
model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print(len(model.layers),"Layers")

#history = model.fit(x_train, y_train, epochs = num_epochs, batch_size = batch, callbacks=callbacks_list, validation_data=(x_val, y_val))

"""
# record model history
train_acc = history.history['acc']
train_loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

if save_history is True:
    # save model history: loss and accuracy
    np.save(history_path + '/train_acc.npy', train_acc)
    np.save(history_path + '/train_loss.npy', train_loss)
    np.save(history_path + '/val_acc.npy', val_acc)
    np.save(history_path + '/val_loss.npy', val_loss)
    print("Model History Saved!")
    print('\n')
"""