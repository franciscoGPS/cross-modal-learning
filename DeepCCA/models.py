from keras.layers import Dense
#from keras.layers import Concatenate
from keras.utils import plot_model
from keras.layers.merge import concatenate
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.models import Model  
from keras.regularizers import l2
from keras.layers import *
from keras.metrics import top_k_categorical_accuracy
from DeepCCA.losses import cca_loss
import ipdb


def create_model(layer_sizes1, layer_sizes2, input_size1, input_size2, 
                    learning_rate, reg_par, outdim_size, use_all_singular_values):
    """
    builds the whole model
    the structure of each sub-network is defined in build_mlp_net,
    and it can easily get substituted with a more efficient and powerful network like CNN
    """
    view1_model = build_mlp_net(layer_sizes1, input_size1, reg_par)
    view2_model = build_mlp_net(layer_sizes2, input_size2, reg_par)

    model = Sequential()
    ipdb.set_trace()
    ##Merge not working
    model.add(Merge([view1_model, view2_model]))

    model_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=cca_loss(outdim_size, use_all_singular_values), optimizer=model_optimizer)

    return model



def build_model(layer_sizes1, layer_sizes2, input_size1, input_size2,
                    learning_rate, reg_par, outdim_size, use_all_singular_values, neurons):

    input1 = Input(shape=(1, input_size1), name='audios')
    input2 = Input(shape=(input_size2, ), name='lyrics')

    activation_model = 'sigmoid'
    dense1_1 = Dense(neurons, activation=activation_model, name='view_1_1', kernel_regularizer=l2(reg_par))(input1)
    dense1_2 = Dense(neurons, activation=activation_model, name='view_1_2', kernel_regularizer=l2(reg_par))(dense1_1)
    dense1_3 = Dense(neurons, activation=activation_model,  name='view_1_3', kernel_regularizer=l2(reg_par))(dense1_2)
    output1 = Dense(outdim_size, activation='linear', name='view_1_4', kernel_regularizer=l2(reg_par))(dense1_3)

    dense2_1 = Dense(neurons, activation=activation_model,  name='view_2_1', kernel_regularizer=l2(reg_par))(input2)
    dense2_2 = Dense(neurons, activation=activation_model,  name='view_2_2', kernel_regularizer=l2(reg_par))(dense2_1)
    dense2_3 = Dense(neurons, activation=activation_model, name='view_2_3', kernel_regularizer=l2(reg_par))(dense2_2)
    output2 = Dense(outdim_size, activation='linear', name='view_2_4', kernel_regularizer=l2(reg_par))(dense2_3)

    added = Add()([output1, output2])  # equivalent to added = keras.layers.add([x1, x2])

    out = Dense(20)(added)
    model = Model(inputs=[input1, input2], outputs=out)


    model_optimizer = Adam(lr=learning_rate)
    model.compile(loss=cca_loss(outdim_size, use_all_singular_values), optimizer=model_optimizer)

    return model

 

def build_BLSTM_model(input_size1, input_size2, dense_size,
                    learning_rate, reg_par, outdim_size, activation_lstm,
                    use_all_singular_values, neurons_lstm, dropout, 
                    activation_model1, activation_model2, activation_model3,
                    neurons1, neurons2, neurons3 ):

    input1 = Input(shape=(1, input_size1), name='audios')
    input2 = Input(shape=(input_size2, ), name='lyrics')


    dense1_1 = Dense(input_size1, activation=activation_model1, name='view_1_1', kernel_regularizer=l2(reg_par))(input1)
    dense1_2 = Dense(neurons2, activation=activation_model2, name='view_1_2', kernel_regularizer=l2(reg_par))(dense1_1)
    output1 = Dense(outdim_size, activation='linear', name='view_1_4', kernel_regularizer=l2(reg_par))(dense1_2)

    dense2_1 = Dense(input_size2, activation=activation_model1,  name='view_2_1', kernel_regularizer=l2(reg_par))(input2)
    dense2_2 = Dense(neurons2, activation=activation_model2,  name='view_2_2', kernel_regularizer=l2(reg_par))(dense2_1)
    output2 = Dense(outdim_size, activation='linear', name='view_2_4', kernel_regularizer=l2(reg_par))(dense2_2)

    output1_lstm = create_BLSTM_model(output1, neurons_lstm, dropout, reg_par, outdim_size, activation_lstm)
    added = Add()([output1_lstm, output2])  # equivalent to added = keras.layers.add([x1, x2])

    out = Dense(dense_size)(added)
    model = Model(inputs=[input1, input2], outputs=out)

    model_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=cca_loss(outdim_size, use_all_singular_values), optimizer=model_optimizer)

    return model


def build_DRNN_model(input_size1, input_size2, dense_size,
                    learning_rate, reg_par, outdim_size, activation_lstm,
                    use_all_singular_values, neurons_lstm, dropout,
                    activation_model1, activation_model2, activation_model3,
                    neurons1, neurons2, neurons3 ):
    

    input1 = Input(shape=(1, input_size1), name='audios')
    input2 = Input(shape=(input_size2, ), name='lyrics')

    output1_lstm = create_BLSTM_model(input1, input_size1, dropout, reg_par, outdim_size, activation_lstm)
    batchno_1 = BatchNormalization()(output1_lstm)
    dense1_1 = Dense(neurons1, activation=activation_model1, name='view_1_1', kernel_regularizer=l2(reg_par))(output1_lstm)
    batchno_2 = BatchNormalization()(dense1_1)
    dense1_2 = Dense(neurons2, activation=activation_model2, name='view_1_2', kernel_regularizer=l2(reg_par))(batchno_2)
    batchno_3 = BatchNormalization()(dense1_2)
    output1 = Dense(20, activation='linear', name='view_1_4', kernel_regularizer=l2(reg_par))(batchno_3)

    dense2_1 = Dense(input_size2, activation=activation_model1,  name='view_2_1', kernel_regularizer=l2(reg_par))(input2)
    dense2_2 = Dense(neurons2, activation=activation_model2,  name='view_2_2', kernel_regularizer=l2(reg_par))(dense2_1)
    output2 = Dense(20, activation='linear', name='view_2_4', kernel_regularizer=l2(reg_par))(dense2_2)

    added = Add()([output1, output2])  # equivalent to added = keras.layers.add([x1, x2])

    out = Dense(20)(added)
    model = Model(inputs=[input1, input2], outputs=out)

    model_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=cca_loss(outdim_size, False), optimizer=model_optimizer )

    return model

   # create Bidirectional LSTM model
def create_BLSTM_model(input1, hidden_size, dropout, reg_par, outdim_size, activation_lstm):
   
   blstm = Bidirectional(LSTM(hidden_size, stateful=False, activation=activation_lstm, kernel_regularizer=l2(reg_par)))(input1)
   dropout_out = Dropout(dropout)(blstm)
   dense = Dense(outdim_size, activation=activation_lstm, kernel_regularizer=l2(reg_par))(dropout_out)

   return dense



   


def build_GRU_model(input_size1, input_size2, dense_size,
                    learning_rate, reg_par, outdim_size, activation_lstm,
                    use_all_singular_values, neurons_lstm, dropout,
                    activation_model1, activation_model2, activation_model3,
                    neurons1, neurons2, neurons3 ):

    input1 = Input(shape=( 1, input_size1 ), name='audios')
    input2 = Input(shape=(input_size2, ), name='lyrics')

    output1_lstm = create_GRU_RNN(input1, input_size1, dropout, reg_par, outdim_size, activation_lstm)

    dense1_1 = Dense(neurons1, activation=activation_model1, name='view_1_1')(output1_lstm)
    dense1_2 = Dense(neurons2, activation=activation_model2, name='view_1_2')(dense1_1)
    output1 = Dense(60, activation='linear', name='view_1_4', kernel_regularizer=l2(reg_par))(dense1_2)

    dense2_1 = Dense(input_size2, activation=activation_model1,  name='view_2_1')(input2)
    dense2_2 = Dense(neurons2, activation=activation_model2,  name='view_2_2')(dense2_1)
    output2 = Dense(60, activation='linear', name='view_2_4', kernel_regularizer=l2(reg_par))(dense2_2)

    added = Add()([output1, output2])  # equivalent to added = keras.layers.add([x1, x2])

    out = Dense(20)(added)
    model = Model(inputs=[input1, input2], outputs=out)

    model_optimizer = RMSprop(lr=learning_rate)
    model.compile(loss=cca_loss(20, False), optimizer=model_optimizer)

    return model

def create_GRU_RNN(input1, hidden_size, dropout, reg_par, outdim_size, activation_lstm):
   
   blstm = GRU(hidden_size, stateful=False, activation=activation_lstm)(input1)
   dropout_out = Dropout(dropout)(blstm)

   
   return dropout_out



def top_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def create_CNN_RNN(height, width, depth, input_size2, dense_size, 
                    learning_rate, reg_par, outdim_size, activation_lstm,
                    use_all_singular_values, neurons_lstm, dropout,
                    activation_model1, activation_model2, activation_model3,
                    neurons1, neurons2, neurons3 ):

    kernel_size_1 = 3 # we will use 3x3 kernels throughout
    kernel_size_2 = 3
    kernel_size_3 = 2    
    pool_size_1 = 2 
    pool_size_2 = 1 # we will use 2x2 pooling throughout
    conv_depth_1 = 1 # we will initially have 32 kernels per conv. layer...
    conv_depth_2 = 3 # ...switching to 64 after the first pooling layer
    conv_depth_3 = 1
    drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
    drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
    hidden_size = 20 # the FC layer will have 512 neurons

    input1 = Input(shape=(height, width, depth),    name='audios') # depth goes last in TensorFlow back-end (first in Theano)
    
    conv_1 = Convolution2D(conv_depth_1, (kernel_size_1, kernel_size_1), 
     kernel_regularizer=l2(reg_par), padding='same',  activation='softsign')(input1)
    pool_1 = MaxPooling2D(pool_size=(pool_size_1, pool_size_1))(conv_1)
    drop_1 = Dropout(drop_prob_2)(pool_1)
    conv_2 = Convolution2D(conv_depth_2, (kernel_size_2, kernel_size_2),
     kernel_regularizer=l2(reg_par), padding='same', activation='softsign')(drop_1)
    pool_2 = MaxPooling2D(pool_size=(pool_size_2, pool_size_2))(conv_2)
    drop_2 = Dropout(drop_prob_2)(pool_2)
    conv_3 = Convolution2D(conv_depth_3, (kernel_size_3, kernel_size_3),
     kernel_regularizer=l2(reg_par), padding='same', activation='softsign')(drop_2)
    pool_3 = MaxPooling2D(pool_size=(pool_size_1, pool_size_2))(conv_3)
    # Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
    drop_3 = Dropout(drop_prob_1)(pool_3)

    reshape = Reshape((25, 1), input_shape=(5,5,1))(drop_3)
    blstm = Bidirectional(LSTM(25, stateful=False, activation="softsign", return_sequences=True, kernel_regularizer=l2(reg_par)))(reshape)
    blstm_2 = Bidirectional(LSTM(25, stateful=False, activation="softsign", return_sequences=True, kernel_regularizer=l2(reg_par)))(blstm)
       
    flat = Flatten()(blstm_2) 
    output1 = Dense(512, activation='linear', kernel_regularizer=l2(reg_par), name='view_1_3')(flat)


    input2 = Input(shape=(input_size2,), name='lyrics')
    batchno_2 = BatchNormalization()(input2)
    dense2_1 = Dense(512, activation='relu', kernel_regularizer=l2(reg_par), name='view_2_1')(batchno_2)
    dense2_2 = Dense(512, activation='relu', kernel_regularizer=l2(reg_par), name='view_2_2')(dense2_1)
    output2 = Dense(512, activation='linear', kernel_regularizer=l2(reg_par), name='view_2_3')(dense2_2)

    added = Add()([output1, output2])  # equivalent to added = keras.layers.add([x1, x2])

    out = Dense(1024, activation='linear',  name='output')(added)
    
    model = Model(inputs=([input1, input2]), outputs=out)

    model_optimizer = RMSprop(lr=0.001)
    model_optimizer_1 = Adam(lr=0.001)

    model.compile(loss=cca_loss(1024, True), optimizer=model_optimizer_1, metrics=["acc", top_accuracy])  
    
 
    print(model.summary())

    return model

