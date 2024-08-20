import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_config
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank

tf.keras.utils.set_random_seed(1234)

def build_kapre_model(summary=None):
    fs = 22050
    winlen = 512
    winstep = 64
    nfft = 512  # 1024 default 
    # preemph = 0.5 # not uded here
    mel_filt = 40
    input_shape = (2*fs, 1) # two second records
    EPS = np.finfo(np.float64).eps
    scale = 27 # original value on biophony tutorial 33.15998, there is no explanation

    kwargs = {'sample_rate': fs, 
            'n_freq': nfft // 2 + 1,
            'n_mels': mel_filt,
            'f_min': 0.0,
            'f_max': None,
            'htk': False,
            'norm': 'slaney'
            }

    # Kapre layers
    kapre_model = tf.keras.models.Sequential()
    kapre_model.add(layers.Normalization(axis=-1, input_shape=input_shape))
    kapre_model.add(layers.GaussianNoise(stddev=1e-5))
    kapre_model.add(STFT(n_fft=nfft, win_length=winlen, hop_length=winstep, input_shape=input_shape, window_name='hann_window', 
                        input_data_format='channels_last', output_data_format='channels_last'))
    kapre_model.add(Magnitude())
    kapre_model.add(ApplyFilterbank(type='mel', filterbank_kwargs=kwargs, data_format='channels_last'))

    kapre_model.add(layers.Lambda(lambda x: tf.transpose(x, perm=[0,2,1,3]), name='transpose'))
    # kapre_model.add(layers.Lambda(lambda x: tf.math.log(x+EPS), name='log')) # parece que melhora sem esta trasformação
    # kapre_model.add(layers.Lambda(lambda x: tf.subtract(x, tf.reduce_mean(x, axis=2, keepdims=True)), name = 'brackground')) # não vi deferença
    # kapre_model.add(layers.Lambda(lambda x: tf.divide(x, scale), name='scale')) # sem isto piora o resultado, esta camada deve ser combinada com a logaritmica

    # kapre_model.build(input_shape=input_shape)
    if summary is not None:
        kapre_model.summary()
    return kapre_model

def build_biophony_model(summary=None):
    load_model = model_from_config(json.load(open('./biophony_model/resources/cmi_mbam01.json', 'r')))
    load_model.load_weights('./biophony_model/resources/cmi_mbam01.h5')

    biophony_model = tf.keras.Sequential()
    for layer in load_model.layers[:-4]:
        biophony_model.add(layer)

    biophony_model.trainable = True
    if summary is not None:
        biophony_model.summary() # visualize the embedding model
    return biophony_model

def build_model(kapre_model, biophony_model, summary=None):
    model = tf.keras.models.Sequential()
    model.add(kapre_model)
    model.add(biophony_model)
    model.add(layers.Dropout(0.1, name='drop1'))
    model.add(layers.Dense(512, activation='LeakyReLU', name='den1')) # com 512 da uma acuracia de 54
    model.add(layers.Dropout(0.1, name='drop2'))
    model.add(layers.Dense(2, activation='softmax', name='den2'))
    
    if summary is not None:
        model.summary()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.trainable = True
    return model

def spectrogram_model(model, summary=None):
    spectrogram_model = tf.keras.models.Sequential()
    # for layer in model.layers[0].layers[:2]:
    for layer in model.layers[0].layers[:4]:
        spectrogram_model.add(layer)
        
    if summary is not None:
        spectrogram_model.summary()

    spectrogram_model.trainable = False
    return spectrogram_model

def shap_model(model, summary=None):
    shap_model = tf.keras.models.Sequential()
    # input_shape = model.layers[0].layers[2].input_shape[1:]
    input_shape = model.layers[0].layers[4].input_shape[1:]
    shap_model.add(layers.Input(shape=input_shape))
    
    # for layer in (model.layers[0].layers[2:] + model.layers[1].layers + model.layers[2:]):
    for layer in (model.layers[0].layers[4:] + model.layers[1].layers + model.layers[2:]):
        shap_model.add(layer)

    shap_model.add(layers.Activation("softmax"))
    shap_model.build((None, input_shape))
    
    if summary is not None:
        shap_model.summary()

    shap_model.trainable = False
    return shap_model