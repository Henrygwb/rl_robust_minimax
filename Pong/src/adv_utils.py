import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.callbacks import EarlyStopping


def RL_func(input_dim, num_class):
    model = Sequential([
        Dense(64, input_shape=(input_dim,), kernel_initializer='he_normal', \
              kernel_regularizer=regularizers.l2(0.01), name="rl_model/d1"),
        Activation('relu', name="rl_model/r1"),
        Dense(32, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01), name="rl_model/d2"),
        Activation('relu', name="rl_model/r2"),
        Dense(num_class, name="rl_model/d3", kernel_regularizer=regularizers.l2(0.01))
    ])
    return model


class MimicModel(object):

    def __init__(self, sess):
        # todo add gpu support
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        if sess != None:
           self.session = sess
        else:
           self.session = tf.Session(config=tf_config)
        K.set_session(self.session)
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            with self.session.as_default():
                self.model = self.build_model()
        self.session.run(tf.global_variables_initializer())

    def build_model(self):
        model = Sequential([
            Dense(64, input_shape=(13,),name="MnistModel/d1"),
            Activation('relu', name="rl_model/r1"),
            Dense(32, name="rl_model/d2"),
            Activation('relu', name="rl_model/r2"),
            Dense(2, name="rl_model/d3")
        ])

        model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

        return model

    def fit(self, x_train, y_train, batch_size, epoch):

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)

        self.model.fit(x = x_train,
                       y = y_train,
                       batch_size = batch_size,
                       validation_split=0.33,
                       epochs = epoch,
                       callbacks=[early_stopping])
        return 0

    def evaluate(self, x, y, batch_size=32, verbose = 0):
        if x.ndim ==3:
            x = np.expand_dims(x, 0)
        loss, acc = self.model.evaluate(x = x, y = y, batch_size= batch_size, verbose = verbose)
        return loss, acc

    def predict(self, x, batch_size=32, verbose = 0):
        if x.ndim ==3:
            x = np.expand_dims(x, 0)
        pred = self.model.predict(x = x, batch_size = batch_size, verbose = verbose)
        return pred

    def predict_for_RL(self, x):
        pred = self.model.predict(x=x, steps=1)
        return pred

    def save(self, model_url):
        self.model.save(model_url)
        return 0

    def load(self, model_url):
        self.model = load_model(model_url)
        return 0


class GradientExp(object):
    def __init__(self, model, sess=None):
        K.set_learning_phase(0)
        self.model = model

        with self.model.graph.as_default():
            with self.model.session.as_default():
                self.class_grads = K.function([self.model.model.input],
                                              K.gradients(self.model.model.output, self.model.model.input))
                self.out = K.function([self.model.model.input], self.model.model.output)

    def output(self, x):
        with self.model.graph.as_default():
            with self.model.session.as_default():
                out_v = self.out([x])

        return out_v

    def grad(self, x, normalize=True):
        with self.model.graph.as_default():
            with self.model.session.as_default():
                sal_x = self.class_grads([x])[0]
                if normalize:
                    sal_x = np.abs(sal_x)
                    sal_x_max = np.max(sal_x, axis=1)
                    sal_x_max[sal_x_max == 0] = 1e-16
                    sal_x = sal_x / sal_x_max[:, None]
        return sal_x

    def integratedgrad(self, x, x_baseline=None, x_steps=25, normalize=True):
        with self.model.graph.as_default():
            with self.model.session.as_default():

                if x_baseline is None:
                    x_baseline = np.zeros_like(x)
                else:
                    assert x_baseline.shape == x.shape

                x_diff = x - x_baseline
                total_gradients = np.zeros_like(x)

                for alpha in np.linspace(0, 1, x_steps):
                    x_step = x_baseline + alpha * x_diff
                    grads = self.class_grads([x_step])[0]
                    total_gradients += grads
                sal_x = total_gradients * x_diff

                if normalize:
                    sal_x = np.abs(sal_x)
                    sal_x_max = np.max(sal_x, axis=1)
                    sal_x_max[sal_x_max == 0] = 1e-16
                    sal_x = sal_x / sal_x_max[:, None]
        return sal_x

    def smoothgrad(self, x, stdev_spread=0.1, nsamples=25, magnitude=True, normalize=True):

        with self.model.graph.as_default():
            with self.model.session.as_default():

                stdev = stdev_spread * (np.max(x) - np.min(x))
                total_gradients = np.zeros_like(x)

                for i in range(nsamples):
                    noise = np.random.normal(0, stdev, x.shape)
                    x_plus_noise = x + noise
                    grads = self.class_grads([x_plus_noise])[0]

                    if magnitude:
                        total_gradients += (grads * grads)
                    else:
                        total_gradients += grads

                sal_x = total_gradients / nsamples

                if normalize:
                    sal_x = np.abs(sal_x)
                    sal_x_max = np.max(sal_x, axis=1)
                    sal_x_max[sal_x_max == 0] = 1e-16
                    sal_x = sal_x / sal_x_max[:, None]
        return sal_x
