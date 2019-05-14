import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
tf.logging.set_verbosity(tf.logging.ERROR)

class Network():
    def __init__(self,num_inputs=8, num_outputs=3, layers=2, neurons=6, learning_rate=0.005, l = 0):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers = layers
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.l = l
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.neurons,input_shape=(self.num_inputs,),activation='relu'))
        for _ in range(self.layers-1):
            model.add(layers.Dense(self.neurons,activation='relu'))            
        model.add(layers.Dense(self.num_outputs,activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=self.learning_rate),loss='mse')                               
        return model

    def predict(self,input):
        return self.model.predict(input)

    def train_on_batch(self,inputs,targets):
        return self.model.train_on_batch(inputs,targets)

    def set_weights(self,weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def save_model(self,filename):
        tf.keras.models.save_model(self.model,filename,overwrite=True)