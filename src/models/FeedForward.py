from sklearn.model_selection import train_test_split
import numpy
import tensorflow as tf
import tflearn
import utilities
from base import Base


class FeedForwardNet(Base):

    def __init__(self, drug_no):

        super().__init__(drug_no)
        self.model_type = 'FF'
        self.mod_input()

    def mod_input(self):
        pass

    @staticmethod
    def add_deep_layers(net, layer_sizes):

        for idx, layer in enumerate(layer_sizes[0:-1]):
            net = tflearn.fully_connected(net, layer_sizes[idx], activation='prelu')

        out_rnn = tflearn.fully_connected(net, layer_sizes[-1], activation='prelu')

        return out_rnn

    def model(self, layer_size=None, tensorboard_verbose=3, batch_norm=2, learning_rate=0.001):

        input_shape = [None, self.xTr.shape[1]]

        net = tflearn.input_data(shape=input_shape)
        net = tflearn.layers.normalization.batch_normalization(net)

        deep_layers_output = self.add_deep_layers(net, layer_size)
        net = tflearn.layers.normalization.batch_normalization(deep_layers_output)

        if batch_norm > 0:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.dropout(net, 0.3)

        net = tflearn.fully_connected(net, 1, activation='sigmoid')

        if batch_norm > 1:
            net = tflearn.layers.normalization.batch_normalization(net)

        with tf.name_scope("TargetsData"):  # placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y")

        network = tflearn.regression(net,
                                     placeholder=targetY,
                                     optimizer=self.optimizer(learning_rate),
                                     learning_rate=learning_rate,
                                     loss=tflearn.mean_square(net, targetY),
                                     metric=self.accuracy(net, targetY))

        model = tflearn.DNN(network, tensorboard_verbose=tensorboard_verbose)

        self.populate_params(['model_type', 'layer_size', 'tensorboard_verbose', 'batch_norm', 'n_layers',
                              'learning_rate', 'drug_no'], [self.model_type, layer_size, tensorboard_verbose, batch_norm,
                                                            len(layer_size), learning_rate, self.drug_no])
        return model



if __name__ == '__main__':

    DRUG_ID = 152
    pipe = FeedForwardNet(DRUG_ID)

    mymodel = pipe.model(layer_size=[700, 500, 200],
                         tensorboard_verbose=1,
                         batch_norm=1,
                         learning_rate=0.0001)

    trained_model = pipe.train(mymodel,
                               num_epochs=40,
                               batch_size=80,
                               validation_set=0.1)

    df, r = pipe.predict(trained_model)

    print(df)
    print(r)




