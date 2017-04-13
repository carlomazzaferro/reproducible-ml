import numpy
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split
from utils import definitions
import os


from utils import utilities


class Base(object):

    def __init__(self, data_file):

        self.params = {}
        self.data_file = data_file
        self.data_path = os.path.join(definitions.DATA_DIR, 'processed/')
        self.res_dir = os.path.join(definitions.RESUTLS_DIR, 'model_outputs/')
        self.models_dir = definitions.MODELS_DIR
        self.x_train, self.x_test, self.y_train, self.y_test = self.generate_train_test()

    def generate_train_test(self):
        x, y = utilities.clean_data(self.data_file)
        return train_test_split(x, y)

    def populate_params(self, k, v):
        for val, idx in enumerate(k):
            self.params[idx] = v[val]

    @staticmethod
    def optimizer(lr):
        opt = tflearn.RMSProp(learning_rate=lr)  # , decay=0.9)
        return opt

    @staticmethod
    def loss_func(y_pred, y_true):
        return tf.reduce_mean(tf.square(y_pred - y_true))

    @staticmethod
    def accuracy(y_pred, y_true):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))))

    def train(self, model, num_epochs=20, batch_size=64, validation_set=0.1):

        run_id = utilities.assign_id(self.params, self.res_dir)

        model.fit(self.x_train, self.y_train,
                  n_epoch=num_epochs,
                  validation_set=validation_set,
                  batch_size=batch_size,
                  shuffle=True,
                  run_id=run_id)

        if run_id is None:
            run_id = str(numpy.random.randint(50000))

        self.populate_params(['n_epochs', 'validation_set', 'batch_size', 'data_dir', 'run_id'],
                             [num_epochs, validation_set, batch_size, self.res_dir, run_id])

        print("Done!")
        return model

    def predict(self, model):

        res = model.predict(self.x_test)
        as_df, fpr, tpr, roc_auc, plot = utilities.scoring(res, self.params['model_type'], self.yTe)
        self.populate_params(['roc_auc'], [roc_auc])
        utilities.save_results_to_file(self.params, as_df, plot, fpr, tpr)

        return as_df, roc_auc
