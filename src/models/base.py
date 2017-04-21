import numpy
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split
from utils import definitions
import os
import pandas


from utils import utilities


class Project(object):

    types = ['classification', 'regression']

    def __init__(self, problem_type, num_classes):

        self.params = {}
        self.raw_data_files = os.listdir(os.path.join(definitions.DATA_DIR, 'raw/'))
        self.num_classes = num_classes
        self.problem_type = problem_type
        if num_classes > 0 and problem_type == 'regression':
            raise ValueError('Num classes must be left unfilled if you are working on a regression problem')
        print('Available data to be preprocessed: %s' % "".join(self.raw_data_files))

    def populate_params(self, k, v):
        for val, idx in enumerate(k):
            self.params[idx] = v[val]


class Base(Project):

    def __init__(self, problem_type='multiclass', num_classes=0):

        super().__init__(problem_type, num_classes)

        self.data_path = os.path.join(definitions.DATA_DIR, 'processed/')
        self.res_dir = os.path.join(definitions.RESUTLS_DIR, 'model_outputs/')
        self.models_dir = definitions.MODELS_DIR
        self.x_train, self.x_test, self.y_train, self.y_test = (None, None, None, None)

    @staticmethod
    def generate_train_test(data_file):
        """ define how to clean your data in the utilities module """

        x, y = utilities.clean_data(data_file)
        x_train, x_test, y_train, y_test = train_test_split(x, y)

        x_train.to_csv(os.path.join(definitions.DATA_DIR, 'processed/xtrain.csv'))
        x_test.to_csv(os.path.join(definitions.DATA_DIR, 'processed/xtest.csv'))
        y_train.to_csv(os.path.join(definitions.DATA_DIR, 'processed/xtest.csv'))
        y_test.to_csv(os.path.join(definitions.DATA_DIR, 'processed/xtest.csv'))

        return 'Data now accessible at %s as csv files: xtrain.csv, xtest.csv, ytrain.csv,' \
               ' ytest.csv' % os.path.join(definitions.DATA_DIR, 'processed')

    def load_data(self):
        """ loader function, customize as needed """

        self.x_train = pandas.read_csv(os.path.join(definitions.DATA_DIR, 'processed/xtrain.csv'))
        self.x_test = pandas.read_csv(os.path.join(definitions.DATA_DIR, 'processed/xtest.csv'))
        self.y_train = pandas.read_csv(os.path.join(definitions.DATA_DIR, 'processed/xtest.csv'))
        self.y_test = pandas.read_csv(os.path.join(definitions.DATA_DIR, 'processed/xtest.csv'))

        return 'Data now accessible as class attribute'

    @staticmethod
    def optimizer(lr):
        """ define your optimizer here """
        opt = tflearn.RMSProp(learning_rate=lr)  # , decay=0.9)
        return opt

    @staticmethod
    def loss_func(y_pred, y_true):
        """ define your loss function here here """
        loss = tf.reduce_mean(tf.square(y_pred - y_true))
        return loss

    @staticmethod
    def accuracy(y_pred, y_true):
        """ define your accuracy measure here """
        acc = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true))))
        return acc

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
        as_df, fpr, tpr, roc_auc, plot = utilities.scoring(res, self.params['model_type'], self.y_test)
        self.populate_params(['roc_auc'], [roc_auc])
        utilities.save_results_to_file(self.params, as_df, plot, fpr, tpr)

        return as_df, roc_auc


