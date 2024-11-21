import os
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from Components.preprocess import process
from yaml_loader import load_yaml
from Components.utils import Utility
from models.models import modelList
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import save_model as save_keras_model
from tensorflow.keras.models import load_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adadelta
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from yaml_loader import load_yaml
from keras.layers import Input

class Trainer:
    def __init__(self):
        config = load_yaml("config.yaml")
        self.epochs = config['CNN_model']['epochs']
        self.hidden = config['CNN_model']['hidden_dim']
        self.loss=globals()[config['CNN_model']['loss']]
        self.optimizer=globals()[config['CNN_model']['optimizer']]()
        self.model=modelList()
    def train_cnn(self, X_train_cnn, y_train_cnn,path_savedmodel=None):
        """
        Train a CNN model. If a saved model exists, load it and skip training.
        """
        cnn = self.model.CNN(X_train_cnn.shape[1])
        cnn.fit(
            X_train_cnn, y_train_cnn,
            epochs=self.epochs,
            batch_size=32,
            verbose=0
        )
        return cnn

    def train_svm(self, X_train_ml, y_train_ml):
        """
        Train an SVM model.
        """
        svm = self.model.svm()
        svm.fit(X_train_ml, y_train_ml)
        return svm

    def train_xgboost(self, X_train_ml, y_train_ml):
        """
        Train an XGBoost model.
        """
        xgboost = self.model.xgboost()
        xgboost.fit(X_train_ml, y_train_ml)
        return xgboost
