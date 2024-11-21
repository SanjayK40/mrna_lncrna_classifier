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
class modelList:
    def __init__(self):
        config=load_yaml("config.yaml")
        self.loss=globals()[config['CNN_model']['loss']]
        self.optimizer=globals()[config['CNN_model']['optimizer']]()
    def CNN(self,input_dim):
        # dim=X_train.shape[1]
        kernel_size=3
        # if(input_dim==2):
        #     kernel_size=2
        model = Sequential()
        model.add(Input(shape=(input_dim, input_dim, 1)))
        model.add(Conv2D(32, kernel_size, activation='relu', input_shape=(input_dim,input_dim, 1), padding='same'))
        model.add(Conv2D(64, kernel_size, activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

        return model

    def svm(self):
        clf = SVC(gamma='auto')

        # clf.fit(X, y)

        return clf

    def xgboost(self):
        clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=1, random_state=0)

        return clf