import numpy as np
from yaml_loader import load_yaml
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model as save_keras_model
import joblib
from tensorflow.keras.models import load_model
import pickle
from sklearn.ensemble import GradientBoostingClassifier
class Utility:
    def __init__(self):
        self.config=load_yaml("config.yaml")

    def split_cnn(self,df):

        X = df.iloc[:, :-1].values 
        y = df.iloc[:, -1].values  
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        dim=int((X.shape[1])**(1/2))
        X_train = X_train.reshape(-1, dim,dim, 1)
        X_val=X_val.reshape(-1,dim,dim,1)
        y_train_cat=to_categorical(y_train)
        y_val=to_categorical(y_val)
        return X_train,X_val,y_train_cat,y_val


    def split_ml(self,df):
        X = df.iloc[:, :-1].values 
        y = df.iloc[:, -1].values  
        le = LabelEncoder()
        y=le.fit_transform(y)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train,X_val,y_train,y_val


    def save_model(self,model,model_path):
        """
        Save a trained model to a file.

        Parameters:
        - model (object): Trained model to be saved.
        - model_name (str): Name or identifier for the model.
        - model_path (str): File path to save the model.
        """
        if 'keras' in str(type(model)):  # Check if it's a Keras model
            save_keras_model(model, model_path)
        else:
            # Assume it's a scikit-learn model, you might need to adjust based on the actual ML library used
            joblib.dump(model, model_path)

        # Optionally, you can save additional information or metadata
        # with open(model_path + '_metadata.txt', 'w') as file:
        #     file.write(f"Model Name: {model_name}\n")
        # Add more metadata if needed

    def load_cnn_model(self,path):
        try:
            model = load_model(path)
            print(f"CNN model loaded from {path}")
            return model
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            
    # Load the SVM model (saved as .pkl using pickle)
    def load_svm_model(self,path):
        try:
            with open(path, 'rb') as file:
                model = pickle.load(file)
            print(f"SVM model loaded from {path}")
            return model
        except Exception as e:
            print(f'Error loading SVM model: {e}')
    # Load the XGBoost model (saved as .json)
    def load_xgboost_model(self,path):
        try:
            with open(path, 'rb') as file:
                model = pickle.load(file)
            print(f"GradientBoostingClassifier model loaded from {path}")
            return model
        except Exception as e:
            print(f"Error loading XGBOOST model: {e}")