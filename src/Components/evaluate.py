from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, recall_score, confusion_matrix
from Components.utils import Utility
from sklearn.metrics import multilabel_confusion_matrix


class evaluate:
    def __init__(self):
        pass
    def evaluater(self,model,X_val,y_val,name):
        print('-'*60)
        print(name)
        y_val_pred = model.predict(X_val)
        threshold = 0.5  
        if(name=='cnn'):
            y_val_pred = (y_val_pred > threshold).astype(int)
        
        accuracy = accuracy_score(y_val, y_val_pred)
        print(f'Accuracy: {accuracy:.2f}')

        precision = precision_score(y_val, y_val_pred, average='weighted')
        print(f'Precision: {precision:.2f}')

        recall = recall_score(y_val, y_val_pred, average='weighted')
        print(f'Recall: {recall:.2f}')

        f1 = f1_score(y_val, y_val_pred, average='weighted')
        print(f'F1 Score: {f1:.2f}')

        confusion_mat = multilabel_confusion_matrix(y_val, y_val_pred)
        print(f'Confusion Matrix:\n{confusion_mat}')

        # Utility.evaluate_model(model,X_val,y_val,name)
