import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np

class Plotter:
    def __init__(self):
        pass
    
    def plot_res(self, model, X_val, y_val,plot_name,save_dir='/home/snjay/Desktop/projects/mrna_lncrna_classifier/saved_plots'):
        print('-'*60)
        # print(name)
        
        # Get model predictions
        y_val_pred = model.predict(X_val)
        threshold = 0.5  # Threshold for binary classification
        
        y_val_pred = (y_val_pred > threshold).astype(int)  # Convert continuous to binary predictions
        
        # Binarize y_val if multilabel classification
        y_val_bin = label_binarize(y_val, classes=np.unique(y_val)) if y_val.ndim == 1 else y_val
        
        # Compute ROC curve and AUC for each class
        fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_val_pred.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(12, 6))

        # Plot ROC curve
        lw = 2
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Compute confusion matrix (multilabel handling)
        if y_val_bin.ndim == 2:  # Handle multilabel classification
            confusion_mat = confusion_matrix(y_val_bin.ravel(), y_val_pred.ravel())
            # Plot Confusion Matrix for multilabel classification
            plt.subplot(1, 2, 2)
            plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix (Multilabel)')
            plt.colorbar()
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
        else:  # Handle binary or multiclass classification
            confusion_mat = confusion_matrix(y_val, y_val_pred)  # Now using binary or multiclass labels
            plt.subplot(1, 2, 2)
            plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

        # Show plot and save it if save_dir is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
            save_path = os.path.join(save_dir, f'{plot_name}_roc_confusion.png')
            plt.savefig(save_path)  # Save the plot as a PNG file
            print(f"Plot saved at: {save_path}")
        else:
            plt.show()  # If no directory is specified, just show the plot