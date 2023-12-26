def extract_kmers(sequence, k):
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        kmers.append(kmer)
    return kmers


def create_kmer_matrix(fasta_data, all_kmers_set, k):
    num_sequences = len(fasta_data)
    num_unique_kmers = len(all_kmers_set)
    kmer_matrix = np.zeros((num_sequences, num_unique_kmers), dtype=int)

    for i, record in enumerate(fasta_data):
        sequence = str(record.seq).upper()
        kmers = extract_kmers(sequence, k)
        for j, kmer in enumerate(all_kmers_set):
            kmer_matrix[i, j] = kmers.count(kmer)
    return kmer_matrix


def normalize(mRNA_kmer_matrix,lncRNA_kmer_matrix):
    max=mRNA_kmer_matrix.max() if mRNA_kmer_matrix.max()>lncRNA_kmer_matrix.max() else lncRNA_kmer_matrix.max()
    mRNA_kmer_matrix=mRNA_kmer_matrix/max
    lncRNA_kmer_matrix=lncRNA_kmer_matrix/max
    ones_column = np.ones((mRNA_kmer_matrix.shape[0], 1))
    zeros_column=np.zeros((lncRNA_kmer_matrix.shape[0],1))
    mRNA_kmer_matrix = np.hstack((mRNA_kmer_matrix, ones_column))
    lncRNA_kmer_matrix = np.hstack((lncRNA_kmer_matrix, zeros_column))
    df_mrna = pd.DataFrame(mRNA_kmer_matrix)
    df_lncRNA=pd.DataFrame(lncRNA_kmer_matrix)
    return df_mrna,df_lncRNA


def preprocess(mRNA_loc,lncRNA_loc):

    
    mRNA_data = list(SeqIO.parse(mRNA_loc, 'fasta'))
    lncRNA_data = list(SeqIO.parse(lncRNA_loc, 'fasta'))
    combined_sequences = [str(record.seq).upper() for record in mRNA_data] + [str(record.seq).upper() for record in lncRNA_data]
    
    all_kmers_set_1 = set()
    k = 1 
    all_kmers_set_2 = set()
    k = 2  
    all_kmers_set_3 = set()
    k = 3 
    all_kmers_set_4 = set()
    k = 4  
    all_kmers_set_5 = set()
    k = 5  

    for sequence in combined_sequences:
        kmers_1 = extract_kmers(sequence, 1)
        all_kmers_set_1.update(kmers_1)
        kmers_2 = extract_kmers(sequence, 2)
        all_kmers_set_2.update(kmers_2)
        kmers_3 = extract_kmers(sequence, 3)
        all_kmers_set_3.update(kmers_3)
        kmers_4 = extract_kmers(sequence, 4)
        all_kmers_set_4.update(kmers_4)
        kmers_5 = extract_kmers(sequence, 5)
        all_kmers_set_5.update(kmers_5)
        
        
    mRNA_kmer_matrix_1 = create_kmer_matrix(mRNA_data, all_kmers_set_1, 1)
    lncRNA_kmer_matrix_1 = create_kmer_matrix(lncRNA_data, all_kmers_set_1, 1)

    mRNA_kmer_matrix_2 = create_kmer_matrix(mRNA_data, all_kmers_set_2, 2)
    lncRNA_kmer_matrix_2 = create_kmer_matrix(lncRNA_data, all_kmers_set_2, 2)

    mRNA_kmer_matrix_3 = create_kmer_matrix(mRNA_data, all_kmers_set_3, 3)
    lncRNA_kmer_matrix_3 = create_kmer_matrix(lncRNA_data, all_kmers_set_3, 3)

    mRNA_kmer_matrix_4 = create_kmer_matrix(mRNA_data, all_kmers_set_4, 4)
    lncRNA_kmer_matrix_4 = create_kmer_matrix(lncRNA_data, all_kmers_set_4, 4)

    mRNA_kmer_matrix_5 = create_kmer_matrix(mRNA_data, all_kmers_set_5, 5)
    lncRNA_kmer_matrix_5 = create_kmer_matrix(lncRNA_data, all_kmers_set_5, 5)
    df_mrna_1,df_lncRNA_1 = normalize(mRNA_kmer_matrix_1,lncRNA_kmer_matrix_1)
    df_mrna_2,df_lncRNA_2 = normalize(mRNA_kmer_matrix_2,lncRNA_kmer_matrix_2)
    df_mrna_3,df_lncRNA_3 = normalize(mRNA_kmer_matrix_3,lncRNA_kmer_matrix_3)
    df_mrna_4,df_lncRNA_4 = normalize(mRNA_kmer_matrix_4,lncRNA_kmer_matrix_4)
    df_mrna_5,df_lncRNA_5 = normalize(mRNA_kmer_matrix_5,lncRNA_kmer_matrix_5)
    
    df1=pd.concat([df_mrna_1, df_lncRNA_1], ignore_index=True)
    df2=pd.concat([df_mrna_2, df_lncRNA_2], ignore_index=True)
    df3=pd.concat([df_mrna_3, df_lncRNA_3], ignore_index=True)
    df4=pd.concat([df_mrna_4, df_lncRNA_4], ignore_index=True)
    df5=pd.concat([df_mrna_5, df_lncRNA_5], ignore_index=True)
    
    return df1,df2,df3,df4,df5



def split_cnn(df):

    X = df.iloc[:, :-1].values 
    y = df.iloc[:, -1].values  
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    dim=int((X.shape[1])**(1/2))
    X_train = X_train.reshape(-1, dim,dim, 1)
    X_val=X_val.reshape(-1,dim,dim,1)
    y_train_cat=to_categorical(y_train)
    y_val=to_categorical(y_val)
    return X_train,X_val,y_train_cat,y_val



def split_ml(df):
    X = df.iloc[:, :-1].values 
    y = df.iloc[:, -1].values  
    le = LabelEncoder()
    y=le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train,X_val,y_train,y_val










def plot_cnn(model,X_val,y_val):
    print('-'*60)
    print('cnn')
    y_val_pred = model.predict(X_val)
    threshold = 0.5  
    y_val_pred = (y_val_pred > threshold).astype(int)
    
    fpr, tpr, _ = roc_curve(label_binarize(y_val, classes=[0, 1]).ravel(), y_val_pred.ravel())
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

    # Plot Confusion Matrix for each class
    num_classes = confusion_mat.shape[0]
    for i in range(num_classes):
        plt.subplot(1, 2, 2)
        plt.imshow(confusion_mat[i], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Class {i}')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()


def evaluate_cnn(model,X_val,y_val):
    print('-'*60)
    print('cnn')
    y_val_pred = model.predict(X_val)
    threshold = 0.5  
    y_val_pred = (y_val_pred > threshold).astype(int)
    
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Accuracy: {accuracy:.2f}')

    precision = precision_score(y_val, y_val_pred, average='micro')
    print(f'Precision: {precision:.2f}')

    recall = recall_score(y_val, y_val_pred, average='micro')
    print(f'Recall: {recall:.2f}')

    f1 = f1_score(y_val, y_val_pred, average='micro')
    print(f'F1 Score: {f1:.2f}')

    confusion_mat = multilabel_confusion_matrix(y_val, y_val_pred)
    print(f'Confusion Matrix:\n{confusion_mat}')


def plot_ml(model,X_val,y_val,name):
    print('-'*60)
    print(name)
    y_val_pred = model.predict(X_val)

    fpr, tpr, _ = roc_curve(label_binarize(y_val, classes=[0, 1]).ravel(), y_val_pred.ravel())
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
    
    # Plot Confusion Matrix for each class
    num_classes = confusion_mat.shape[0]
    for i in range(num_classes):
        plt.subplot(1, 2, 2)
        plt.imshow(confusion_mat[i], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - Class {i}')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
    print('\n')

def evaluate_ml(model,X_val,y_val,name):
    print('-'*60)
    print(name)
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Accuracy: {accuracy:.2f}')

    precision = precision_score(y_val, y_val_pred, average='micro')
    print(f'Precision: {precision:.2f}')

    recall = recall_score(y_val, y_val_pred, average='micro')
    print(f'Recall: {recall:.2f}')

    f1 = f1_score(y_val, y_val_pred, average='micro')
    print(f'F1 Score: {f1:.2f}')

    confusion_mat = multilabel_confusion_matrix(y_val, y_val_pred)
    print(f'Confusion Matrix:\n{confusion_mat}')





def save_model(model,model_path):
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
