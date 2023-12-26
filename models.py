def train_cnn(X_train,y_train,epoch,model_name,path_savedmodel=None):
    dim=X_train.shape[1]
    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', input_shape=(dim,dim, 1), padding='same'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(), metrics=['accuracy'])

    if path_savedmodel is None:
        checkpoint = ModelCheckpoint(model_name, monitor='accuracy', verbose=1,
        save_best_only=True, mode='max', save_format='tf')
        history = model.fit(X_train, y_train, epochs=epoch, batch_size=32, callbacks=[checkpoint],verbose=0)
    else:
        model = load_model(path_savedmodel)
    model.fit(X_train, y_train, epochs=200, batch_size=32,verbose=0)
    return model



def train_svm(X, y):
    clf = SVC(gamma='auto')
    clf.fit(X, y)

    return clf

def train_xgboost(X, y):
    clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=1, random_state=0).fit(X, y)

    return clf