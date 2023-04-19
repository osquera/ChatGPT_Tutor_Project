from keras.models import Sequential
from keras.layers import Dense
import sklearn
import numpy as np
import requests

class CommitteeMember:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(128, input_dim=768*2, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=25, batch_size=64)

def task(args):
    id, Xtrain, ytrain, Xpool = args
    committee_member = CommitteeMember()
    Xtr,ytr=sklearn.utils.resample(Xtrain,ytrain,stratify=ytrain)
    #Train committee member on Xtr,ytr
    committee_member.train(np.asarray(list(Xtr)).astype('float32'), np.array([np.array(xi) for xi in ytr]))
    #predict
    return committee_member.model.predict(np.asarray(list(Xpool.values)).astype('float32'))