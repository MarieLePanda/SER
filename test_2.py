# Import libraries
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
import IPython.display as ipd  # To play sound in the notebook
import os
import sys
import warnings
# ignore warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Keras
import keras
from keras import regularizers
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
#from keras.utils import np_utils, to_categorical
# https://stackoverflow.com/questions/45149341/importerror-cannot-import-name-np-utils
from keras.utils import to_categorical
#to_categorical(y, num_classes=None) instead of "np_utils.to_categorical(lb.fit_transform(y_train))"
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint, EarlyStopping

# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import json
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob
import os
import pickle


check_point_model = "C:/Users/gprajapati/OneDrive - Microsoft/Hackathon_23/SER/male-female_3class/sgd/male_checkpoints/best_saved_CNN-31-0.58.h5"
data_path = "C:/Users/gprajapati/OneDrive - Microsoft/Hackathon_23/SER/male-female_3class/Data_path_test.csv"
test_results_file = "C:/Users/gprajapati/OneDrive - Microsoft/Hackathon_23/SER/male-female_3class/Data_path_test_sgd.csv"

# lets pick up the meta-data that we got from our first part of the Kernel
ref = pd.read_csv(data_path)
print(ref.head())


# Note this takes a couple of minutes (~10 mins) as we're iterating over 4 datasets
df = pd.DataFrame(columns=['feature'])


# loop feature extraction over the entire dataset
counter = 0
for index, path in enumerate(ref.path):
    X, sample_rate = librosa.load(path
                                  , res_type='kaiser_fast'
                                  , duration=3.0
                                  , sr=44100
                                  , offset=0.5
                                  )
    sample_rate = np.array(sample_rate)

    # mean as the feature. Could do min and max etc as well.
    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=13),axis=0)
    df.loc[counter] = [mfccs]
    counter = counter + 1

# Check a few records to make sure its processed successfully
print(len(df))

print(df.head())

# Now extract the mean bands to its own feature columns
df = pd.concat([ref,pd.DataFrame(df['feature'].values.tolist())],axis=1)

# replace NA with 0
df=df.fillna(0)
print(df.shape)

# Split between train and test
X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','labels','source'],axis=1)
                                                    , df.labels
                                                    , test_size=9
                                                    , shuffle=True
                                                    , random_state=42
                                                   )

print(y_test)
# Lets few preparation steps to get it into the correct format for Keras
# X_train = np.array(X_train)
# y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# print(y_test)
# one hot encode the target
lb = LabelEncoder()
# y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))

# print(X_train.shape)
print(lb.classes_)
# quit()
# Pickel the lb object for future use
filename = 'labels'
outfile = open(filename,'wb')
pickle.dump(lb,outfile)
outfile.close()

X_test = np.expand_dims(X_test, axis=2)

################### load final model ##########################

loaded_model = keras.models.load_model(check_point_model)

print("Loaded model from disk")

# Keras optimiser
opt = keras.optimizers.SGD(lr=0.00001, momentum=0.0, nesterov=False)

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# print('loaded_loss', loaded_loss)
# print('loaded_accuracy', loaded_accuracy)

results = []
prob = []
actual = []
count = 0

 # Actual labels
# actual=y_test.argmax(axis=1)
# actual = actual.astype(int).flatten()
# actual = (lb.inverse_transform((actual)))
# actual = pd.DataFrame({'actualvalues': actual})

for _, sample in enumerate(y_test):
        actual.append(np.array([sample]))

for _, sample in enumerate(X_test):
        results.append(np.argmax(loaded_model.predict(np.array([sample]))))
        prob.append(loaded_model.predict(np.array([sample])))
        print("utt.: ", count+1)
        print("Prediction_probability=%s, Predicted=%s" % (loaded_model.predict(np.array([sample])),np.argmax(loaded_model.predict(np.array([sample])))))
predictions = tuple(results)
probabilities = tuple(prob)

print(y_test)
print(predictions)
quit()

pred = []
prob_class = []

for i in range(len(predictions)):
        # print(i)("neutral", "noise", "negative")
        if predictions[i] == 0:
            pred.append('neutral')
            prob_class.append(probabilities[i])
        elif predictions[i] == 1:
            pred.append('negative')
            prob_class.append(probabilities[i])

# dictionary of lists
dict_1 = {'actual': actual, 'predicted': pred, 'probabilities':prob_class}
df1 = pd.DataFrame(dict_1)
df1.to_excel(test_results_file)

preds = loaded_model.predict(X_test,
                         batch_size=16,
                         verbose=1)
print(preds)
preds=preds.argmax(axis=1)
print(preds)

# predictions
preds = preds.astype(int).flatten()
preds = (lb.inverse_transform((preds)))
preds = pd.DataFrame({'predictedvalues': preds})

#

# Lets combined both of them into a single dataframe
finaldf = actual.join(preds)
print(finaldf[170:180])

# Write out the predictions to disk
finaldf.to_csv('Predictions_test.csv', index=False)
finaldf.groupby('predictedvalues').count()

