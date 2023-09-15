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


check_point_dir = "/home/lgirardin/SER/male_checkpoints"
data_path = "/home/lgirardin/SER/Data_path.csv"

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
                                                    , test_size=0.25
                                                    , shuffle=True
                                                    , random_state=42
                                                   )

# Lets see how the data present itself before normalisation
print(X_train[150:160])

# Lts do data normalization
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# Check the dataset now
print(X_train[150:160])


# Lets few preparation steps to get it into the correct format for Keras
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# one hot encode the target
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_test = to_categorical(lb.fit_transform(y_test))

print(X_train.shape)
print(lb.classes_)

# Pickel the lb object for future use
filename = 'labels'
outfile = open(filename,'wb')
pickle.dump(lb,outfile)
outfile.close()


X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
print(X_train.shape)

model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns
model.add(Activation('relu'))
# model.add(Conv1D(256, 8, padding='same'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
# model.add(Conv1D(128, 8, padding='same'))
# model.add(Activation('relu'))
# model.add(Conv1D(128, 8, padding='same'))
# model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
# model.add(Conv1D(64, 8, padding='same'))
# model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(6)) # Target class number
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.00001, momentum=0.0, nesterov=False)
# opt = keras.optimizers.Adam(lr=0.0001)
# opt = keras.optimizers.RMSprop(lr=0.000001)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, mode='max')

model_name = "CNN"
checkpoint_filepath = check_point_dir + "/best_saved_" + model_name + "-{epoch:02d}-{val_accuracy:.2f}.h5" # can be hdf5 also
        # print(checkpoint_filepath)
        # quit()
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)
model_history=model.fit(X_train, y_train, batch_size=16, epochs=100, validation_data=(X_test, y_test),callbacks=[early_stopping,model_checkpoint_callback])

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("loss.png")

# Save model and weights
model_name = 'Emotion_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Save model and weights at %s ' % model_path)

# Save the model to disk
model_json = model.to_json()
with open("model_json.json", "w") as json_file:
    json_file.write(model_json)

################### load final model ##########################

# loading json and model architecture
json_file = open('model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("saved_models/Emotion_Model.h5")
print("Loaded model from disk")

# Keras optimiser
opt = keras.optimizers.SGD(lr=0.00001, momentum=0.0, nesterov=False)
# opt = keras.optimizers.Adam(lr=0.00001)
# opt = keras.optimizers.RMSprop(lr=0.000001)
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

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

# Actual labels
actual=y_test.argmax(axis=1)
actual = actual.astype(int).flatten()
actual = (lb.inverse_transform((actual)))
actual = pd.DataFrame({'actualvalues': actual})

# Lets combined both of them into a single dataframe
finaldf = actual.join(preds)
print(finaldf[170:180])

# Write out the predictions to disk
finaldf.to_csv('Predictions.csv', index=False)
finaldf.groupby('predictedvalues').count()


# the confusion matrix heat map plot
def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Gender recode function
def gender(row):
    if row == 'female_disgust' or 'female_fear' or 'female_happy' or 'female_sad' or 'female_surprise' or 'female_neutral':
        return 'female'
    elif row == 'male_angry' or 'male_fear' or 'male_happy' or 'male_sad' or 'male_surprise' or 'male_neutral' or 'male_disgust':
        return 'male'


# Get the predictions file
finaldf = pd.read_csv("Predictions.csv")
classes = finaldf.actualvalues.unique()
classes.sort()

# Confusion matrix
c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)
print(accuracy_score(finaldf.actualvalues, finaldf.predictedvalues))
print_confusion_matrix(c, class_names=classes)


# Classification report
classes = finaldf.actualvalues.unique()
classes.sort()
print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes))

modidf = finaldf
modidf['actualvalues'] = finaldf.actualvalues.replace({'female_angry':'female'
                                       , 'female_disgust':'female'
                                       , 'female_fear':'female'
                                       , 'female_happy':'female'
                                       , 'female_sad':'female'
                                       , 'female_surprise':'female'
                                       , 'female_neutral':'female'
                                       , 'male_angry':'male'
                                       , 'male_fear':'male'
                                       , 'male_happy':'male'
                                       , 'male_sad':'male'
                                       , 'male_surprise':'male'
                                       , 'male_neutral':'male'
                                       , 'male_disgust':'male'
                                      })

modidf['predictedvalues'] = finaldf.predictedvalues.replace({'female_angry':'female'
                                       , 'female_disgust':'female'
                                       , 'female_fear':'female'
                                       , 'female_happy':'female'
                                       , 'female_sad':'female'
                                       , 'female_surprise':'female'
                                       , 'female_neutral':'female'
                                       , 'male_angry':'male'
                                       , 'male_fear':'male'
                                       , 'male_happy':'male'
                                       , 'male_sad':'male'
                                       , 'male_surprise':'male'
                                       , 'male_neutral':'male'
                                       , 'male_disgust':'male'
                                      })

classes = modidf.actualvalues.unique()
classes.sort()

# Confusion matrix
c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)
print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))
print_confusion_matrix(c, class_names = classes)

# Classification report
classes = modidf.actualvalues.unique()
classes.sort()
print(classification_report(modidf.actualvalues, modidf.predictedvalues, target_names=classes))


modidf = pd.read_csv("Predictions.csv")
modidf['actualvalues'] = modidf.actualvalues.replace({'female_angry':'angry'
                                       , 'female_disgust':'disgust'
                                       , 'female_fear':'fear'
                                       , 'female_happy':'happy'
                                       , 'female_sad':'sad'
                                       , 'female_surprise':'surprise'
                                       , 'female_neutral':'neutral'
                                       , 'male_angry':'angry'
                                       , 'male_fear':'fear'
                                       , 'male_happy':'happy'
                                       , 'male_sad':'sad'
                                       , 'male_surprise':'surprise'
                                       , 'male_neutral':'neutral'
                                       , 'male_disgust':'disgust'
                                      })

modidf['predictedvalues'] = modidf.predictedvalues.replace({'female_angry':'angry'
                                       , 'female_disgust':'disgust'
                                       , 'female_fear':'fear'
                                       , 'female_happy':'happy'
                                       , 'female_sad':'sad'
                                       , 'female_surprise':'surprise'
                                       , 'female_neutral':'neutral'
                                       , 'male_angry':'angry'
                                       , 'male_fear':'fear'
                                       , 'male_happy':'happy'
                                       , 'male_sad':'sad'
                                       , 'male_surprise':'surprise'
                                       , 'male_neutral':'neutral'
                                       , 'male_disgust':'disgust'
                                      })

classes = modidf.actualvalues.unique()
classes.sort()

# Confusion matrix
c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)
print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))
print_confusion_matrix(c, class_names = classes)

# Classification report
classes = modidf.actualvalues.unique()
classes.sort()
print(classification_report(modidf.actualvalues, modidf.predictedvalues, target_names=classes))




#
# ################### load best checkpoint ##########################
#
# # loading json and model architecture
# json_file = open('checkpoint_json.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
#
# # load weights into new model
# loaded_model.load_weights("saved_models/Emotion_Model.h5")
# print("Loaded model from disk")
#
# # Keras optimiser
# opt = keras.optimizers.Adam(lr=0.00001)
# # opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
# loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# score = loaded_model.evaluate(X_test, y_test, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
#
# preds = loaded_model.predict(X_test,
#                          batch_size=16,
#                          verbose=1)
# print(preds)
# preds=preds.argmax(axis=1)
# print(preds)
#
# # predictions
# preds = preds.astype(int).flatten()
# preds = (lb.inverse_transform((preds)))
# preds = pd.DataFrame({'predictedvalues': preds})
#
# # Actual labels
# actual=y_test.argmax(axis=1)
# actual = actual.astype(int).flatten()
# actual = (lb.inverse_transform((actual)))
# actual = pd.DataFrame({'actualvalues': actual})
#
# # Lets combined both of them into a single dataframe
# finaldf = actual.join(preds)
# print(finaldf[170:180])
#
# # Write out the predictions to disk
# finaldf.to_csv('Predictions.csv', index=False)
# finaldf.groupby('predictedvalues').count()
#
#
# # the confusion matrix heat map plot
# def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
#     """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
#
#     Arguments
#     ---------
#     confusion_matrix: numpy.ndarray
#         The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
#         Similarly constructed ndarrays can also be used.
#     class_names: list
#         An ordered list of class names, in the order they index the given confusion matrix.
#     figsize: tuple
#         A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
#         the second determining the vertical size. Defaults to (10,7).
#     fontsize: int
#         Font size for axes labels. Defaults to 14.
#
#     Returns
#     -------
#     matplotlib.figure.Figure
#         The resulting confusion matrix figure
#     """
#     df_cm = pd.DataFrame(
#         confusion_matrix, index=class_names, columns=class_names,
#     )
#     fig = plt.figure(figsize=figsize)
#     try:
#         heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
#     except ValueError:
#         raise ValueError("Confusion matrix values must be integers.")
#
#     heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
#     heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# # Gender recode function
# def gender(row):
#     if row == 'female_disgust' or 'female_fear' or 'female_happy' or 'female_sad' or 'female_surprise' or 'female_neutral':
#         return 'female'
#     elif row == 'male_angry' or 'male_fear' or 'male_happy' or 'male_sad' or 'male_surprise' or 'male_neutral' or 'male_disgust':
#         return 'male'
#
#
# # Get the predictions file
# finaldf = pd.read_csv("Predictions.csv")
# classes = finaldf.actualvalues.unique()
# classes.sort()
#
# # Confusion matrix
# c = confusion_matrix(finaldf.actualvalues, finaldf.predictedvalues)
# print(accuracy_score(finaldf.actualvalues, finaldf.predictedvalues))
# print_confusion_matrix(c, class_names=classes)
#
#
# # Classification report
# classes = finaldf.actualvalues.unique()
# classes.sort()
# print(classification_report(finaldf.actualvalues, finaldf.predictedvalues, target_names=classes))
#
# modidf = finaldf
# modidf['actualvalues'] = finaldf.actualvalues.replace({'female_angry':'female'
#                                        , 'female_disgust':'female'
#                                        , 'female_fear':'female'
#                                        , 'female_happy':'female'
#                                        , 'female_sad':'female'
#                                        , 'female_surprise':'female'
#                                        , 'female_neutral':'female'
#                                        , 'male_angry':'male'
#                                        , 'male_fear':'male'
#                                        , 'male_happy':'male'
#                                        , 'male_sad':'male'
#                                        , 'male_surprise':'male'
#                                        , 'male_neutral':'male'
#                                        , 'male_disgust':'male'
#                                       })
#
# modidf['predictedvalues'] = finaldf.predictedvalues.replace({'female_angry':'female'
#                                        , 'female_disgust':'female'
#                                        , 'female_fear':'female'
#                                        , 'female_happy':'female'
#                                        , 'female_sad':'female'
#                                        , 'female_surprise':'female'
#                                        , 'female_neutral':'female'
#                                        , 'male_angry':'male'
#                                        , 'male_fear':'male'
#                                        , 'male_happy':'male'
#                                        , 'male_sad':'male'
#                                        , 'male_surprise':'male'
#                                        , 'male_neutral':'male'
#                                        , 'male_disgust':'male'
#                                       })
#
# classes = modidf.actualvalues.unique()
# classes.sort()
#
# # Confusion matrix
# c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)
# print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))
# print_confusion_matrix(c, class_names = classes)
#
# # Classification report
# classes = modidf.actualvalues.unique()
# classes.sort()
# print(classification_report(modidf.actualvalues, modidf.predictedvalues, target_names=classes))
#
#
# modidf = pd.read_csv("Predictions.csv")
# modidf['actualvalues'] = modidf.actualvalues.replace({'female_angry':'angry'
#                                        , 'female_disgust':'disgust'
#                                        , 'female_fear':'fear'
#                                        , 'female_happy':'happy'
#                                        , 'female_sad':'sad'
#                                        , 'female_surprise':'surprise'
#                                        , 'female_neutral':'neutral'
#                                        , 'male_angry':'angry'
#                                        , 'male_fear':'fear'
#                                        , 'male_happy':'happy'
#                                        , 'male_sad':'sad'
#                                        , 'male_surprise':'surprise'
#                                        , 'male_neutral':'neutral'
#                                        , 'male_disgust':'disgust'
#                                       })
#
# modidf['predictedvalues'] = modidf.predictedvalues.replace({'female_angry':'angry'
#                                        , 'female_disgust':'disgust'
#                                        , 'female_fear':'fear'
#                                        , 'female_happy':'happy'
#                                        , 'female_sad':'sad'
#                                        , 'female_surprise':'surprise'
#                                        , 'female_neutral':'neutral'
#                                        , 'male_angry':'angry'
#                                        , 'male_fear':'fear'
#                                        , 'male_happy':'happy'
#                                        , 'male_sad':'sad'
#                                        , 'male_surprise':'surprise'
#                                        , 'male_neutral':'neutral'
#                                        , 'male_disgust':'disgust'
#                                       })
#
# classes = modidf.actualvalues.unique()
# classes.sort()
#
# # Confusion matrix
# c = confusion_matrix(modidf.actualvalues, modidf.predictedvalues)
# print(accuracy_score(modidf.actualvalues, modidf.predictedvalues))
# print_confusion_matrix(c, class_names = classes)
#
# # Classification report
# classes = modidf.actualvalues.unique()
# classes.sort()
# print(classification_report(modidf.actualvalues, modidf.predictedvalues, target_names=classes))
#
#
