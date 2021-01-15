import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pickle
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras_metrics as km
from IPython.display import Image
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
 
data_dir = "./video_data"
model_folder = './lstmcnn'

img_height, img_width = 64, 64
seq_len = 1
# "ApplyLipstick", "BabyCrawling", "BalanceBeam", "BandMarching", , "ApplyLipstick", "Archery"]
classes = ["ApplyEyeMakeup", "BenchPress"]
 
#  Creating frames from videos
 
def frames_extraction(video_path):
    frames_list = []
     
    vidObj = cv2.VideoCapture(video_path)
    # Used as counter variable 
    count = 1
 
    while count <= seq_len: 
         
        success, image = vidObj.read() 
        if success:
            image = cv2.resize(image, (img_height, img_width))
            frames_list.append(image)
            count += 1
        else:
            break
 
            
    return frames_list
 
def create_data(input_dir):
    X = []
    Y = []

     
    classes_list = os.listdir(input_dir)
	
     
    for c in classes_list:
        print(c)
        files_list = os.listdir(os.path.join(input_dir, c))
        for f in files_list:
           frames = frames_extraction(os.path.join(os.path.join(input_dir, c), f))
           print(frames)
           if len(frames) == seq_len:
                X.append(frames)
                y = [0]*len(classes)
                y[classes.index(c)] = 1
                Y.append(y)
     
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X, Y

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap
 
X, Y= create_data(data_dir)
print(Y)
 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=0)



model = Sequential()
model.add(TimeDistributed(Conv2D(128, (3, 3), strides=(1,1),activation='relu'),input_shape=(seq_len, img_width, img_height, 3)))
model.add(TimeDistributed(MaxPooling2D(2,2)))
model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1,1),activation='relu')))
model.add(TimeDistributed(MaxPooling2D(2,2)))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Flatten()))
model.add(Dropout(0.1))
model.add(LSTM(32,return_sequences=False,dropout=0.2)) # used 32 units
model.add(Dropout(0.1))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(2, activation = "softmax"))
model.summary()

last_conv_layer_name = "lstm_1"
classifier_layer_names = [
    "dropout_2",
    "dense1",
    "dropout_3",
    "dense2"
]

opt = keras.optimizers.SGD(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), model_folder, 'checkpoints')
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
# filepath_ckp = os.path.join(checkpoint_dir, "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5")
filepath_ckp = os.path.join(checkpoint_dir, "weights-best.hdf5")

# save the best model currently
checkpoint = ModelCheckpoint(
    filepath_ckp,
    monitor='val_loss',
    # monitor='val_acc',
    verbose=2,
    save_best_only=True
    )



#earlystop = EarlyStopping(patience=7)
#callbacks = [earlystop]

history = model.fit(x = X_train, y = y_train, epochs=200, batch_size = 4 , shuffle=True, validation_split=0.4, callbacks=[checkpoint], verbose=2)



plt_dir = os.path.join(os.getcwd(), model_folder, 'plots')
if not os.path.isdir(plt_dir):
    os.makedirs(plt_dir)

print('The ploting starts!\n')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# save the history
history_dir = os.path.join(os.getcwd(), model_folder, 'history')
if not os.path.isdir(history_dir):
    os.makedirs(history_dir)

# wb 以二进制写入
data_output = open(os.path.join(history_dir,'history_Baseline.pkl'),'wb')
pickle.dump(history.history,data_output)
data_output.close()

# rb 以二进制读取
data_input = open(os.path.join(history_dir,'history_Baseline.pkl'),'rb')
read_data = pickle.load(data_input)
data_input.close()

epochs_range = range(len(acc))
plt.plot(epochs_range, acc, 'ro', label='Training acc')
plt.plot(epochs_range, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig(os.path.join(plt_dir, 'acc.jpg'))
plt.figure()

plt.plot(epochs_range, loss, 'ro', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig(os.path.join(plt_dir, 'loss.jpg'))
plt.show()
