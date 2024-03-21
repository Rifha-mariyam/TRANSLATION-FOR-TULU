# %%
import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from gtts import gTTS
import os




warnings.filterwarnings('ignore')


# %matplotlib inline

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)

# Function to preprocess spectrogram image
def preprocess_spectrogram(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='kn')
    tts.save(filename)
    os.system(f'start {filename}')

def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        print('input_file', input_file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)

'''
# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/background" directory.
"""

# %%
# create_pngs_from_wavs('Sounds/Aslan', 'Spectrograms/Aslan')

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/chainsaw" directory.
"""

# %%
# create_pngs_from_wavs('Sounds/Esek', 'Spectrograms/Esek')

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/engine" directory.
"""

# %%
# create_pngs_from_wavs('Sounds/Inek', 'Spectrograms/Inek')

# %%
"""
Create PNG files containing spectrograms from all the WAV files in the "Sounds/storm" directory.
"""

# %%
# create_pngs_from_wavs('Sounds/Kedi', 'Spectrograms/Kedi')

# create_pngs_from_wavs('Sounds/Kopek', 'Spectrograms/Kopek')

# create_pngs_from_wavs('Sounds/Koyun', 'Spectrograms/Koyun')

# create_pngs_from_wavs('Sounds/Kurbaga', 'Spectrograms/Kurbaga')

# create_pngs_from_wavs('Sounds/Kus', 'Spectrograms/Kus')

# create_pngs_from_wavs('Sounds/Maymun', 'Spectrograms/Maymun')

#create_pngs_from_wavs('Sounds/Tavuk', 'Spectrograms/Tavuk')

# %%
"""
Define two new helper functions for loading and displaying spectrograms and declare two Python lists — one to store spectrogram images, and another to store class labels.
"""

# %%
from keras.preprocessing import image


def load_images_from_path(path, label):
    images = []
    labels = []

    for file in os.listdir(path):
        images.append(image.img_to_array(image.load_img(os.path.join(path, file), target_size=(224, 224, 3))))
        labels.append((label))

    return images, labels


def show_images(images):
    fig, axes = plt.subplots(1, 8, figsize=(20, 20), subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i] / 255)


x = []
y = []

# %%
"""
Load the background spectrogram images, add them to the list named `x`, and label them with 0s.
"""

# %%
images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/and vante vante barpund", 0)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/bukka vishesha", 1)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/eer_doora_povondullar", 2)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/enk badvondund", 3)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/ini enk mast sustavondund", 4)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/maatergla solmelu", 5)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/mast samaya aand tudu", 6)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/tulu barpunde", 7)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/vanas ande", 8)
x += images
y += labels

images, labels = load_images_from_path("C:/Users/USER/Desktop/Project4U-Intern/Tulu_Project/spectrogram_tulu/yan kudlad baide", 9)
x += images
y += labels

# %%
"""
Split the images and labels into two datasets — one for training, and one for testing. Then divide the pixel values by 255 and one-hot-encode the labels using Keras's [to_categorical](https://keras.io/api/utils/python_utils/#to_categorical-function) function.
"""

# %%
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)

x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

# %%
"""
## Build and train a CNN

State-of-the-art image classification typically isn't done with traditional neural networks. Rather, it is performed with convolutional neural networks that use [convolution layers](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/) to extract features from images and [pooling layers](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/) to downsize images so features can be detected at various resolutions. The next task is to build a CNN containing a series of convolution and pooling layers for feature extraction, a pair of fully connected layers for classification, and a `softmax` layer that outputs probabilities for each class, and to train it with spectrogram images and labels. Start by defining the CNN.
"""

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# %%
"""
Train the CNN and save the `history` object returned by `fit` in a local variable.
"""

# %%
hist = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), batch_size=10, epochs=20)
model.save('new_audio_1.h5')
# %%
"""
Plot the training and validation accuracy.
"""

# %%
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
plt.show()


# %%
"""
The accuracy is decent given that the network was trained with just 280 images, but it might be possible to achieve higher accuracy by employing transfer learning.
"""

# %%
"""
## Use transfer learning to improve accuracy

[Transfer learning](https://towardsdatascience.com/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a) is a powerful technique that allows sophisticated CNNs trained by Google, Microsoft, and others on GPUs to be repurposed and used to solve domain-specific problems. Many pretrained CNNs are available in the public domain, and several are included with Keras. Let's use [`MobileNetV2`](https://keras.io/api/applications/mobilenet/), a pretrained CNN from Google that is optimized for mobile devices, to extract features from spectrogram images.

> `MobileNetV2` requires less processing power and has a smaller memory footprint than CNNs such as `ResNet50V2`. That's why it is ideal for mobile devices. You can learn more about it in the [Google AI blog](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html).

Start by calling Keras's [MobileNetV2](https://keras.io/api/applications/mobilenet/) function to instantiate `MobileNetV2` without the classification layers. Use the [preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/preprocess_input) function for `MobileNet` networks to preprocess the training and testing images. Then run both datasets through `MobileNetV2` to extract features.
"""

# %%

# %%
"""
Run the test images through the network and use a confusion matrix to assess the results.
"""


# %%
'''
from sklearn.metrics import confusion_matrix
import seaborn as sns

sns.set()
model = load_model('new_audio.h5')
create_spectrogram("C:/Users/User/Downloads/Tulu_Project/audios_tulu/bukka vishesha/APMC 86.wav", 'Spectrograms/new_sample13.png')
create_spectrogram("C:/Users/User/Downloads/Tulu_Project/audios_tulu/tulu barpunde/APMC 32.wav", 'Spectrograms/new_sample2.png')

# Preprocess and predict
preprocessed_image1 = preprocess_spectrogram('Spectrograms/new_sample13.png')
preprocessed_image2 = preprocess_spectrogram('Spectrograms/new_sample2.png')

predictions1 = model.predict(preprocessed_image1)
predictions2 = model.predict(preprocessed_image2)

class_labels = ['and vante vante barpund', 'bukka vishesha', 'eer_doora_povondullar', 'enk badvondund', 'ini enk mast sustavondund', 'maatergla solmelu', 'mast samaya aand tudu', 'tulu barpunde', 'vanas ande', 'yan kudlad baide']

# Print predictions for the first audio file
# Print predictions for the first audio file
# Get the label with the maximum average value for the first audio file
# Get the label with the maximum average value for the first audio file
max_label1 = class_labels[np.argmax(np.mean(predictions1, axis=0))]

# Get the label with the maximum average value for the second audio file
max_label2 = class_labels[np.argmax(np.mean(predictions2, axis=0))]

# Print the results
print("First audio file")
print(f'Max Average Label: {max_label1}')

# ... (Previous code remains unchanged)

# Check the max average label for the first audio file
if max_label1 == 'and vante vante barpund':
    speech_text = "Haudu svalpa svalpa bartaḍē"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'bukka vishesha':
    speech_text = "Matthe Vishesha"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'eer_doora_povondullar':
    speech_text = "Nivu ellige hoguttiddira"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'enk badvondund':
    speech_text = "Nanage Hasivagide"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'ini enk mast sustavondund':
    speech_text = "Indu nanu tumba daṇididdene"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'maatergla solmelu':
    speech_text = "Maatergla Solmelu"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'mast samaya aand tudu':
    speech_text = "Tumba Dina Aaiythu Nodadhe"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'tulu barpunde':
    speech_text = "Tulu Barthadha"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'vanas ande':
    speech_text = "Uta Aitha"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')
elif max_label1 == 'yan kudlad baide':
    speech_text = "Nanu kudladindha bande"
    print(speech_text)
    text_to_speech(speech_text, 'speech1.mp3')

os.system('start speech1.mp3')


'''
print("\nSecond audio file")
print(f'Max Average Label: {max_label2}')

# Check the max average label for the second audio file
if max_label2 == 'and vante vante barpund':
    speech_text = "Haudu svalpa svalpa bartaḍē"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'bukka vishesha':
    speech_text = "Matthe Vishesha"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'eer_doora_povondullar':
    speech_text = "Nivu ellige hoguttiddira"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'enk badvondund':
    speech_text = "Nanage Hasivagide"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'ini enk mast sustavondund':
    speech_text = "Indu nanu tumba daṇididdene"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'maatergla solmelu':
    speech_text = "Maatergla Solmelu"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'mast samaya aand tudu':
    speech_text = "Tumba Dina Aaiythu Nodadhe"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'tulu barpunde':
    speech_text = "ನಿನಗೆ ತುಳು ಗೊತ್ತಾ"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'vanas ande':
    speech_text = "Uta Aitha"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')
elif max_label2 == 'yan kudlad baide':
    speech_text = "Nanu kudladindha bande"
    print(speech_text)
    text_to_speech(speech_text, 'speech2.mp3')


os.system('start speech2.mp3')

'''