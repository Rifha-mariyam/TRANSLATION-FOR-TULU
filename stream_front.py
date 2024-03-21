import streamlit as st
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from gtts import gTTS
import os
import sounddevice as sd
import soundfile as sf
import gdown
from keras.models import load_model

st.title("Tulu Audio Classifier and Speech Synthesis")

# Function to create spectrogram
def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)   
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

# Function for text-to-speech
def text_to_speech(text, filename):
    tts = gTTS(text=text, lang='kn')  # 'kn' for Kannada
    tts.save(filename)


# Define the Google Drive file URL
file_url = 'https://drive.google.com/uc?id=1EXqcob9xchsYfKCUlFjxarKp0HgDwPvu'

# Define the local file path where you want to save the model
model_path = 'model.h5'

# Download the model file from Google Drive
gdown.download(file_url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Now you can use your model

# Load model


# Real-time audio recorder
if st.button("Record a Tulu audio"):
    duration = 10  # Duration in seconds
    fs = 44100  # Sampling frequency
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float64")
    sd.wait()
    sf.write("temp_audio.wav", recording, fs)
    st.success("Recording saved as temp_audio.wav")

    # Create spectrogram
    create_spectrogram("temp_audio.wav", "temp_spectrogram.png")

    # Preprocess spectrogram
    preprocessed_image = preprocess_spectrogram("temp_spectrogram.png")

    # Predict
    predictions = model.predict(preprocessed_image)
    class_labels = ['and vante vante barpund', 'bukka vishesha', 'eer_doora_povondullar', 'enk badvondund', 'ini enk mast sustavondund', 'maatergla solmelu', 'mast samaya aand tudu', 'tulu barpunde', 'vanas ande', 'yan kudlad baide']
    max_label = class_labels[np.argmax(np.mean(predictions, axis=0))]

    # Text for speech synthesis
    if max_label == 'and vante vante barpund':
        speech_text = "Haudu svalpa svalpa bartaḍē"
    elif max_label == 'bukka vishesha':
        speech_text = "Matthe Vishesha"
    elif max_label == 'eer_doora_povondullar':
        speech_text = "Nivu ellige hoguttiddira"
    elif max_label == 'enk badvondund':
        speech_text = "Nanage Hasivagide"
    elif max_label == 'ini enk mast sustavondund':
        speech_text = "Indu nanu tumba daṇididdene"
    elif max_label == 'maatergla solmelu':
        speech_text = "Maatergla Solmelu"
    elif max_label == 'mast samaya aand tudu':
        speech_text = "Tumba Dina Aaiythu Nodadhe"
    elif max_label == 'tulu barpunde':
        speech_text = "Tulu Barthadha"
    elif max_label == 'vanas ande':
        speech_text = "Uta Aitha"
    elif max_label == 'yan kudlad baide':
        speech_text = "Nanu kudladindha bande"

    # Perform text-to-speech
    text_to_speech(speech_text, 'output_speech.mp3')

    # Display results
    st.audio("output_speech.mp3", format='audio/mp3')
    st.success(f"The predicted label is: {max_label}")

# File uploader for pre-recorded audio
uploaded_file = st.file_uploader("Upload a Tulu audio file (.wav)")

if uploaded_file is not None:
    # Save the uploaded file
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getvalue())

    if st.button("Make Prediction"):
        # Create spectrogram
        create_spectrogram("temp_audio.wav", "temp_spectrogram.png")

        # Preprocess spectrogram
        preprocessed_image = preprocess_spectrogram("temp_spectrogram.png")

        # Predict
        predictions = model.predict(preprocessed_image)
        class_labels = ['and vante vante barpund', 'bukka vishesha', 'eer_doora_povondullar', 'enk badvondund', 'ini enk mast sustavondund', 'maatergla solmelu', 'mast samaya aand tudu', 'tulu barpunde', 'vanas ande', 'yan kudlad baide']
        max_label = class_labels[np.argmax(np.mean(predictions, axis=0))]

        # Text for speech synthesis
        if max_label == 'and vante vante barpund':
            speech_text = "Haudu svalpa svalpa bartaḍē"
        elif max_label == 'bukka vishesha':
            speech_text = "Matthe Vishesha"
        elif max_label == 'eer_doora_povondullar':
            speech_text = "Nivu ellige hoguttiddira"
        elif max_label == 'enk badvondund':
            speech_text = "Nanage Hasivagide"
        elif max_label == 'ini enk mast sustavondund':
            speech_text = "Indu nanu tumba daṇididdene"
        elif max_label == 'maatergla solmelu':
            speech_text = "Maatergla Solmelu"
        elif max_label == 'mast samaya aand tudu':
            speech_text = "Tumba Dina Aaiythu Nodadhe"
        elif max_label == 'tulu barpunde':
            speech_text = "Tulu Barthadha"
        elif max_label == 'vanas ande':
            speech_text = "Uta Aitha"
        elif max_label == 'yan kudlad baide':
            speech_text = "Nanu kudladindha bande"

        # Perform text-to-speech
        text_to_speech(speech_text, 'output_speech.mp3')

        # Display results
        st.audio("output_speech.mp3", format='audio/mp3')
        st.success(f"The predicted label is: {max_label}")



