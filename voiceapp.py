import streamlit as st
import os
import datetime
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from glob import glob
import io
import librosa 
import tensorflow as tf
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.io.wavfile import read
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.image import resize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback



def load_audio (audiopath, sampling_rate=22000):
    if isinstance(audiopath,str):
        if audiopath.endswith('.mp3'):
            audio, lsr = librosa.load (audiopath, sr=sampling_rate)
            audio= torch.FloatTensor(audio)
        else:
            assert False, "unsupport audio file given"
    if isinstance(audiopath,io.BytesIO):
            audio, lsr = torchaudio.load (audiopath)
            audio= audio[0]
    if lsr!=sampling_rate:
        audio=torchaudio.functional.resample(audio,lsr, sampling_rate)
    if torch.any(audio>2) or not torch.any(audio<0):
        print("error with audio data")
    audio.clip_(-1,1)

    return audio.unsqueeze(0)

def classify_audio_clip(clip):
     classifier = AudioMiniEncoderWithClassifierHead (2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4, 
                                                        resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                        dropout=0, kernel_size=5, distribute_zero_label=False)
     #torch.save(classifier.state_dict(), "classifier.pth")
     state_dict= torch.load ('classifier_new.pth', map_location= torch.device('cpu'))
     print("this is done so far 1")
     classifier.load_state_dict(state_dict)
     print("this is done so far 2")
     clip = clip.cpu().unsqueeze(0)
     results=  F.softmax(classifier(clip),dim=-1)
     return results[0],[0]

def classify_audio_clip2(clip):
    
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    classifier.load_state_dict(torch.load('models/classifier.pth', map_location=torch.device('cpu')))
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]

def save_audio(file):
    if file.size > 4000000:
        return 1
    # if not os.path.exists("audio"):
    #     os.makedirs("audio")
    folder = "audio"
    #datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # clear the folder to avoid storage overload
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    try:
        with open("log0.txt", "a") as f:
            f.write(f"{file.name} - {file.size} ;\n")
    except:
        pass

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


def test_audio(audioclip):
    if audioclip is not None:
                    if not os.path.exists("audio"):
                        os.makedirs("audio")
                    path = os.path.join("audio", audioclip.name)
                    if_save_audio = save_audio(audioclip)
                    if if_save_audio == 1:
                        st.warning("File size is too large. Try another file.")
    # Load and preprocess the audio file
    target_shape = (200, 200)
    model = load_model('audio_classification_model.h5')
    #audioclip.cpu().unsqueeze(0)
    audio_data, sample_rate = librosa.load(path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

    # Make predictions
    predictions = model.predict(mel_spectrogram)
     # Get the class probabilities
    class_probabilities = predictions[0]

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    return class_probabilities, predicted_class_index 

st.set_page_config(layout='wide')


def main():
     st.title("impersonator voice detection Hackathon challenge")
     uploaded_file = st.file_uploader("upload an audio track to identify real or impersonated one", type=["mp3","wav"])
     inference_categories = ['FAKE','REAL']
     if uploaded_file is not None:
          if st.button("analyse mp3 (with small model)"):
               col1, col2, col3 = st.columns(3)

               with col1:
                    st.info ("your results are as below")
                    audio_clip = load_audio(uploaded_file)
                    print("this is done so far 3")
                    result = classify_audio_clip2(audio_clip)
                    print (result)
                    #result = 90
                    result = result.item()
                    st.info(f"Result probability: {result}")
                    st.success(f"The uploaded audio is {result * 100:.2f}% likely to be AI generated")

               with col2:
                    st.info ("your uploaded audio is as below")
                    st.audio(uploaded_file)
                    fig = px.line()
                    fig.add_scatter(x=list(range(len(audio_clip.squeeze()))), y=audio_clip.squeeze())
                    fig.update_layout(
                         title = "Waveform Plot",
                         xaxis_title = "Time",
                         yaxis_title = "Amplitude"
                    )
                    st.plotly_chart(fig, use_container_width=True)

               with col3:
                    st.info ("Disclaimer")
                    st.warning("lets see if this works accurately")
     if uploaded_file is not None:
          if st.button("analyse WAV (with LARGE model)"):
               col1, col2, col3 = st.columns(3)

               with col1:
                    st.info ("your results are as below")
                    #audio_clip = load_audio(uploaded_file)
                    print("this is done so far 3")
                    class_probabilities, predicted_class_index  = test_audio(uploaded_file)
                    # Get the class probabilities
                    # Display results for all classes
                    for i, class_label in enumerate(inference_categories):
                         probability = class_probabilities[i]
                    #print(f'Class: {class_label}, Probability: {probability:.4f}')

                    # Calculate and display the predicted class and accuracy
                    predicted_class = inference_categories[predicted_class_index]
                    accuracy = class_probabilities[predicted_class_index]
                    if accuracy<0.75:
                         predicted_class ='REAL'
                    print(f'The audio is classified as: {predicted_class}')
                    print(f'Accuracy: {accuracy:.4f}')

                    st.info(f"Result probability: {accuracy * 100:.2f}")
                    st.success(f"The uploaded audio is {accuracy * 100:.2f}% likely to be {predicted_class}")

               

if __name__=="__main__":
     main()


                
                    


     