import streamlit as st
import os
from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
from glob import glob
import io
import librosa 
import plotly.express as px
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy.io.wavfile import read


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
    """
    Returns whether or not Tortoises' classifier thinks the given clip came from Tortoise.
    :param clip: torch tensor containing audio waveform data (get it from load_audio)
    :return: True if the clip was classified as coming from Tortoise and false if it was classified as real.
    """
    classifier = AudioMiniEncoderWithClassifierHead(2, spec_dim=1, embedding_dim=512, depth=5, downsample_factor=4,
                                                    resnet_blocks=2, attn_blocks=4, num_attn_heads=4, base_channels=32,
                                                    dropout=0, kernel_size=5, distribute_zero_label=False)
    classifier.load_state_dict(torch.load('models/classifier.pth', map_location=torch.device('cpu')))
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]


st.set_page_config(layout='wide')


def main():
     st.title("impersonator voice detection Hackathon challenge")
     uploaded_file = st.file_uploader("upload an audio track to identify real or impersonated one", type=["mp3"])

     if uploaded_file is not None:
          if st.button("analyze audio"):
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


if __name__=="__main__":
     main()


                
                    


     