from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from voiceapp import classify_audio_clip2
import os
from io import BytesIO
from tensorflow.keras.models import load_model
import librosa 
import torch
import torchaudio
from typing_extensions import Annotated, Doc
from emotapp import get_mfccs, get_title, save_audio
from datetime import datetime
import numpy as np

restapi = FastAPI()

model = load_model("model3.h5")

# constants


CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']
CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
CAT3 = ["positive", "neutral", "negative"]

COLOR_DICT = {"neutral": "grey",
              "positive": "green",
              "happy": "green",
              "surprise": "orange",
              "fear": "purple",
              "negative": "red",
              "angry": "red",
              "sad": "lightblue",
              "disgust": "brown"}

TEST_CAT = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']
TEST_PRED = np.array([.3, .3, .4, .1, .6, .9, .1])


@restapi.get('/ping')
def health():
    return JSONResponse(status_code=200, content= {"status": "UP"})

@restapi.post("/voice/analyze/")
async def upload_audio_file(sample: UploadFile):
    try:
        start = datetime.now()
        error = None
        if not os.path.exists("audio"):
            os.makedirs("audio")
        path = os.path.join("audio", sample.filename)
        with open(path, "wb+") as file_object:
            file_object.write(sample.file.read())
        # if_save_audio = save_audio(bytes_io)
        emotionResult= decideEmotion(path)
        print("The variable, audioFile is of type:", sample)
        if not sample.size > 0:
            return JSONResponse(status_code=400, content={"error": "Found zero byte file"})
        if not sample.filename.endswith('.mp3'):
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})    
        # bytes_io=convert_upload_file_to_bytes_io(sample)
        audio_clip = validateAudioFile(path)
        
        result = classify_audio_clip2(audio_clip)
        
        result = result.item() * 100
        content={"status":"success", 
                "analysis":{
                "detectedVoice": "true",
                "voiceType":  "AI"  if result > 50  else "human",                
                "confidenceScore":{
                    "aiProbability": result,
                    "humanProbability": 100 - result
                },
                "additionInfo":{
                    "emotionalTone":extractEmotionResult(emotionResult),
                    "backgrouif_save_audiondNoiseLevel": "low"
                },
                },
                "responseTime": (datetime.now() - start).total_seconds() 
                }        
        return JSONResponse(status_code=200, content= content)
    except Exception as e:
        print(e)
        error: repr(e)
    finally: 
        os.remove(path)
        if error is not None:
            return JSONResponse(status_code=200, content= {"error":error})


def removeSpaces(string):
    return string.replace(" ", "")

def extractEmotionResult(emotionText):
    if emotionText:
        emotionText= emotionText.replace("MFCCs\n","")
        emotionText= emotionText.split(":")
        if len(emotionText) > 0:
            emotionText=emotionText[1].split("-")
            return removeSpaces(emotionText[0])
        else:
            return removeSpaces(emotionText[0])
    else:
        return "TBD"
    
def get_mfccs_local(audio, sr, limit):
    a = librosa.feature.mfcc(audio, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, :a.shape[1]] = a
    return mfccs

def decideEmotion(sample): 

    mfccs = get_mfccs(sample, model.input_shape[-1])
    mfccs = mfccs.reshape(1, *mfccs.shape)
    pred = model.predict(mfccs)[0]
    txt = "MFCCs\n" + get_title(pred, CAT6)
    return txt
    
def validateAudioFile(audio_file,sampling_rate=22000):    
    

    audio, lsr = librosa.load (audio_file, sr=sampling_rate)
    audio= torch.FloatTensor(audio)
    if lsr!=sampling_rate:
        audio=torchaudio.functional.resample(audio,lsr, sampling_rate)
    if torch.any(audio>2) or not torch.any(audio<0):
        return JSONResponse(status_code=400, content={"error": "Invalid/corrupted audio data"})
    audio.clip_(-1,1)

    return audio.unsqueeze(0)

async def convert_upload_file_to_bytes_io(upload_file: UploadFile) -> BytesIO:
    file_bytes = await upload_file.read()
    bytes_io = BytesIO(file_bytes)
    bytes_io.name = upload_file.filename
    bytes_io.seek(0)
    return bytes_io
