from fastapi import FastAPI, File, UploadFile
import azure.functions as func
from fastapi.responses import JSONResponse
from app import classify_audio_clip2
import os
import io
import librosa 
import torch
import torchaudio
from typing_extensions import Annotated, Doc 

restapi = FastAPI()


@restapi.post("/upload/audio/")
async def upload_audio_file(sample: Annotated[bytes, File()]):
    audio_clip = validateAudioFile(sample)
    result = classify_audio_clip2(audio_clip)
    result = result.item()
    content={"status":"success", 
             "analysis":{
               "detectedVoice": "true",
               "voiceType":  "AI"  if result > 50  else "human"
             },
             "confidenceScore":{
                 "aiProbability": "{0:.2f}".format(result * 100),
                "humanProbability": "{0:.2f}".format(100 - result)
             },
             "additionInfo":{
                 "emotionalTone":"TBD",
                 "backgroundNoiseLevel": "TBD"
             },
             "message": "The uploaded audio is {0:.2f} % likely to be AI generated".format(result)}
    return JSONResponse(status_code=200, content= content)

def validateAudioFile(audioFile,sampling_rate=22000):    
    print("The variable, audioFile is of type:", type(audioFile))
    updatedAudioFile=io.BytesIO(audioFile);
    if isinstance(updatedAudioFile,str):
        if updatedAudioFile.endswith('.mp3'):
            audio, lsr = librosa.load (updatedAudioFile, sr=sampling_rate)
            audio= torch.FloatTensor(audio)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format"})
    if isinstance(updatedAudioFile,io.BytesIO):
            audio, lsr = torchaudio.load (updatedAudioFile)
            audio= audio[0]
    if lsr!=sampling_rate:
        audio=torchaudio.functional.resample(audio,lsr, sampling_rate)
    if torch.any(audio>2) or not torch.any(audio<0):
        return JSONResponse(status_code=400, content={"error": "Invalid/corrupted audio data"})
    audio.clip_(-1,1)

    return audio.unsqueeze(0)
