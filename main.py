# Imports
import os
import time
import re
import torch
import torchaudio
import wave
import pyaudio
from glob import glob #globe is require for batching audio input for STT
import speech_recognition as sr
from openai import OpenAI
import re
from omegaconf import OmegaConf #used by Silaro
from silero.utils import (init_jit_model, split_into_batches, read_audio, read_batch, prepare_model_input)

local_ai = OpenAI(base_url="http://localhost:1234/v1/chat/completions", api_key="not-needed")

def send_to_ai(message):
    mem = [{"role": "system", "content": "You are a super rude ai bot"}]
    mem.append({"role": "user", "content": message})

    completion = local_ai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=mem,
        temperature=0.8,
        frequency_penalty=0.2,
        stop=["<br>"],
    )
    response = completion.choices[0].message.content
    print(f"> {response}")
    return response

#Required variables

audio_input_file = "micAudioInput.wav"
audio_output_file = "ttsAudioOutput.wav"
language = 'en'
speaker = 'en_0'
sample_rate = 48000
CHUNK = 1024
you_said = ""

device = torch.device('cpu')  # you can use any pytorch device

#Initialize Speech Recognizer audio
r_audio = sr.Recognizer() 

p = pyaudio.PyAudio() #for playing speech 

SpeakerStream = p.open(format = 8, channels = 1, rate = sample_rate, output = True)

#initialize Speech to Text model of Silaro  https://github.com/snakers4/silero-models 
stt_model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models', 
                                           model='silero_stt', 
                                           jit_model='jit_xlarge', 
                                           language='en', 
                                           device=device)
stt_model.to(device)


tts_model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models', 
                                         model='silero_tts', 
                                         language=language,
                                         speaker="v3_en")
tts_model.to(device)

# Memory to store the last 5 messages



  
while True:
    loop_start_time = time.time()

    # Step 1 - Take Mic input
    try:
        with sr.Microphone() as SR_AudioSource:
        #r.adjust_for_ambient_noise(source,duration=1.0) # Try these if you need to adjust mic for ambient noise
        #r.energy_threshold = 2000 # Try these to adjust hearing sensitivity - value between 50-4000
            print("Say something...")
            mic_audio = r_audio.listen(SR_AudioSource,1,4) # mic_audio=r_audio.listen(source,timeout=1,phrase_time_limit=4)
            #mic_audio = r_audio.listen(source) #you may also simply call .listen without any parameters 
    except: #This is required if there is no Audio input or some error
        print("Cound not capture mic input or no audio...")
        continue #continue next mic input loop if there was any error
    
    #save captured audio to a file
    with open(audio_input_file, "wb") as file:
        file.write(mic_audio.get_wav_data())
        file.flush()
        file.close()   

    #time log    
    audio_capture_time = time.time()
    print("Audio Capture time = ", audio_capture_time-loop_start_time)

    # Step 2 - Convert recorded Speech to Text
    
    # Check if there is input audio file saved otherwise continue listening 
    if not os.path.exists(audio_input_file):
        print("no input file exists")
        continue
    
    # Read audio file into batches and create input pipelie for STT
    batches = split_into_batches(glob(audio_input_file), batch_size=10)
    readbatchout = read_batch(batches[0])
    input = prepare_model_input(read_batch(batches[0]), device=device)

    #feed to STT model and get the text output
    output = stt_model(input)
    you_said = decoder(output[0].cpu())
    
    if(you_said == ""):
        print("No speech recognized...")
        continue
    

    
    #check if user wants to stop - This can also be achieved by implementing a Hotword Detection
    if(re.search("exit", you_said) or re.search("stop", you_said) or re.search("end",you_said)):
        break

    print(">", you_said, not re.search("computer", you_said))

    if (not re.search("computer", you_said)):
        continue
    


    
    #time log
    stt_time = time.time()
    print("time for Speech to Text conversion = ",stt_time-audio_capture_time)
    
    #clear input file so that it is not referred again and again in future
    if os.path.exists(audio_input_file):
        os.remove(audio_input_file)  

    bot_said = send_to_ai(you_said)
    
    #check if bot response is valid text
    if bot_said  == "":
        continue
    
    #Text to Speech block
    try:
        tts_audio = tts_model.save_wav(text=bot_said,
                                        speaker=speaker,
                                        sample_rate=sample_rate,
                                        audio_path=audio_output_file)
    except:
        print("we made a boo boo")
        continue
    
    wf = wave.open(audio_output_file, 'rb')
    
    # Read data in chunks
    data = wf.readframes(CHUNK)
    
    #Speak the output
    # Play the sound by writing the audio data to the stream
    while len(data) > 0:
        SpeakerStream.write(data)
        data = wf.readframes(CHUNK)
    
    wf.close()
    
    #clear output file so that it is not referred again and again in future
    if os.path.exists(audio_output_file):
        os.remove(audio_output_file)  
