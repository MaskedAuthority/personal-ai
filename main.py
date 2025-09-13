from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
import torch
import cv2
from gtts import gTTS
import os
import platform
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme
from rich.text import Text
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

model_id = "google/gemma-3n-e4b-it"
hf_token = "your_api_key"
system_prompt = "how do you want your model to act"

custom_theme = Theme({
    "user_name": "bold orange3",
    "model_name": "bold deep_sky_blue1",
    "user_text": "#E6E6FA",
    "model_text": "#F0FFF0",
    "welcome_panel": "bold #FFD700",
    "info": "dim #B0C4DE"
})

console = Console(theme=custom_theme)

try:
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
    whisper_model = whisper.load_model("base")
except Exception as e:
    console.print(f"Error loading model or processor: {e}", style="bold red")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    console.print("Error: Could not open webcam.", style="bold red")
    exit()

def listen_and_transcribe():
    fs = 44100
    seconds = 5
    console.print("Recording...", style="info")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    console.print("Recording finished.", style="info")
    write('output.wav', fs, myrecording)
    result = whisper_model.transcribe("output.wav")
    return result["text"]

while True:
    try:
        console.print("Press Enter to start recording, then speak.", style="user_name")
        input()
        user_text = listen_and_transcribe()
        console.print(f"You said: {user_text}", style="user_text")

        if "exit" in user_text.lower():
            console.print("\n[info]Exiting chat...[/info]")
            break
    except (EOFError, KeyboardInterrupt):
        console.print("\n[info]Exiting chat...[/info]")
        break


    ret, frame = cap.read()
    if not ret:
        console.print("Error: Could not read frame from webcam.", style="bold red")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "image", "image": image},
                {"type": "text", "text": user_text}
            ]
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)

        model_output = Text.from_markup(f"[model_text]{decoded}[/model_text]")
        console.print(Panel(
            model_output,
            title="Athey",
            title_align="left",
            border_style="model_name",
            padding=(1, 2)
        ))

        tts = gTTS(text=decoded, lang='en')
        tts.save("output.mp3")

        if platform.system() == "Darwin":
            os.system("afplay output.mp3")
        elif platform.system() == "Linux":
            os.system("mpg123 output.mp3")
        else:
            os.system("start output.mp3")

    except Exception as e:
        console.print(f"An error occurred: {e}", style="bold red")

cap.release()
cv2.destroyAllWindows()