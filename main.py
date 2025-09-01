# Required Imports
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
import torch
import cv2
from gtts import gTTS
import os
import platform


model_id = "google/gemma-3n-e4b-it"

hf_token = "your hugging face token"

model = Gemma3nForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=hf_token,
).eval()

processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Welcome to the Athey! Talk to the AI and see what it thinks.")

while True:
    try:
        user_text = input("You: ")
    except EOFError:
        break

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "you are a friendly ai bot that uses both the user qestion and the image to reply in a kidn short manner. Be funny and inquisitive"},
                {"type": "image", "image": image},
                {"type": "text", "text": user_text}
            ]
        }
    ]

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
    print(f"Model: {decoded}")

    tts = gTTS(text=decoded, lang='ru')
    tts.save("output.mp3")

    try:
        if platform.system() == "Darwin":  # macOS
            os.system("afplay output.mp3")
        elif platform.system() == "Linux":
            os.system("mpg123 output.mp3")
        else:  # Windows
            os.system("start output.mp3")
    except Exception as e:
        print(f"Error playing audio: {e}")

cap.release()
cv2.destroyAllWindows()
