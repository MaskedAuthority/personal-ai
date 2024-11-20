# Required Imports
import os
import re
import torch
import torchaudio
import wave
import pyaudio
from glob import glob
import speech_recognition as sr
from openai import OpenAI
from omegaconf import OmegaConf
from silero.utils import (
    init_jit_model,
    split_into_batches,
    read_audio,
    read_batch,
    prepare_model_input
)
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress
from rich.live import Live
from rich.spinner import Spinner
from time import sleep

# Console for Hacker UI
console = Console()

# Enhanced Welcome Screen with Gradient
WELCOME_ART = Text(
    """
██╗    ██╗███████╗██╗      ██████╗ ██████╗ ███╗   ███╗███████╗
██║    ██║██╔════╝██║     ██╔════╝██╔═══██╗████╗ ████║██╔════╝
██║ █╗ ██║█████╗  ██║     ██║     ██║   ██║██╔████╔██║█████╗  
██║███╗██║██╔══╝  ██║     ██║     ██║   ██║██║╚██╔╝██║██╔══╝  
╚███╔███╔╝███████╗███████╗╚██████╗╚██████╔╝██║ ╚═╝ ██║███████╗
 ╚══╝╚══╝ ╚══════╝╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝
""",
    style="bold purple blink",
)

# Typing Effect for Output
def hacker_type(text, delay=0.05):
    for char in text:
        print(char, end='', flush=True)
        sleep(delay)
    print()

# Enhanced Loading Effect
def loading_effect(task_name="Initializing..."):
    with Progress() as progress:
        task = progress.add_task(f"[cyan]{task_name}[/cyan]", total=100)
        for i in range(100):
            progress.update(task, advance=1)
            sleep(0.02)

# Initialize Models and Devices
local_ai = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

device = torch.device('cpu')  # Use CPU or other PyTorch devices
r_audio = sr.Recognizer()
p = pyaudio.PyAudio()

# Speaker Stream
SpeakerStream = p.open(format=8, channels=1, rate=48000, output=True)

# Load Silero Models
loading_effect("Loading AI Models")
stt_model, decoder, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_stt',
    jit_model='jit_xlarge',
    language='en',
    device=device
)
stt_model.to(device)

tts_model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='en',
    speaker="v3_en"
)
tts_model.to(device)

# Welcome Screen with Spinner Animation
with Live(Spinner("dots", text="Initializing AI Interface...", style="bold purple")) as live:
    sleep(2)
    live.update("[bold green]Initialization Complete![/bold green]")
    sleep(1)

console.print(Panel(WELCOME_ART, expand=False, border_style="purple", title="[cyan]AI Console[/cyan]"))
hacker_type("Welcome to the [AI Interface. You now have full control.\n", delay=0.07)
sleep(1)

# Main Loop
while True:
    try:
        with sr.Microphone() as SR_AudioSource:
            console.print(
                Panel("[cyan bold blink]Listening for input...[/cyan bold blink]", border_style="green", expand=False)
            )
            mic_audio = r_audio.listen(SR_AudioSource, timeout=1, phrase_time_limit=4)
    except Exception as e:
        console.print(Panel(f"[red]No input detected or error occurred:[/red] {e}", expand=False))
        continue

    # Save Captured Audio
    audio_input_file = "micAudioInput.wav"
    with open(audio_input_file, "wb") as file:
        file.write(mic_audio.get_wav_data())
        file.flush()

    # Read and Process Audio
    if not os.path.exists(audio_input_file):
        console.print("[red]Audio file missing. Restarting...[/red]")
        continue

    loading_effect("Processing Audio Input")
    batches = split_into_batches(glob(audio_input_file), batch_size=10)
    input_data = prepare_model_input(read_batch(batches[0]), device=device)

    output = stt_model(input_data)
    you_said = decoder(output[0].cpu())

    if not you_said:
        console.print(Panel("[yellow]No recognizable speech detected. Try again![/yellow]", expand=False))
        continue

    # Debugging: Display what was heard
    console.print(f"[bold blue]Debug: Recognized speech:[/bold blue] {you_said}")
    print(">", you_said, not re.search("computer", you_said, re.IGNORECASE))

    # Skip iteration if "computer" is not mentioned
    if not re.search("computer", you_said, re.IGNORECASE):
        console.print("[yellow]Skipping because 'computer' was not mentioned...[/yellow]")
        continue

    # Respond to the input
    console.print("[yellow bold blink]You mentioned 'computer'![/yellow bold blink]")
    loading_effect("Generating AI Response")
    bot_response = local_ai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": you_said}],
        temperature=0.8,
        frequency_penalty=0.2,
    ).choices[0].message.content

    # Display the AI response
    console.print(Panel(f"[bold green]AI Response:[/bold green] {bot_response}", border_style="blue", expand=False))

    # Text-to-Speech
    loading_effect("Preparing Voice Output")
    tts_audio = tts_model.save_wav(
        text=bot_response,
        speaker="en_0",
        sample_rate=48000,
        audio_path="ttsAudioOutput.wav"
    )
    with wave.open("ttsAudioOutput.wav", 'rb') as wf:
        data = wf.readframes(1024)
        while len(data) > 0:
            SpeakerStream.write(data)
            data = wf.readframes(1024)

    # Cleanup
    if os.path.exists("micAudioInput.wav"):
        os.remove("micAudioInput.wav")
    if os.path.exists("ttsAudioOutput.wav"):
        os.remove("ttsAudioOutput.wav")

    # Exit Command
    if re.search(r"(exit|stop|end)", you_said, re.IGNORECASE):
        hacker_type("Exiting [bold red]AI Terminal[/bold red]. Goodbye, Agent.", delay=0.05)
        break
