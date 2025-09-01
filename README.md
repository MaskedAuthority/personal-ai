# Athey - A Multimodal AI Chatbot

Athey is a friendly, inquisitive, and funny AI chatbot that uses both text and images to interact with you. It captures an image from your webcam, combines it with your text input, and generates a response that is both spoken and displayed on the screen.

## How it Works

The script uses the `google/gemma-3n-e4b-it` model from Hugging Face for text generation and `gTTS` for text-to-speech. It captures an image from your webcam, and you can then type a question or a comment. The model processes both the image and your text to generate a response.

## Features

- **Multimodal Interaction:**  Combines text and images for more engaging conversations.
- **Text-to-Speech:**  Reads the AI's responses aloud.
- **Cross-Platform:** Works on macOS, Linux, and Windows.
- **Rich CLI:** Uses the `rich` library for a more visually appealing command-line interface.

## Requirements

- Python 3.x
- The following Python libraries:
  - `transformers`
  - `Pillow`
  - `requests`
  - `torch`
  - `opencv-python`
  - `gtts`
  - `rich`
  - `mpg123` (for Linux)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/athey.git
    cd athey
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    On Linux, you may also need to install `mpg123`:
    ```bash
    sudo apt-get install mpg123
    ```

3.  **Set up your Hugging Face token:**
    The script requires a Hugging Face token to download the model. You need to have a Hugging Face account and an access token.  Set it as an environment variable:
    ```bash
    export HF_TOKEN="your-hugging-face-token"
    ```
    Alternatively, you can hardcode it in the `main.py` file, but this is not recommended.

## Usage

1.  **Run the script:**
    ```bash
    python main.py
    ```

2.  **Interact with Athey:**
    - The script will open your webcam.
    - Type your message in the terminal and press Enter.
    - Athey will respond with text and speech.

3.  **Exit the chat:**
    - Press `Ctrl+C` or `Ctrl+D` to exit.

## To-Do

- [ ]  Add a graphical user interface (GUI).
- [ ]  Allow the user to select a different model.
- [ ]  Improve error handling.
