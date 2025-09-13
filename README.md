# Athey - A Multimodal AI Assistant

Athey is a multimodal AI assistant that uses your webcam and microphone to engage in a conversation with you. You can ask it questions, and it will respond with both text and speech. It can also see what you're showing it through your webcam and incorporate that into the conversation.

## Features

*   **Multimodal Interaction:** Athey can understand both text and images, allowing for a richer and more intuitive user experience.
*   **Speech-to-Text:** Speak your questions and have them transcribed into text using OpenAI's Whisper model.
*   **Text-to-Speech:** Athey's responses are converted into speech using Google's Text-to-Speech (gTTS) service.
*   **Webcam Integration:** Athey can see what you're showing it through your webcam and use that information in the conversation.
*   **Gemma-3n Language Model:** Powered by Google's Gemma-3n, a powerful and versatile language model.

## Requirements

*   Python 3.7+
*   Hugging Face Token
*   `mpg123` (for Linux) or `afplay` (for macOS) for audio playback.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/personal-ai.git
    ```
2.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  Install `mpg123` if you are on Linux:
    ```bash
    sudo apt-get install mpg123
    ```

## Usage

1.  Make sure you have a webcam and microphone connected to your computer.
2.  Set your Hugging Face token as an environment variable:
    ```bash
    export HF_TOKEN="your-hugging-face-token"
    ```
3.  Run the application:
    ```bash
    python main.py
    ```
4.  Press Enter to start recording, then speak your question.

## Dependencies

*   [transformers](https://pypi.org/project/transformers/)
*   [Pillow](https://pypi.org/project/Pillow/)
*   [requests](https://pypi.org/project/requests/)
*   [torch](https://pypi.org/project/torch/)
*   [opencv-python](https://pypi.org/project/opencv-python/)
*   [gTTS](https://pypi.org/project/gTTS/)
*   [rich](https://pypi.org/project/rich/)
*   [openai-whisper](https://pypi.org/project/openai-whisper/)
*   [sounddevice](https://pypi.org/project/sounddevice/)
*   [scipy](https://pypi.org/project/scipy/)
*   [numpy](https://pypi.org/project/numpy/)
