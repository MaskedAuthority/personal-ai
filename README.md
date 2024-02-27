# Personal AI

This is an entirely local AI bot you can talk to it and will talk back. You can craft its personality as you like and have a private local AI similiar to "Jarvis" from the Iron Man movie which was also my inspiration.

> [!NOTE]
> To use the AI after running the below commands and ensuring you have the LMStudio API running use the voice command `computer` when chatting with the AI.
> For example "hey computer what are large language models". And say `exit` to make it shutdown.

### How to start

> [!IMPORTANT]  
> You will need to install LMStudio https://lmstudio.ai/ and start the "Local Inference Server" from the UI. Also pick a model to use. I recommend `phi-2.Q4_K_S.gguf`

```console
pip install -r requirements.txt
python main.py
```
