ChatGPT-style desktop app with support for LLaMA or other local models, in two variants:

Windows desktop GUI app (PySide6 / Qt)

Ubuntu/Linux web app (FastAPI + simple frontend)

Both will share the same idea: connect to a local LLM (like llama-cpp-python).

0. Tech Stack (both repos)

Language: Python 3.10+

LLM Backend: llama-cpp-python (works with GGUF models locally)

Windows UI: PySide6 (Qt-based GUI)

Linux UI: FastAPI backend + simple HTML/JS frontend

Features (MVP):

Chat history in the session

System prompt configuration

Model settings: temperature, max tokens

Clear conversation

Support for multiple local models (you select by path)

1. Common Backend Design

We’ll use the same style of backend on both:

LLMClient → wraps the local model

ChatSession → stores messages and calls the client




--------------------------------------------------------



Basic README.md idea (summary)

How to create venv

pip install -r requirements.txt

Download a GGUF model (e.g. LLaMA 3, Mistral etc.)

Run python main_win.py

Click Load GGUF Model, select file, start chatting
