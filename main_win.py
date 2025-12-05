# main_win.py
import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal

from llm_backend import ChatSession, LLMClient


class GenerateThread(QThread):
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, llm_client: LLMClient, chat: ChatSession, max_tokens: int, temperature: float):
        super().__init__()
        self.llm_client = llm_client
        self.chat = chat
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self):
        try:
            text = self.llm_client.generate(
                self.chat,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self.finished.emit(text)
        except Exception as e:
            self.error.emit(str(e))


class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local AI Chat (Windows)")
        self.resize(900, 600)

        self.chat_session = ChatSession()
        self.llm_client = None  # will be set when model is loaded

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Top: model load + settings
        top_layout = QHBoxLayout()
        self.model_path_label = QLabel("No model loaded")
        self.btn_load_model = QPushButton("Load GGUF Model")
        self.btn_load_model.clicked.connect(self.load_model)

        top_layout.addWidget(self.model_path_label)
        top_layout.addWidget(self.btn_load_model)

        # Settings row
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("Max tokens:"))
        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(16, 4096)
        self.spin_max_tokens.setValue(256)
        settings_layout.addWidget(self.spin_max_tokens)

        settings_layout.addWidget(QLabel("Temperature:"))
        self.spin_temperature = QDoubleSpinBox()
        self.spin_temperature.setDecimals(2)
        self.spin_temperature.setRange(0.0, 2.0)
        self.spin_temperature.setSingleStep(0.05)
        self.spin_temperature.setValue(0.7)
        settings_layout.addWidget(self.spin_temperature)

        layout.addLayout(top_layout)
        layout.addLayout(settings_layout)

        # System prompt
        layout.addWidget(QLabel("System prompt:"))
        self.txt_system_prompt = QTextEdit()
        self.txt_system_prompt.setPlainText(self.chat_session.system_prompt)
        layout.addWidget(self.txt_system_prompt)

        # Chat history
        layout.addWidget(QLabel("Conversation:"))
        self.txt_chat_history = QTextEdit()
        self.txt_chat_history.setReadOnly(True)
        layout.addWidget(self.txt_chat_history)

        # Bottom: input + send/clear
        input_layout = QHBoxLayout()
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type your message...")
        self.input_line.returnPressed.connect(self.on_send_clicked)

        self.btn_send = QPushButton("Send")
        self.btn_send.clicked.connect(self.on_send_clicked)

        self.btn_clear = QPushButton("Clear Chat")
        self.btn_clear.clicked.connect(self.on_clear_clicked)

        input_layout.addWidget(self.input_line)
        input_layout.addWidget(self.btn_send)
        input_layout.addWidget(self.btn_clear)

        layout.addLayout(input_layout)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", "", "GGUF Files (*.gguf);;All Files (*)")
        if not file_path:
            return

        try:
            self.model_path_label.setText("Loading model...")
            self.repaint()
            QApplication.processEvents()

            self.llm_client = LLMClient(model_path=file_path)
            self.model_path_label.setText(f"Model: {Path(file_path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error loading model", str(e))
            self.model_path_label.setText("No model loaded")
            self.llm_client = None

    def on_send_clicked(self):
        text = self.input_line.text().strip()
        if not text:
            return

        if self.llm_client is None:
            QMessageBox.warning(self, "No model", "Please load a GGUF model first.")
            return

        # update system prompt
        self.chat_session.system_prompt = self.txt_system_prompt.toPlainText().strip()

        # Add user message
        self.chat_session.add_user_message(text)
        self.append_to_chat("You", text)
        self.input_line.clear()

        # Disable UI while generating
        self.btn_send.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.input_line.setEnabled(False)

        # Threaded generation
        max_tokens = self.spin_max_tokens.value()
        temperature = self.spin_temperature.value()

        self.gen_thread = GenerateThread(self.llm_client, self.chat_session, max_tokens, temperature)
        self.gen_thread.finished.connect(self.on_generation_finished)
        self.gen_thread.error.connect(self.on_generation_error)
        self.gen_thread.start()

    def on_generation_finished(self, reply: str):
        self.chat_session.add_assistant_message(reply)
        self.append_to_chat("Assistant", reply)
        self._enable_ui()

    def on_generation_error(self, error_msg: str):
        QMessageBox.critical(self, "Generation error", error_msg)
        self._enable_ui()

    def _enable_ui(self):
        self.btn_send.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.input_line.setEnabled(True)
        self.input_line.setFocus()

    def on_clear_clicked(self):
        self.chat_session.reset()
        self.txt_chat_history.clear()

    def append_to_chat(self, sender: str, text: str):
        self.txt_chat_history.append(f"<b>{sender}:</b> {text}")
        self.txt_chat_history.verticalScrollBar().setValue(
            self.txt_chat_history.verticalScrollBar().maximum()
        )


def main():
    app = QApplication(sys.argv)
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
