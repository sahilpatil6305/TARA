import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TTS_HOME", os.path.join(PROJECT_ROOT, ".tts_cache"))

from TTS.api import TTS


class TTSEngineWrapper:
    def __init__(self):
        # Initialize a lightweight, high-quality local model from Coqui
        self.tts = TTS("tts_models/en/ljspeech/fast_pitch", gpu=False)
        print("[*] TTS engine initialized successfully on CPU.")

    def generate_audio(self, text, output_path):
        """
        Generates audio and saves to .wav file using Coqui TTS.
        """
        print(f"[*] Synthesizing prompt to audio.")
        self.tts.tts_to_file(text=text, file_path=output_path)
        print(f"[*] Saved local audio: {output_path}")

_tts_engine_instance = None

def get_tts_engine():
    global _tts_engine_instance
    if _tts_engine_instance is None:
        _tts_engine_instance = TTSEngineWrapper()
    return _tts_engine_instance

def generate_audio(text, output_path):
    engine = get_tts_engine()
    engine.generate_audio(text, output_path)
