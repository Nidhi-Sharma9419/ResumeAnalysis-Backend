import os
import numpy as np
import librosa
import webrtcvad
import stt
from typing import Dict, List, Tuple

class AudioTranscriber:
    def __init__(self):
        self.model_path = "models/deepspeech-0.8.2-models.tflite"
        self.scorer_path = "models/deepspeech-0.8.2-models.scorer"
        self.filler_words = ["um", "uh", "ah", "er", "like", "you know", "well", "so", "actually", "basically"]
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("Coqui STT model not found. Please download and place in models/ directory")
        
        self.model = stt.Model(self.model_path)
        self.model.enableExternalScorer(self.scorer_path)

    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and normalize audio to 16kHz mono"""
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return (audio * 32767).astype(np.int16)

    def transcribe(self, audio_path: str) -> str:
        """Convert speech to text"""
        audio = self.load_audio(audio_path)
        return self.model.stt(audio)

    def detect_fillers(self, text: str) -> Dict:
        """Analyze transcript for filler words"""
        words = text.lower().split()
        fillers = [word for word in words if word in self.filler_words]
        
        return {
            "total_fillers": len(fillers),
            "filler_words": list(set(fillers)),  # Unique fillers
            "filler_instances": fillers,
            "filler_frequency": len(fillers) / max(1, len(words))  # Fillers per word
        }

    def detect_gaps(self, audio_path: str, min_gap_duration: float = 0.5) -> List[Dict]:
        """Detect silent gaps in audio using VAD"""
        audio = self.load_audio(audio_path)
        sample_rate = 16000
        frame_duration = 30  # ms
        samples_per_frame = int(sample_rate * frame_duration / 1000)
        
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i in range(0, len(audio), samples_per_frame):
            frame = audio[i:i + samples_per_frame]
            if len(frame) < samples_per_frame:
                break
                
            is_speech = self.vad.is_speech(frame.tobytes(), sample_rate)
            
            if not is_speech and not in_gap:
                in_gap = True
                gap_start = i / sample_rate
            elif is_speech and in_gap:
                in_gap = False
                gap_end = i / sample_rate
                duration = gap_end - gap_start
                if duration >= min_gap_duration:
                    gaps.append({
                        "start": round(gap_start, 2),
                        "end": round(gap_end, 2),
                        "duration": round(duration, 2)
                    })
        
        return gaps

    def analyze_audio(self, audio_path: str) -> Dict:
        """Full analysis pipeline"""
        transcript = self.transcribe(audio_path)
        filler_analysis = self.detect_fillers(transcript)
        gap_analysis = self.detect_gaps(audio_path)
        
        # Calculate speech rate (words per minute)
        audio_duration = librosa.get_duration(filename=audio_path)
        word_count = len(transcript.split())
        speech_rate = (word_count / audio_duration) * 60 if audio_duration > 0 else 0
        
        return {
            "transcript": transcript,
            "speech_rate": round(speech_rate, 1),
            "audio_duration": round(audio_duration, 2),
            "word_count": word_count,
            "filler_analysis": filler_analysis,
            "gap_analysis": gap_analysis,
            "gap_count": len(gap_analysis)
        }