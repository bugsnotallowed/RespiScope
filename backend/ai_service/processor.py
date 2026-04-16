import librosa
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import base64
import torch

from ai_model import RespiScopeModel, CONFIG, TYPE_NAMES, COND_NAMES, load_audio as model_load_audio

class AudioAIProcessor:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        # Load the PyTorch Model
        self.device = CONFIG['device']
        self.model = RespiScopeModel(CONFIG).to(self.device)
        model_path = os.path.join(os.path.dirname(__file__), "respiscope_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Successfully loaded respiscope_model.pth inside AudioAIProcessor")
        else:
            print(f"WARNING: Cannot find {model_path}. Running with random weights.")

    def load_audio(self, file_path):
        """Loads audio and normalizes it."""
        y, sr = librosa.load(file_path, sr=self.sr)
        y = librosa.util.normalize(y)
        return y, sr

    def detect_abnormal_peaks(self, y):
        """Detects high-energy transient peaks that might indicate crackles or wheezing."""
        # Calculate energy envelope
        envelope = np.abs(librosa.onset.onset_strength(y=y, sr=self.sr))
        
        # Use scipy to find peaks in the onset envelope
        # prominence helps filter out small ripples
        peaks, properties = signal.find_peaks(envelope, height=np.mean(envelope)*2, distance=self.sr//10)
        
        # Convert peak indices to timestamps
        timestamps = librosa.frames_to_time(peaks, sr=self.sr)
        return timestamps.tolist()

    def extract_acoustic_features(self, y):
        """Extracts industry-standard features for respiratory analysis."""
        # MFCCs (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        
        # Spectral Centroid (Indicates 'brightness' of sound)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0]
        
        # Zero Crossing Rate (Helps detect 'hissing' wheezes)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        features = {
            "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
            "spectral_centroid_mean": float(np.mean(spectral_centroid)),
            "zcr_mean": float(np.mean(zcr))
        }
        return features

    def generate_spectrogram(self, y):
        """Generates a Mel-Spectrogram and returns it as a base64 string."""
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        librosa.display.specshow(S_dB, sr=self.sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency Spectrogram')
        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64

    def classify_sound(self, file_path):
        """
        Uses the loaded PyTorch model to classify the sound.
        """
        try:
            # Format strictly exactly like training
            waveform = model_load_audio(file_path, CONFIG)
            waveform = waveform.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out_type, out_cond = self.model(waveform)
                
            prob_type = torch.softmax(out_type, dim=1).squeeze().cpu().numpy()
            prob_cond = torch.softmax(out_cond, dim=1).squeeze().cpu().numpy()
            
            pred_type = int(prob_type.argmax())
            pred_cond = int(prob_cond.argmax())
            
            label = f"{TYPE_NAMES[pred_type]} - {COND_NAMES[pred_cond]}"
            confidence = float(prob_cond[pred_cond])
            
            return label, confidence
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Inference error: {e}")
            return "Unknown Error", 0.0

    def process_and_filter(self, file_path, filter_type="heart"):
        """
        Applies DSP filters using scipy/librosa and returns the processed audio as a buffer.
        """
        y, sr = self.load_audio(file_path)
        
        if filter_type == "heart":
            # Low-pass < 200Hz
            sos = signal.butter(10, 200, 'lp', fs=sr, output='sos')
            y_filt = signal.sosfilt(sos, y)
        else:
            # Band-pass 100-2000Hz for lung
            sos = signal.butter(10, [100, 2000], 'bp', fs=sr, output='sos')
            y_filt = signal.sosfilt(sos, y)

        # Normalize
        y_filt = librosa.util.normalize(y_filt)
        
        # Save to memory as WAV
        buf = io.BytesIO()
        import soundfile as sf
        sf.write(buf, y_filt, sr, format='WAV')
        buf.seek(0)
        return buf.read()

    def analyze(self, file_path):
        """Main entry point for analysis."""
        y, sr = self.load_audio(file_path)
        
        peaks = self.detect_abnormal_peaks(y)
        features = self.extract_acoustic_features(y)
        spectrogram_img = self.generate_spectrogram(y)
        
        # PyTorch classification handles file formatting internally via torchaudio
        label, confidence = self.classify_sound(file_path)

        return {
            "label": label,
            "confidence": confidence,
            "abnormal_peaks": peaks,
            "features": features,
            "spectrogram": spectrogram_img
        }
