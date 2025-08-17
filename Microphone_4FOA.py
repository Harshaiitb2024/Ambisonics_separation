# Microphone_4FOA.py
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# ==============================
# FOA Microphone Directivities
# ==============================
def omni(theta, phi):
    """Omnidirectional component"""
    return 1 / np.sqrt(2)

def dipole_x(theta, phi):
    """Dipole along X-axis"""
    return np.cos(theta) * np.sin(phi)

def dipole_y(theta, phi):
    """Dipole along Y-axis"""
    return np.sin(theta) * np.sin(phi)

def dipole_z(theta, phi):
    """Dipole along Z-axis"""
    return np.cos(phi)


# ==============================
# FOAMicrophone Class
# ==============================
class FOAMicrophone:
    def __init__(self, fs=16000):
        """
        First Order Ambisonics Microphone (B-format, WXYZ).
        fs: sampling rate (default 16kHz for LibriSpeech)
        """
        self.fs = fs

    def simulate_recording(self, theta, phi, signal):
        """
        Simulate FOA (W, X, Y, Z) channels for a source at (theta, phi).
        signal: mono numpy array
        Returns: numpy array of shape (samples, 4)
        """
        W = omni(theta, phi) * signal
        X = dipole_x(theta, phi) * signal
        Y = dipole_y(theta, phi) * signal
        Z = dipole_z(theta, phi) * signal
        return np.stack([W, X, Y, Z], axis=1)

    def save_recording(self, recording, filename="foa_recording.wav"):
        """
        Save a 4-channel FOA recording to disk.
        """
        sf.write(filename, recording, self.fs)
        print(f"[INFO] FOA recording saved to {filename}")

    def plot_analysis(self, recording, theta, phi):
        """
        Plot time and frequency domain analysis for each FOA channel.
        """
        fig, axes = plt.subplots(4, 2, figsize=(12, 8))
        labels = ["W (Omni)", "X (Dipole-X)", "Y (Dipole-Y)", "Z (Dipole-Z)"]

        for i in range(4):
            # Time-domain
            axes[i, 0].plot(recording[:, i])
            axes[i, 0].set_title(f"{labels[i]} - Time Domain")

            # Frequency-domain
            freqs = np.fft.rfftfreq(len(recording[:, i]), 1 / self.fs)
            fft_mag = np.abs(np.fft.rfft(recording[:, i]))
            axes[i, 1].plot(freqs, fft_mag)
            axes[i, 1].set_title(f"{labels[i]} - Frequency Domain")

        plt.suptitle(f"FOA Recording (theta={theta:.2f}, phi={phi:.2f})")
        plt.tight_layout()
        plt.show()
