# foa_simulator.py
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
# Signal Generator
# ==============================
def generate_signal(duration=1.0, fs=16000, freq=440):
    """
    Generate a sine wave burst for testing.
    Returns: (signal, fs)
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t), fs


# ==============================
# FOA Recording Simulator
# ==============================
def simulate_recording(theta, phi, signal):
    """
    Simulate FOA (W, X, Y, Z) channels given source position (theta, phi).
    Returns: numpy array of shape (samples, 4)
    """
    W = omni(theta, phi) * signal
    X = dipole_x(theta, phi) * signal
    Y = dipole_y(theta, phi) * signal
    Z = dipole_z(theta, phi) * signal
    return np.stack([W, X, Y, Z], axis=1)


# ==============================
# Save FOA Recording
# ==============================
def save_recording(recording, fs, filename="foa_recording.wav"):
    """Save the 4-channel FOA recording to a WAV file"""
    sf.write(filename, recording, fs)
    print(f"[INFO] FOA recording saved to {filename}")


# ==============================
# Analysis Plot
# ==============================
def plot_analysis(recording, fs, theta, phi):
    """Plot time-domain and frequency-domain for each channel"""
    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    labels = ["W (Omni)", "X (Dipole-X)", "Y (Dipole-Y)", "Z (Dipole-Z)"]

    for i in range(4):
        # Time-domain
        axes[i, 0].plot(recording[:, i])
        axes[i, 0].set_title(f"{labels[i]} - Time Domain")

        # Frequency-domain
        freqs = np.fft.rfftfreq(len(recording[:, i]), 1 / fs)
        fft_mag = np.abs(np.fft.rfft(recording[:, i]))
        axes[i, 1].plot(freqs, fft_mag)
        axes[i, 1].set_title(f"{labels[i]} - Frequency Domain")

    plt.suptitle(f"FOA Recording (theta={theta:.2f}, phi={phi:.2f})")
    plt.tight_layout()
    plt.show()
