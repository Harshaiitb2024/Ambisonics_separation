# convolver.py
import numpy as np
import soundfile as sf
import os
from scipy.signal import fftconvolve
from Microphone_4FOA import FOAMicrophone


class DummyConvolver:
    def __init__(self, fs=16000, out_dir="outputs"):
        """
        Dummy FOA Convolver (2-source version).
        Uses simple exponential decay RIRs instead of real room simulation.
        """
        self.fs = fs
        self.mic = FOAMicrophone(fs=fs)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def generate_dummy_rir(self, delay_samples=200, decay=0.5, length=2048):
        """
        Create a simple exponentially decaying impulse response.
        """
        rir = np.zeros(length)
        rir[delay_samples] = 1.0
        for i in range(delay_samples+1, length):
            rir[i] = rir[i-1] * decay
        return rir

    def convolve_source(self, signal, rir):
        """
        Convolve source with RIR.
        """
        return fftconvolve(signal, rir, mode="full")[:len(signal)]

    def simulate_two_sources(self, src1, src2, theta1, phi1, theta2, phi2, prefix="mix"):
        """
        Take 2 mono signals, convolve with dummy RIRs, project to FOA mic.
        Saves: src1.wav, src2.wav, ambi.wav
        """
        # Generate dummy RIRs
        rir1 = self.generate_dummy_rir(delay_samples=200)
        rir2 = self.generate_dummy_rir(delay_samples=400)

        # Convolve signals
        sig1_rir = self.convolve_source(src1, rir1)
        sig2_rir = self.convolve_source(src2, rir2)

        # FOA encoding
        rec1 = self.mic.simulate_recording(theta1, phi1, sig1_rir)
        rec2 = self.mic.simulate_recording(theta2, phi2, sig2_rir)

        # Mix both sources
        ambi = rec1 + rec2

        # Save outputs
        sf.write(os.path.join(self.out_dir, f"{prefix}_src1.wav"), src1, self.fs)
        sf.write(os.path.join(self.out_dir, f"{prefix}_src2.wav"), src2, self.fs)
        sf.write(os.path.join(self.out_dir, f"{prefix}_ambi.wav"), ambi, self.fs)

        print(f"[INFO] Saved mixture {prefix} (src1, src2, ambi)")
        return ambi
