# convolver_pyroom.py
import numpy as np
import soundfile as sf
import os
import pyroomacoustics as pra
import librosa

from Microphone_4FOA import FOAMicrophone


class PyroomConvolver:
    def __init__(self, fs=16000, out_dir="outputs", room_dim=[6, 5, 3], rt60=0.3):
        """
        Convolver using Pyroomacoustics RIRs.
        fs: sample rate (default 16 kHz)
        room_dim: [Lx, Ly, Lz] room dimensions
        rt60: reverberation time (s)
        """
        self.fs = fs
        self.room_dim = room_dim
        self.rt60 = rt60
        self.mic = FOAMicrophone(fs=fs)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def generate_room(self, mic_pos, src1_pos, src2_pos):
        """
        Create a room with 1 mic + 2 sources.
        mic_pos, src1_pos, src2_pos are [x, y, z] coordinates.
        """
        e_absorption, max_order = pra.inverse_sabine(self.rt60, self.room_dim)

        room = pra.ShoeBox(
            self.room_dim,
            fs=self.fs,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

        # Add mic (1 point, FOA handled externally)
        room.add_microphone_array(np.c_[mic_pos])

        # Add sources
        room.add_source(src1_pos)
        room.add_source(src2_pos)

        # Compute RIRs
        room.compute_rir()
        return room

    def convolve_two_sources(self, src1, src2, mic_pos, src1_pos, src2_pos, theta1, phi1, theta2, phi2, prefix="mix"):
        """
        Simulate 2 sources in a room with FOA mic capture.
        Saves src1.wav, src2.wav, ambi.wav
        """
        # Generate room
        room = self.generate_room(mic_pos, src1_pos, src2_pos)

        # Extract RIRs for mic (only 1 mic so [0][0] and [0][1])
        rir1 = room.rir[0][0]
        rir2 = room.rir[0][1]

        # Convolve signals with RIRs
        sig1_rir = np.convolve(src1, rir1)[:len(src1)]
        sig2_rir = np.convolve(src2, rir2)[:len(src2)]

        # FOA encoding
        rec1 = self.mic.simulate_recording(theta1, phi1, sig1_rir)
        rec2 = self.mic.simulate_recording(theta2, phi2, sig2_rir)

        ambi = rec1 + rec2

        # Save
        sf.write(os.path.join(self.out_dir, f"{prefix}_src1.wav"), src1, self.fs)
        sf.write(os.path.join(self.out_dir, f"{prefix}_src2.wav"), src2, self.fs)
        sf.write(os.path.join(self.out_dir, f"{prefix}_ambi.wav"), ambi, self.fs)

        print(f"[INFO] Saved {prefix}: src1, src2, ambi")
        return ambi
