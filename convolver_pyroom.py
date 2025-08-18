# convolver_pyroom.py
import numpy as np
import soundfile as sf
import os
import pyroomacoustics as pra
from Microphone_4FOA import FOAMicrophone


class PyroomConvolver:
    def __init__(self, fs=16000, out_dir="outputs", room_dim=[6, 5, 3], rt60=0.3):
        """
        Convolver using Pyroomacoustics with FOA microphone encoding.
        """
        self.fs = fs
        self.room_dim = room_dim
        self.rt60 = rt60
        self.mic = FOAMicrophone(fs=fs)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def cartesian_to_angles(self, src_pos, mic_pos):
        """Convert source position (x,y,z) to FOA azimuth & elevation (theta, phi)."""
        dx, dy, dz = src_pos[0]-mic_pos[0], src_pos[1]-mic_pos[1], src_pos[2]-mic_pos[2]
        theta = np.arctan2(dy, dx)  # azimuth
        phi = np.arctan2(dz, np.sqrt(dx**2 + dy**2))  # elevation
        return theta, phi

    def generate_room(self, mic_pos, src1_pos, src2_pos):
        """Create shoebox room with omni sources + mic."""
        e_absorption, max_order = pra.inverse_sabine(self.rt60, self.room_dim)

        room = pra.ShoeBox(
            self.room_dim,
            fs=self.fs,
            materials=pra.Material(e_absorption),
            max_order=max_order,
        )

        # Add mic (omni in Pyroom, FOA encoding applied later)
        room.add_microphone_array(np.c_[mic_pos])

        # Add sources (omni by default)
        room.add_source(src1_pos)
        room.add_source(src2_pos)

        room.compute_rir()
        return room

    def convolve_two_sources(self, src1, src2, mic_pos, src1_pos, src2_pos, prefix="mix"):
        """Simulate 2 omni sources, convolve with room, encode with FOA mic."""
        # Room + RIRs
        room = self.generate_room(mic_pos, src1_pos, src2_pos)
        rir1, rir2 = room.rir[0][0], room.rir[0][1]

        # Convolution
        sig1_rir = np.convolve(src1, rir1)[:len(src1)]
        sig2_rir = np.convolve(src2, rir2)[:len(src2)]

        # Compute FOA encoding angles
        theta1, phi1 = self.cartesian_to_angles(src1_pos, mic_pos)
        theta2, phi2 = self.cartesian_to_angles(src2_pos, mic_pos)

        # FOA encoding
        rec1 = self.mic.simulate_recording(theta1, phi1, sig1_rir)
        rec2 = self.mic.simulate_recording(theta2, phi2, sig2_rir)

        # Mixture
        ambi = rec1 + rec2

        # Save
        sf.write(os.path.join(self.out_dir, f"{prefix}_src1.wav"), src1, self.fs)
        sf.write(os.path.join(self.out_dir, f"{prefix}_src2.wav"), src2, self.fs)
        sf.write(os.path.join(self.out_dir, f"{prefix}_ambi.wav"), ambi, self.fs)

        print(f"[INFO] Saved {prefix}: src1, src2, ambi")
        return ambi
