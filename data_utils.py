# data_utils.py
import os
import librosa
import soundfile as sf

class LibriSpeechProcessor:
    def __init__(self, target_sr=16000):
        """
        Processor for LibriSpeech dataset.
        Handles file collection, resampling, and batch conversion.
        target_sr: desired sampling rate (default=16kHz)
        """
        self.target_sr = target_sr

    def collect_files(self, base_path, ext=".flac"):
        """
        Recursively collect all audio files with given extension.
        Returns: list of file paths
        """
        files = []
        for root, _, filenames in os.walk(base_path):
            for f in filenames:
                if f.endswith(ext):
                    files.append(os.path.join(root, f))
        return files

    def resample_and_save(self, input_file, output_file):
        """
        Load, resample, and save audio as wav.
        """
        y, sr = librosa.load(input_file, sr=None)
        if sr != self.target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.target_sr)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        sf.write(output_file, y, self.target_sr)

    def batch_process(self, input_folder, output_folder):
        """
        Batch convert dataset to wav + resample.
        Maintains LibriSpeech folder structure.
        """
        os.makedirs(output_folder, exist_ok=True)
        files = self.collect_files(input_folder, ext=".flac")

        for f in files:
            rel_path = os.path.relpath(f, input_folder)
            out_path = os.path.join(output_folder, os.path.splitext(rel_path)[0] + ".wav")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            self.resample_and_save(f, out_path)

        print(f"✅ Processed {len(files)} files into {output_folder}")
