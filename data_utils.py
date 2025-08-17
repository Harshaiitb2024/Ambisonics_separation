# data_utils.py
import os
import librosa
import soundfile as sf

def collect_files(base_path, ext=".flac"):
    """Recursively collect all audio files with given extension."""
    files = []
    for root, _, filenames in os.walk(base_path):
        for f in filenames:
            if f.endswith(ext):
                files.append(os.path.join(root, f))
    return files

def resample_and_save(input_file, output_file, target_sr=16000):
    """Load, resample, and save audio as wav."""
    y, sr = librosa.load(input_file, sr=None)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sf.write(output_file, y, target_sr)

def batch_process(input_folder, output_folder, target_sr=16000):
    """Batch convert dataset to wav + resample."""
    os.makedirs(output_folder, exist_ok=True)
    files = collect_files(input_folder, ext=".flac")

    for f in files:
        rel_path = os.path.relpath(f, input_folder)
        out_path = os.path.join(output_folder, os.path.splitext(rel_path)[0] + ".wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        resample_and_save(f, out_path, target_sr)

    print(f"✅ Processed {len(files)} files into {output_folder}")
