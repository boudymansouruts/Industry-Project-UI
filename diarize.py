import os
import json
import argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from preprocess import (
    load_audio,
    segment_speakers_with_global_analysis,
)


def diarize_audio(audio: np.ndarray, sr: int = 16000) -> List[Tuple[float, float, str]]:
    """Return list of (start_sec, end_sec, speaker_label) segments for the given audio array."""
    return segment_speakers_with_global_analysis(audio, sr)


def diarize_file(audio_path: str, output_json: Optional[str] = None) -> List[Dict[str, Any]]:
    """Diarize a single audio file. Optionally write segments to JSON.

    Returns a list of dicts: {start, end, speaker} in seconds.
    """
    audio, sr = load_audio(audio_path, target_sr=16000)
    segments = diarize_audio(audio, sr)
    results = [
        {"start": float(start), "end": float(end), "speaker": speaker}
        for start, end, speaker in segments
    ]

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"audio_path": audio_path, "segments": results}, f, indent=2)

    return results


def diarize_directory(input_dir: str, output_dir: Optional[str] = None) -> None:
    """Diarize all .wav files under input_dir. Writes a JSON per file.

    If output_dir is None, writes next to each .wav with .diarization.json suffix.
    """
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(root, name)

            if output_dir:
                rel = os.path.relpath(wav_path, input_dir)
                base, _ = os.path.splitext(rel)
                out_path = os.path.join(output_dir, base + ".diarization.json")
            else:
                out_path = os.path.splitext(wav_path)[0] + ".diarization.json"

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            diarize_file(wav_path, out_path)
            print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple diarization using clustering over speech segments")
    sub = parser.add_subparsers(dest="cmd")

    p_file = sub.add_parser("file", help="Diarize a single audio file")
    p_file.add_argument("audio", type=str, help="Path to .wav file")
    p_file.add_argument("--out", type=str, default=None, help="Output JSON path")

    p_dir = sub.add_parser("dir", help="Diarize all .wav files in a directory tree")
    p_dir.add_argument("input_dir", type=str, help="Input directory containing .wav files")
    p_dir.add_argument("--out_dir", type=str, default=None, help="Output root directory for JSON files")

    args = parser.parse_args()

    if args.cmd == "file":
        segments = diarize_file(args.audio, args.out)
        print(json.dumps({"audio_path": args.audio, "segments": segments}, indent=2))
    elif args.cmd == "dir":
        diarize_directory(args.input_dir, args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


