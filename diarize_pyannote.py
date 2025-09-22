import os
import argparse
import json
from typing import Dict, Any

from pyannote.audio import Pipeline


def load_pipeline() -> Pipeline:
    """Load the pretrained pyannote diarization pipeline.

    This uses the open 'pyannote/speaker-diarization-3.1' pipeline. For best results,
    set the environment variable HF_TOKEN with your Hugging Face token if required.
    """
    model_id = "pyannote/speaker-diarization-3.1"
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        pipeline = Pipeline.from_pretrained(model_id, use_auth_token=hf_token)
    else:
        pipeline = Pipeline.from_pretrained(model_id)
    return pipeline


def diarize_file(input_wav: str, output_json: str) -> None:
    pipeline = load_pipeline()
    diarization = pipeline(input_wav)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"audio_path": input_wav, "segments": segments}, f, indent=2)

    print(f"Wrote {output_json}")


def diarize_dir(input_dir: str, output_dir: str) -> None:
    pipeline = load_pipeline()
    for root, _, files in os.walk(input_dir):
        for name in files:
            if not name.lower().endswith(".wav"):
                continue
            wav_path = os.path.join(root, name)
            rel = os.path.relpath(wav_path, input_dir)
            out_path = os.path.join(output_dir, os.path.splitext(rel)[0] + ".diarization.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            diarization = pipeline(wav_path)
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                })

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"audio_path": wav_path, "segments": segments}, f, indent=2)
            print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust diarization with pyannote.audio")
    sub = parser.add_subparsers(dest="cmd")

    p_file = sub.add_parser("file", help="Diarize a single WAV file")
    p_file.add_argument("audio", type=str, help="Path to .wav file")
    p_file.add_argument("--out", type=str, default=None, help="Output JSON path (default next to WAV)")

    p_dir = sub.add_parser("dir", help="Diarize all WAV files under a directory")
    p_dir.add_argument("input_dir", type=str, help="Directory with WAV files")
    p_dir.add_argument("output_dir", type=str, help="Directory to save JSON outputs")

    args = parser.parse_args()

    if args.cmd == "file":
        out = args.out or (os.path.splitext(args.audio)[0] + ".diarization.json")
        diarize_file(args.audio, out)
    elif args.cmd == "dir":
        diarize_dir(args.input_dir, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


