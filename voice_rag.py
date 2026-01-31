from __future__ import annotations

import argparse
import collections
import queue
import sys
import time
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 32
    channels: int = 1
    preroll_ms: int = 100
    end_silence_ms: int = 500

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def preroll_frames(self) -> int:
        return max(1, int(self.preroll_ms / self.frame_ms))


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"


def _blue(text: str) -> str:
    return f"\033[34m{text}\033[0m"


def _load_vad(threshold: float, sample_rate: int):
    """Loads Silero VAD iterator for streaming use."""
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for Silero VAD") from e

    # The official Silero VAD is commonly loaded from torch.hub.
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    vad_iter = VADIterator(model, threshold=threshold, sampling_rate=sample_rate)
    return vad_iter


def _load_stt(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is required. Install with: pip install faster-whisper"
        ) from e

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _transcribe(
    stt_model,
    audio: np.ndarray | str,
    sample_rate: int,
    beam_size: int,
) -> str:
    segments, _info = stt_model.transcribe(
        audio,
        beam_size=beam_size,
        language=None,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def _run_rag(question: str) -> str:
    from rag.graph import build_app

    app = build_app()
    result = app.invoke({"question": question})
    return result.get("answer", "")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Level 1: Mic -> Silero VAD -> Faster-Whisper -> LangGraph RAG"
    )
    parser.add_argument(
        "--audio-file",
        default=None,
        help=(
            "Path to an audio file to transcribe (wav/mp3/m4a/etc). Useful on WSL when "
            "no microphone devices are exposed to Linux."
        ),
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit.",
    )
    parser.add_argument(
        "--mic",
        default=None,
        help=(
            "Input device for the microphone. Use an integer index from --list-devices "
            "or a device name substring. If omitted, uses the OS default input device; "
            "if no default is set, selects the first input-capable device."
        ),
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD threshold (0-1). Use ~0.65 in noisy rooms.",
    )
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Faster-Whisper model name (e.g. large-v3-turbo)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Whisper device",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="CTranslate2 compute type (e.g. float16, int8, auto)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for transcription",
    )
    parser.add_argument(
        "--end-silence-ms",
        type=int,
        default=500,
        help="Silence duration to trigger end-of-sentence.",
    )
    args = parser.parse_args()

    cfg = AudioConfig(end_silence_ms=args.end_silence_ms)

    # Decide device/compute defaults.
    device = args.device
    compute_type = args.compute_type
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    print(_blue("---INITIALIZING VAD + STT---"))
    vad_iter = _load_vad(threshold=args.vad_threshold, sample_rate=cfg.sample_rate)
    stt_model = _load_stt(args.model, device=device, compute_type=compute_type)
    print(_yellow(f"[STT] faster-whisper model={args.model} device={device} compute={compute_type}"))

    if args.audio_file:
        print(_blue("---TRANSCRIBING FILE---"))
        transcript = _transcribe(
            stt_model,
            audio=args.audio_file,
            sample_rate=cfg.sample_rate,
            beam_size=args.beam_size,
        )
        if transcript:
            print(_yellow(f"[USER] {transcript}"))
            print(_blue("---RAG---"))
            answer = _run_rag(transcript)
            print(_yellow(f"[RAG ANSWER] {answer}"))
        else:
            print(_yellow("[USER] (no speech detected)"))
        return 0

    try:
        import sounddevice as sd
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "sounddevice is required for live mic mode. Install with: pip install sounddevice"
        ) from e

    if args.list_devices:
        print(sd.query_devices())
        return 0

    def _pick_input_device(mic: object | None) -> int:
        devices = sd.query_devices()

        def is_input_capable(d: dict) -> bool:
            return int(d.get("max_input_channels", 0)) > 0

        if mic is None:
            default_in = sd.default.device[0]
            if isinstance(default_in, int) and default_in >= 0:
                return default_in
            for idx, d in enumerate(devices):
                if is_input_capable(d):
                    return idx
            raise RuntimeError(
                "No input audio devices found.\n\n"
                "If you're on WSL, this often means the Linux VM cannot see your Windows microphone.\n"
                "Workarounds:\n"
                "- Run this script on Windows (native Python) instead of WSL, OR\n"
                "- Record audio on Windows and run: python voice_rag.py --audio-file <path>\n\n"
                "Otherwise, check OS audio setup/permissions and run: python voice_rag.py --list-devices"
            )

        if isinstance(mic, str) and mic.strip().isdigit():
            mic = int(mic.strip())

        if isinstance(mic, int):
            if mic < 0 or mic >= len(devices):
                raise RuntimeError(
                    f"Invalid --mic index {mic}. Run with --list-devices to see valid indexes."
                )
            if not is_input_capable(devices[mic]):
                raise RuntimeError(
                    f"Device {mic} has no input channels. Pick a device with max_input_channels>0."
                )
            return mic

        if isinstance(mic, str):
            needle = mic.strip().lower()
            for idx, d in enumerate(devices):
                name = str(d.get("name", "")).lower()
                if needle and needle in name and is_input_capable(d):
                    return idx
            raise RuntimeError(
                f"Could not find an input device matching '{mic}'. Run with --list-devices."
            )

        raise RuntimeError("Unsupported --mic value. Use an integer index or device name substring.")

    mic_device = _pick_input_device(args.mic)
    mic_info = sd.query_devices(mic_device)
    print(_yellow(f"[AUDIO] Using input device {mic_device}: {mic_info.get('name', '')}"))

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)

    def callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            print(status, file=sys.stderr)
        # indata is float32 already if dtype=float32
        chunk = indata[:, 0].copy()
        try:
            audio_q.put_nowait(chunk)
        except queue.Full:
            pass

    frame_size = cfg.frame_size

    preroll: Deque[np.ndarray] = collections.deque(maxlen=cfg.preroll_frames)
    voice_buffer: list[np.ndarray] = []

    in_speech = False
    silence_ms = 0

    print(_blue("---LISTENING (Ctrl+C to stop)---"))

    with sd.InputStream(
        device=mic_device,
        channels=cfg.channels,
        samplerate=cfg.sample_rate,
        blocksize=frame_size,
        dtype="float32",
        callback=callback,
    ):
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            preroll.append(chunk)

            # Silero VAD iterator expects a torch tensor chunk.
            import torch

            speech_dict: Optional[dict] = vad_iter(torch.from_numpy(chunk))
            is_speech = speech_dict is not None

            if is_speech:
                if not in_speech:
                    # Prepend preroll so we don't cut off the first phoneme.
                    voice_buffer = list(preroll)
                    in_speech = True
                    silence_ms = 0
                else:
                    silence_ms = 0

                voice_buffer.append(chunk)
                continue

            # Not speech
            if in_speech:
                silence_ms += cfg.frame_ms
                if silence_ms >= cfg.end_silence_ms:
                    in_speech = False
                    silence_ms = 0

                    audio = np.concatenate(voice_buffer, axis=0)
                    voice_buffer = []

                    # Transcribe
                    print(_blue("---TRANSCRIBING---"))
                    transcript = _transcribe(
                        stt_model,
                        audio=audio,
                        sample_rate=cfg.sample_rate,
                        beam_size=args.beam_size,
                    )

                    if transcript:
                        print(_yellow(f"[USER] {transcript}"))
                        print(_blue("---RAG---"))
                        answer = _run_rag(transcript)
                        print(_yellow(f"[RAG ANSWER] {answer}"))
                    else:
                        print(_yellow("[USER] (no speech detected in segment)"))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise
