from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any

from flask import Flask, jsonify, render_template, request


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m"


def _blue(text: str) -> str:
    return f"\033[34m{text}\033[0m"


def _load_stt(model_name: str, device: str, compute_type: str):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is required. Install with: pip install faster-whisper"
        ) from e

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _transcribe_file(stt_model, path: str, beam_size: int) -> str:
    segments, _info = stt_model.transcribe(
        path,
        beam_size=beam_size,
        language=None,
        vad_filter=False,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def _run_rag(question: str) -> str:
    from rag.graph import get_app

    app = get_app()
    result = app.invoke({"question": question})
    return result.get("answer", "")


def create_app(stt_model, beam_size: int) -> Flask:
    app = Flask(
        __name__,
        template_folder="web/templates",
    )

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/voice_rag")
    def voice_rag():
        if "audio" not in request.files:
            return jsonify({"error": "Missing multipart form field 'audio'"}), 400

        f = request.files["audio"]
        if not f.filename:
            return jsonify({"error": "Empty filename"}), 400

        suffix = os.path.splitext(f.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            f.save(tmp_path)

        try:
            transcript = _transcribe_file(stt_model, tmp_path, beam_size=beam_size)
            answer = _run_rag(transcript) if transcript else ""
            return jsonify({"transcript": transcript, "answer": answer})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Browser mic -> upload WAV -> Faster-Whisper -> LangGraph RAG (WSL-friendly)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
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
    args = parser.parse_args()

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

    print(_blue("---INITIALIZING STT + WEB---"))
    stt_model = _load_stt(args.model, device=device, compute_type=compute_type)
    print(_yellow(f"[STT] faster-whisper model={args.model} device={device} compute={compute_type}"))

    app = create_app(stt_model, beam_size=args.beam_size)

    print(_blue(f"---SERVING http://{args.host}:{args.port} ---"))
    print(
        _yellow(
            "If you're on WSL, open this from Windows in your browser: http://localhost:5000"
        )
    )

    app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
