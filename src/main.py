"""CLI: MIDI -> matemática JSON -> Blender -> WAV -> MP4 com áudio.

    python src/main.py input.mid --soundfont instrumento.sf2
    python -m src.main input.mid --soundfont instrumento.sf2 -o output_final.mp4
"""

import argparse
import ctypes.util
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.audio_renderer import render_midi_audio
    from src.config import FPS
    from src.layout_solver import solve_layout
    from src.midi_parser import parse_midi
else:
    from .audio_renderer import render_midi_audio
    from .config import FPS
    from .layout_solver import solve_layout
    from .midi_parser import parse_midi


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera um vídeo musical 3D vertical a partir de MIDI.")
    parser.add_argument("midi", type=Path, help="Arquivo MIDI de entrada")
    parser.add_argument("--soundfont", type=Path, default=os.environ.get("SOUNDFONT"),
                        help="SoundFont .sf2 (ou variável de ambiente SOUNDFONT)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output_final.mp4"))
    parser.add_argument("--blender", default="blender", help="Executável do Blender 4.2+")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Executável do FFmpeg")
    parser.add_argument("--descent", type=float, default=0.35, help="Queda em Z por metro em X")
    parser.add_argument("--tail-seconds", type=float, default=1.0,
                        help="Tempo após o fim do MIDI; acima de 1 s acrescenta silêncio")
    parser.add_argument("--resolution-percentage", type=int, default=100,
                        help="Escala de 1 a 100; use 10 para uma prévia rápida")
    parser.add_argument("--render-samples", type=int, default=64, help="Amostras EEVEE por frame")
    parser.add_argument("--overwrite", action="store_true", help="Permite substituir a saída existente")
    return parser


def run_pipeline(args: argparse.Namespace) -> Path:
    """Limpa temporários inclusive em falhas; publica a saída só após o muxing."""
    midi_path = args.midi.expanduser().resolve()
    if args.soundfont is None:
        raise ValueError("Informe --soundfont arquivo.sf2 ou configure SOUNDFONT.")
    soundfont_path = args.soundfont.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path in (midi_path, soundfont_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.suffix.lower() != ".mp4" or output in (midi_path, soundfont_path):
        raise ValueError("A saída deve ser um arquivo .mp4 diferente das entradas.")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"A saída já existe: {output}. Use --overwrite para substituí-la.")
    if not math.isfinite(args.descent) or args.descent < 0:
        raise ValueError("--descent deve ser finito e não negativo.")
    if not math.isfinite(args.tail_seconds) or args.tail_seconds < 0:
        raise ValueError("--tail-seconds deve ser finito e não negativo.")
    if not 1 <= args.resolution_percentage <= 100 or args.render_samples < 1:
        raise ValueError("Use resolução entre 1 e 100 e pelo menos uma amostra.")
    blender = shutil.which(args.blender)
    ffmpeg = shutil.which(args.ffmpeg)
    if blender is None or ffmpeg is None:
        raise RuntimeError("Blender e FFmpeg precisam estar instalados e acessíveis no PATH.")
    if ctypes.util.find_library("fluidsynth") is None:
        raise RuntimeError("Instale a biblioteca nativa libfluidsynth para sintetizar o MIDI.")

    print("A: Interpretando MIDI e calculando trajetórias...", flush=True)
    timeline = parse_midi(midi_path)
    arcs = solve_layout(timeline.timestamps)
    frame_count = max(
        math.ceil((timeline.end_time + args.tail_seconds) * FPS),
        math.ceil(timeline.timestamps[-1] * FPS) + 1,
    )
    duration = frame_count / FPS
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="marble-", dir=output.parent) as directory:
        work = Path(directory)
        plan_path = work / "scene.json"
        video_path = work / "temp_video.mp4"
        audio_path = work / "temp_audio.wav"
        final_path = work / "output_final.mp4"
        plan = {
            "schema_version": 1,
            "fps": FPS,
            "timestamps": timeline.timestamps,
            "segments": [asdict(arc) for arc in arcs],
            "frame_end": frame_count,
            "descent": args.descent,
            "resolution_percentage": args.resolution_percentage,
            "render_samples": args.render_samples,
        }
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, allow_nan=False), encoding="utf-8")

        print("B: Renderizando vídeo no Blender...", flush=True)
        subprocess.run(
            [blender, "--background", "--python-exit-code", "1", "--python",
             "src/blender_builder.py", "--", "--scene-json", str(plan_path),
             "--video-output", str(video_path)],
            cwd=PROJECT_ROOT, check=True,
        )
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise RuntimeError("O Blender não gerou o vídeo esperado.")

        print("C: Sintetizando áudio com FluidSynth...", flush=True)
        render_midi_audio(midi_path, soundfont_path, audio_path, duration=duration)

        print("D: Combinando vídeo e áudio...", flush=True)
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", str(video_path), "-i", str(audio_path),
             "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "320k", "-af", "apad",
             "-t", f"{duration:.9f}", "-movflags", "+faststart", str(final_path)],
            check=True,
        )
        if not final_path.is_file() or final_path.stat().st_size == 0:
            raise RuntimeError("O FFmpeg não gerou o MP4 final.")
        final_path.replace(output)
    print(f"E: Temporários removidos. Vídeo final: {output}", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_pipeline(args)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
