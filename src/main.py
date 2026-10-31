"""CLI: MIDI -> matemática JSON -> Blender -> WAV -> MP4 com áudio.

    python src/main.py input.mid --soundfont instrumento.sf2
    python -m src.main input.mid --soundfont instrumento.sf2 -o output_final.mp4
"""

import argparse
import ctypes.util
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
    from src.frame_cache import FrameCache, atomic_json, cache_identity, render_dimensions
    from src.layout_solver import solve_layout
    from src.midi_parser import parse_midi
else:
    from .audio_renderer import render_midi_audio
    from .config import FPS
    from .frame_cache import FrameCache, atomic_json, cache_identity, render_dimensions
    from .layout_solver import solve_layout
    from .midi_parser import parse_midi


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gera um vídeo musical 3D vertical a partir de MIDI.")
    parser.add_argument("midi", type=Path, help="Arquivo MIDI de entrada")
    parser.add_argument("--soundfont", type=Path, default=os.environ.get("SOUNDFONT"),
                        help="SoundFont .sf2 (ou variável de ambiente SOUNDFONT)")
    parser.add_argument("-o", "--output", type=Path, default=Path("output_final.mp4"))
    parser.add_argument("--blender", default="blender", help="Executável do Blender 4.0.2")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Executável do FFmpeg")
    parser.add_argument("--descent", type=float, default=0.35, help="Queda em Z por metro em X")
    parser.add_argument("--tail-seconds", type=float, default=1.0,
                        help="Tempo após o fim do MIDI; acima de 1 s acrescenta silêncio")
    parser.add_argument("--max-duration", type=float,
                        help="Limita notas, geometria, vídeo e áudio aos primeiros N segundos")
    parser.add_argument("--resolution-percentage", type=int, default=100,
                        help="Escala de 1 a 100; use 10 para uma prévia rápida")
    parser.add_argument("--render-samples", type=int, default=64, help="Amostras Cycles por frame (máximo 64)")
    parser.add_argument("--overwrite", action="store_true", help="Permite substituir a saída existente")
    parser.add_argument("--cache-dir", type=Path,
                        help="Pasta persistente para frames (ex.: Google Drive já montado)")
    parser.add_argument("--force", action="store_true", help="Ignora o cache e renderiza todos os frames novamente")
    return parser


def run_pipeline(args: argparse.Namespace) -> Path:
    """Preserva os frames; limpa apenas os temporários de codificação/muxing."""
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
    if args.max_duration is not None and (
        not math.isfinite(args.max_duration) or args.max_duration < 1 / FPS
    ):
        raise ValueError(f"--max-duration deve ser finito e permitir pelo menos um frame ({1 / FPS:.6f} s).")
    if not 1 <= args.resolution_percentage <= 100 or args.render_samples < 1:
        raise ValueError("Use resolução entre 1 e 100 e pelo menos uma amostra.")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise RuntimeError("FFmpeg precisa estar instalado e acessível no PATH.")
    if ctypes.util.find_library("fluidsynth") is None:
        raise RuntimeError("Instale a biblioteca nativa libfluidsynth para sintetizar o MIDI.")

    print("A: Interpretando MIDI e calculando trajetórias...", flush=True)
    timeline = parse_midi(midi_path)
    arcs = solve_layout(timeline.timestamps, max_duration=args.max_duration)
    timestamps = (arcs[0].start_time, *(arc.end_time for arc in arcs))
    frame_count = max(
        math.ceil((timeline.end_time + args.tail_seconds) * FPS),
        math.ceil(timestamps[-1] * FPS) + 1,
    )
    if args.max_duration is not None and args.max_duration < frame_count / FPS:
        # Frames completos: a duração codificada nunca ultrapassa o limite.
        frame_count = math.floor(args.max_duration * FPS)
    duration = frame_count / FPS
    if timestamps[0] >= duration:
        raise ValueError("O limite não contém notas antes do fim dos frames completos do vídeo.")
    print(f"Plano: {len(timestamps)} contatos, {frame_count} frames, {duration:.3f} s.", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": 1,
        "fps": FPS,
        "timestamps": list(timestamps),
        "segments": [asdict(arc) for arc in arcs],
        "frame_end": frame_count,
        "descent": args.descent,
        "resolution_percentage": args.resolution_percentage,
        "render_samples": min(args.render_samples, 64),
        "max_duration": args.max_duration,
    }
    identity = cache_identity(midi_path, plan, PROJECT_ROOT / "src")
    cache_root = (args.cache_dir or output.parent / ".marble-cache").expanduser().resolve()
    width, height = render_dimensions(args.resolution_percentage)
    cache = FrameCache(cache_root / identity["key"], frame_count, width, height)
    cache.prepare(identity, force=args.force)
    plan_path = cache.directory / "scene.json"
    atomic_json(plan_path, plan)
    status = cache.inspect()
    if status.complete:
        print(f"B: Cache completo: {frame_count}/{frame_count} frames válidos. Pulando Blender.", flush=True)
    else:
        blender = shutil.which(args.blender)
        if blender is None:
            raise RuntimeError("Blender precisa estar instalado para completar o cache.")
        print(f"B: Cache incompleto: {len(status.valid)}/{frame_count} frames. Retomando Blender...", flush=True)
        cache.prune_invalid(status)
        subprocess.run(
            [blender, "--background", "--factory-startup", "--python-exit-code", "1", "--python",
             "src/blender_builder.py", "--", "--scene-json", str(plan_path),
             "--frames-dir", str(cache.frames)],
            cwd=PROJECT_ROOT, check=True,
        )
        status = cache.inspect()
        if not status.complete:
            raise RuntimeError(f"Renderização incompleta: {len(status.valid)}/{frame_count} frames válidos.")

    with TemporaryDirectory(prefix="marble-", dir=output.parent) as directory:
        work = Path(directory)
        video_path = work / "temp_video.mp4"
        audio_path = work / "temp_audio.wav"
        final_path = work / "output_final.mp4"
        print("B: Codificando sequência de frames em H.264...", flush=True)
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-framerate", str(FPS), "-start_number", "1", "-i", str(cache.frames / "frame_%06d.png"),
             "-frames:v", str(frame_count), "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", str(video_path)], check=True,
        )
        if not video_path.is_file() or video_path.stat().st_size == 0:
            raise RuntimeError("O FFmpeg não gerou o vídeo esperado.")

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
    print(f"E: Temporários removidos; frames preservados em {cache.frames}. Vídeo final: {output}", flush=True)
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
