"""Síntese MIDI offline com FluidSynth e TTS opcional em arquivo separado."""

import math
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf


def render_midi_audio(
    midi_path: str | Path,
    soundfont_path: str | Path,
    output_path: str | Path,
    *,
    sample_rate: int = 48_000,
    duration: float | None = None,
) -> Path:
    """Sintetiza o MIDI completo em WAV mono PCM de 24 bits, a 48 kHz por padrão.

    pretty_midi/FluidSynth respeita instrumentos, velocidades, controles e pitch
    bends e fornece um segundo de cauda após os últimos eventos. ``duration``
    filtra notas/eventos antes da síntese e pode recortar essa cauda ou estender
    com silêncio para coincidir com o vídeo;
    o silêncio inicial do MIDI é mantido. Não abre dispositivo de áudio.
    Requer a biblioteca nativa libfluidsynth, além dos pacotes Python.
    """
    midi_path = Path(midi_path).expanduser().resolve()
    soundfont_path = Path(soundfont_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    for path in (midi_path, soundfont_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_path in (midi_path, soundfont_path):
        raise ValueError("A saída WAV deve ser diferente dos arquivos de entrada.")
    if not isinstance(sample_rate, int) or sample_rate < 8_000:
        raise ValueError("A taxa de amostragem deve ser um inteiro de pelo menos 8000 Hz.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("A duração deve ser finita e positiva.")

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    if duration is not None:
        # Evita sintetizar uma música inteira para exportar somente uma prévia.
        # Notas já em execução são preservadas até o corte. O arquivo não muda.
        for part in midi.instruments:
            part.notes = [note for note in part.notes if note.start < duration]
            for note in part.notes:
                note.end = min(note.end, duration)
            part.control_changes = [event for event in part.control_changes if event.time < duration]
            part.pitch_bends = [event for event in part.pitch_bends if event.time < duration]
        midi.instruments = [part for part in midi.instruments if part.notes]
    if not any(part.notes for part in midi.instruments):
        raise ValueError("O MIDI não contém notas para sintetizar.")
    audio = midi.fluidsynth(fs=sample_rate, synthesizer=str(soundfont_path), normalize=False)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("A síntese produziu áudio vazio ou inválido.")
    peak = float(np.max(np.abs(audio)))
    if peak == 0:
        raise ValueError("A síntese ficou silenciosa; verifique os presets da SoundFont.")
    # Normaliza uma vez após a mistura, preservando a dinâmica e evitando clipping.
    audio *= 0.95 / peak
    if duration is not None:
        sample_count = round(duration * sample_rate)
        if sample_count < 1:
            raise ValueError("A duração precisa conter pelo menos uma amostra.")
        audio = audio[:sample_count]
        if len(audio) < sample_count:
            audio = np.pad(audio, (0, sample_count - len(audio)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate, format="WAV", subtype="PCM_24")
    return output_path


def render_tts(text: str, output_path: str | Path, *, language: str = "pt") -> Path:
    """Gera um anúncio MP3 separado via gTTS (serviço online, requer internet).

    Não é chamado pelo pipeline musical e não altera o alinhamento da música.
    """
    from gtts import gTTS

    if not text.strip():
        raise ValueError("Informe um texto para o anúncio.")
    output_path = Path(output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".mp3":
        raise ValueError("A saída TTS deve ter extensão .mp3.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gTTS(text=text, lang=language).save(str(output_path))
    return output_path
