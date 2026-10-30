"""Converte MIDI em tempos absolutos, incluindo mudanças de andamento."""

from dataclasses import dataclass
from pathlib import Path

import pretty_midi


@dataclass(frozen=True)
class MidiTimeline:
    timestamps: tuple[float, ...]
    end_time: float


def parse_midi(midi_path: str | Path) -> MidiTimeline:
    """Agrupa acordes em um contato; preserva silêncio inicial e tempo original.

    Todos os instrumentos, inclusive percussão, contribuem com contatos. O MIDI
    original permanece intacto para a síntese polifônica do áudio.
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    timestamps = tuple(sorted({note.start for part in midi.instruments for note in part.notes}))
    if len(timestamps) < 2:
        raise ValueError("O MIDI precisa de pelo menos dois instantes distintos de notas.")
    return MidiTimeline(timestamps=timestamps, end_time=float(midi.get_end_time()))
