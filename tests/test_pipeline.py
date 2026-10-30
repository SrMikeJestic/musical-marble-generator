"""Execute com: python -m unittest discover -s tests -v."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mido
import numpy as np
import pretty_midi
import soundfile as sf

from src.audio_renderer import render_midi_audio, render_tts
from src.layout_solver import solve_layout
from src.main import _parser, run_pipeline
from src.midi_parser import parse_midi


def write_midi(path: Path) -> None:
    """Acorde após silêncio inicial, seguido de nota após mudança de andamento."""
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([
        mido.MetaMessage("set_tempo", tempo=500_000),
        mido.Message("note_on", note=60, velocity=100, time=120),
        mido.Message("note_on", note=64, velocity=90, time=0),
        mido.Message("note_off", note=60, time=120),
        mido.Message("note_off", note=64, time=0),
        mido.MetaMessage("set_tempo", tempo=1_000_000),
        mido.Message("note_on", note=67, velocity=100, time=240),
        mido.Message("note_off", note=67, time=240),
    ])
    midi.save(path)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="marble test ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.midi = self.root / "input test.mid"
        write_midi(self.midi)
        self.soundfont = self.root / "test.sf2"
        self.soundfont.write_bytes(b"placeholder")
        self.output = self.root / "output.mp4"

    def args(self, *extra):
        return _parser().parse_args([
            str(self.midi), "--soundfont", str(self.soundfont),
            "-o", str(self.output), *extra,
        ])

    def test_tempo_chords_initial_silence_and_note_release(self):
        timeline = parse_midi(self.midi)
        self.assertEqual(timeline.timestamps, (0.125, 0.75))
        self.assertEqual(timeline.end_time, 1.25)
        self.assertEqual(len(solve_layout(timeline.timestamps)), 1)

    def test_midi_without_two_contacts_is_rejected(self):
        for note_count in (0, 1, 2):
            midi = pretty_midi.PrettyMIDI()
            part = pretty_midi.Instrument(0)
            part.notes = [pretty_midi.Note(100, 60 + i, 0, 1) for i in range(note_count)]
            midi.instruments.append(part)
            midi.write(str(self.midi))
            with self.subTest(note_count=note_count), self.assertRaises(ValueError):
                parse_midi(self.midi)

    def test_wav_precision_duration_and_silence(self):
        output = self.root / "audio.wav"
        samples = np.concatenate((np.zeros(6000), np.ones(6000) * 0.25))
        with patch.object(pretty_midi.PrettyMIDI, "fluidsynth", return_value=samples):
            render_midi_audio(self.midi, self.soundfont, output, duration=0.5)
        info = sf.info(str(output))
        self.assertEqual((info.samplerate, info.subtype, info.frames), (48_000, "PCM_24", 24_000))
        audio, _ = sf.read(str(output))
        self.assertEqual(np.count_nonzero(audio[:6000]), 0)
        self.assertAlmostEqual(float(audio.max()), 0.95, places=6)
        self.assertEqual(np.count_nonzero(audio[12000:]), 0)

    def test_tts_is_separate_mp3_without_network_in_test(self):
        output = self.root / "announcement.mp3"
        with patch("gtts.gTTS") as tts:
            self.assertEqual(render_tts("Piano", output), output)
            tts.assert_called_once_with(text="Piano", lang="pt")
            tts.return_value.save.assert_called_once_with(str(output))
        with self.assertRaises(ValueError):
            render_tts("Piano", self.root / "wrong.wav")

    def fake_run(self, command, **kwargs):
        self.assertIsInstance(command, list)
        self.assertTrue(kwargs["check"])
        if "--scene-json" in command:
            plan_path = Path(command[command.index("--scene-json") + 1])
            plan = json.loads(plan_path.read_text())
            self.assertEqual(plan["timestamps"], [0.125, 0.75])
            self.assertEqual(plan["segments"][0]["start_time"], 0.125)
            self.assertEqual(plan["segments"][0]["end_time"], 0.75)
            # Inclui release da última nota (1.25s) + cauda (1s), não só o último contato.
            self.assertEqual(plan["frame_end"], 135)
            Path(command[command.index("--video-output") + 1]).write_bytes(b"video")
        else:
            self.assertIn("aac", command)
            self.assertEqual(command[command.index("-t") + 1], "2.250000000")
            Path(command[-1]).write_bytes(b"final mp4")
        return subprocess.CompletedProcess(command, 0)

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_pipeline_json_mux_and_cleanup(self, *_):
        with patch("src.main.subprocess.run", side_effect=self.fake_run), patch(
            "src.main.render_midi_audio"
        ) as render_audio:
            self.assertEqual(run_pipeline(self.args()), self.output)
            self.assertEqual(render_audio.call_args.kwargs["duration"], 2.25)
        self.assertEqual(self.output.read_bytes(), b"final mp4")
        self.assertEqual(list(self.root.glob("marble-*")), [])

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_failure_cleans_temporary_files_and_preserves_existing_output(self, *_):
        self.output.write_bytes(b"previous video")
        with patch("src.main.subprocess.run", side_effect=self.fake_run), patch(
            "src.main.render_midi_audio", side_effect=RuntimeError("synthesis failed")
        ), self.assertRaises(RuntimeError):
            run_pipeline(self.args("--overwrite"))
        self.assertEqual(self.output.read_bytes(), b"previous video")
        self.assertEqual(list(self.root.glob("marble-*")), [])

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_blender_failure_cleans_temporary_files(self, *_):
        with patch("src.main.subprocess.run", side_effect=subprocess.CalledProcessError(1, "blender")), \
             self.assertRaises(subprocess.CalledProcessError):
            run_pipeline(self.args())
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("marble-*")), [])

    def test_existing_output_requires_explicit_overwrite(self):
        self.output.write_bytes(b"previous video")
        with self.assertRaises(FileExistsError):
            run_pipeline(self.args())
        self.assertEqual(self.output.read_bytes(), b"previous video")

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_mux_failure_never_publishes_partial_video(self, *_):
        def fail_mux(command, **kwargs):
            if "--scene-json" in command:
                return self.fake_run(command, **kwargs)
            Path(command[-1]).write_bytes(b"partial mux")
            raise subprocess.CalledProcessError(1, command)

        with patch("src.main.subprocess.run", side_effect=fail_mux), patch(
            "src.main.render_midi_audio"
        ), self.assertRaises(subprocess.CalledProcessError):
            run_pipeline(self.args())
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("marble-*")), [])

    @unittest.skipUnless(os.environ.get("SOUNDFONT"), "Defina SOUNDFONT para testar síntese real")
    def test_real_fluidsynth_soundfont(self):
        output = self.root / "real.wav"
        render_midi_audio(self.midi, os.environ["SOUNDFONT"], output, duration=2.25)
        info = sf.info(str(output))
        self.assertEqual((info.frames, info.samplerate, info.subtype), (108_000, 48_000, "PCM_24"))
        audio, _ = sf.read(str(output))
        self.assertGreater(float(np.max(np.abs(audio[6000:]))), 0.1)
        self.assertLess(float(np.max(np.abs(audio[:5000]))), 0.001)


if __name__ == "__main__":
    unittest.main()
