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
from png_helpers import finish_frames, png_bytes


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
            "-o", str(self.output), "--resolution-percentage", "1", *extra,
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

    def test_solver_discards_later_notes_before_calculating_positions(self):
        times = [i * 0.5 for i in range(1601)]
        limited = solve_layout(iter(times), max_duration=15.0)
        self.assertEqual(len(limited), 30)
        self.assertEqual(len(solve_layout(times)), 1600)
        self.assertEqual(limited, solve_layout(times[:31]))
        self.assertEqual(limited[-1].end_time, 15.0)
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                solve_layout(times, max_duration=value)
        with self.assertRaises(ValueError):
            solve_layout([0.0, 16.0], max_duration=15.0)

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_long_midi_json_only_contains_first_fifteen_seconds(self, *_):
        midi = pretty_midi.PrettyMIDI(resolution=960)
        part = pretty_midi.Instrument(0)
        part.notes = [pretty_midi.Note(100, 60, i * 0.5, i * 0.5 + 0.2) for i in range(1601)]
        midi.instruments.append(part)
        midi.write(str(self.midi))
        original_midi = self.midi.read_bytes()

        def render_limited(command, **kwargs):
            if "--scene-json" in command:
                plan_path = Path(command[command.index("--scene-json") + 1])
                plan = json.loads(plan_path.read_text())
                self.assertEqual(plan["max_duration"], 15.0)
                self.assertEqual(plan["timestamps"], [i * 0.5 for i in range(31)])
                self.assertEqual(len(plan["segments"]), 30)
                self.assertEqual(plan["segments"][-1]["end_time"], 15.0)
                self.assertEqual(plan["frame_end"], 900)
                self.assertLess(plan_path.stat().st_size, 12_000)
                finish_frames(command)
            elif "-framerate" in command:
                self.assertEqual(command[command.index("-frames:v") + 1], "900")
                Path(command[-1]).write_bytes(b"short video")
            else:
                self.assertEqual(command[command.index("-t") + 1], "15.000000000")
                Path(command[-1]).write_bytes(b"short final")
            return subprocess.CompletedProcess(command, 0)

        with patch("src.main.subprocess.run", side_effect=render_limited), patch(
            "src.main.render_midi_audio"
        ) as audio:
            run_pipeline(self.args("--max-duration", "15"))
            self.assertEqual(audio.call_args.kwargs["duration"], 15.0)
        self.assertEqual(self.midi.read_bytes(), original_midi)
        self.assertEqual(list(self.root.glob("marble-*")), [])

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_duration_cap_rounds_down_and_includes_boundary_contact(self, *_):
        for limit, expected_frames in ((1.01, 60), (0.75, 45)):
            def render_limited(command, **kwargs):
                if "--scene-json" in command:
                    plan = json.loads(Path(command[command.index("--scene-json") + 1]).read_text())
                    self.assertEqual(plan["timestamps"], [0.125, 0.75])
                    self.assertEqual(plan["frame_end"], expected_frames)
                    finish_frames(command)
                elif "-framerate" in command:
                    self.assertEqual(command[command.index("-frames:v") + 1], str(expected_frames))
                    Path(command[-1]).write_bytes(b"video")
                else:
                    self.assertEqual(command[command.index("-t") + 1], f"{expected_frames / 60:.9f}")
                    Path(command[-1]).write_bytes(b"final")
                return subprocess.CompletedProcess(command, 0)

            with self.subTest(limit=limit), patch("src.main.subprocess.run", side_effect=render_limited), patch(
                "src.main.render_midi_audio"
            ) as audio:
                run_pipeline(self.args("--max-duration", str(limit), "--overwrite"))
                self.assertEqual(audio.call_args.kwargs["duration"], expected_frames / 60)

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_short_or_invalid_limit_fails_before_starting_blender(self, *_):
        for limit in ("0.5", "0", "-1", "nan", "inf", "0.001"):
            with self.subTest(limit=limit), patch("src.main.subprocess.run") as run, self.assertRaises(ValueError):
                run_pipeline(self.args("--max-duration", limit))
            run.assert_not_called()
        self.assertEqual(list(self.root.glob("marble-*")), [])

    def test_audio_filters_events_before_synthesis(self):
        midi = pretty_midi.PrettyMIDI()
        part = pretty_midi.Instrument(0)
        part.notes = [pretty_midi.Note(100, 60, 0, 10), pretty_midi.Note(100, 67, 20, 21)]
        part.control_changes = [pretty_midi.ControlChange(7, 100, 0), pretty_midi.ControlChange(7, 50, 20)]
        part.pitch_bends = [pretty_midi.PitchBend(100, 0), pretty_midi.PitchBend(200, 20)]
        midi.instruments.append(part)
        midi.write(str(self.midi))
        original_midi = self.midi.read_bytes()

        def synthesize_excerpt(midi, **kwargs):
            self.assertEqual(len(midi.instruments), 1)
            notes = midi.instruments[0].notes
            self.assertEqual(len(notes), 1)
            self.assertEqual((notes[0].start, notes[0].end), (0, 1))
            self.assertEqual(len(midi.instruments[0].control_changes), 1)
            self.assertEqual(len(midi.instruments[0].pitch_bends), 1)
            return np.ones(48_000) * 0.2

        with patch.object(pretty_midi.PrettyMIDI, "fluidsynth", autospec=True, side_effect=synthesize_excerpt):
            render_midi_audio(self.midi, self.soundfont, self.root / "limited.wav", duration=1.0)
        self.assertEqual(self.midi.read_bytes(), original_midi)

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
            finish_frames(command)
        elif "-framerate" in command:
            self.assertIn("libx264", command)
            Path(command[-1]).write_bytes(b"video")
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
    def test_limit_does_not_extend_a_short_song(self, *_):
        with patch("src.main.subprocess.run", side_effect=self.fake_run), patch(
            "src.main.render_midi_audio"
        ) as audio:
            run_pipeline(self.args("--max-duration", "15"))
            self.assertEqual(audio.call_args.kwargs["duration"], 2.25)

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_complete_persistent_cache_skips_blender_even_when_unavailable(self, *_):
        drive = self.root / "Google Drive" / "render cache"
        with patch("src.main.subprocess.run", side_effect=self.fake_run), patch("src.main.render_midi_audio"):
            run_pipeline(self.args("--cache-dir", str(drive)))
        self.assertEqual(len(list(drive.glob("*/frames/*.png"))), 135)
        with patch("src.main.shutil.which", side_effect=lambda name: None if name == "blender" else "/usr/bin/ffmpeg"), \
             patch("src.main.subprocess.run", side_effect=self.fake_run) as run, \
             patch("src.main.render_midi_audio"):
            run_pipeline(self.args("--cache-dir", str(drive), "--overwrite"))
            self.assertEqual(run.call_count, 2)  # Só codificação e muxing FFmpeg.
            self.assertTrue(all("--scene-json" not in call.args[0] for call in run.call_args_list))

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_force_renders_from_zero_even_with_complete_cache(self, *_):
        with patch("src.main.subprocess.run", side_effect=self.fake_run), patch("src.main.render_midi_audio"):
            run_pipeline(self.args())
        frames = next((self.root / ".marble-cache").glob("*/frames"))
        self.assertEqual(len(list(frames.glob("*.png"))), 135)

        def forced_run(command, **kwargs):
            if "--scene-json" in command:
                self.assertEqual(list(frames.glob("*.png")), [])
            return self.fake_run(command, **kwargs)

        with patch("src.main.subprocess.run", side_effect=forced_run) as run, patch("src.main.render_midi_audio"):
            run_pipeline(self.args("--force", "--overwrite"))
            self.assertEqual(sum("--scene-json" in call.args[0] for call in run.call_args_list), 1)

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_incomplete_or_corrupt_cache_never_skips_blender(self, *_):
        with patch("src.main.subprocess.run", side_effect=self.fake_run), patch("src.main.render_midi_audio"):
            run_pipeline(self.args())
        frames = next((self.root / ".marble-cache").glob("*/frames"))
        (frames / "frame_000002.png").unlink()
        (frames / "frame_000003.png").write_bytes(b"corrupt")
        first = frames / "frame_000001.png"
        first.write_bytes(png_bytes(18, 32, 80))
        with patch("src.main.subprocess.run", side_effect=self.fake_run) as run, patch("src.main.render_midi_audio"):
            run_pipeline(self.args("--overwrite"))
            self.assertEqual(sum("--scene-json" in call.args[0] for call in run.call_args_list), 1)
        self.assertEqual(len(list(frames.glob("*.png"))), 135)
        self.assertEqual(first.read_bytes(), png_bytes(18, 32, 80))

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_successful_exit_with_too_few_frames_is_still_incomplete(self, *_):
        def incomplete(command, **kwargs):
            frames = Path(command[command.index("--frames-dir") + 1])
            (frames / "frame_000001.png").write_bytes(png_bytes(18, 32))
            return subprocess.CompletedProcess(command, 0)

        with patch("src.main.subprocess.run", side_effect=incomplete), patch("src.main.render_midi_audio") as audio:
            with self.assertRaisesRegex(RuntimeError, "Renderização incompleta: 1/135"):
                run_pipeline(self.args())
            audio.assert_not_called()
        self.assertEqual(len(list((self.root / ".marble-cache").glob("*/frames/*.png"))), 1)
        self.assertFalse(self.output.exists())

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_disconnection_preserves_partial_cache_for_next_run(self, *_):
        def disconnected(command, **kwargs):
            frames = Path(command[command.index("--frames-dir") + 1])
            (frames / "frame_000001.png").write_bytes(png_bytes(18, 32, 80))
            raise subprocess.CalledProcessError(1, command)

        with patch("src.main.subprocess.run", side_effect=disconnected), self.assertRaises(subprocess.CalledProcessError):
            run_pipeline(self.args())
        frames = next((self.root / ".marble-cache").glob("*/frames"))
        with patch("src.main.subprocess.run", side_effect=self.fake_run), patch("src.main.render_midi_audio"):
            run_pipeline(self.args())
        self.assertEqual((frames / "frame_000001.png").read_bytes(), png_bytes(18, 32, 80))
        self.assertEqual(len(list(frames.glob("*.png"))), 135)

    @patch("src.main.ctypes.util.find_library", return_value="libfluidsynth.so.3")
    @patch("src.main.shutil.which", side_effect=lambda name: f"/usr/bin/{name}")
    def test_camera_or_midi_changes_never_reuse_old_images(self, *_):
        project = self.root / "project"
        (project / "src").mkdir(parents=True)
        original_source = Path(__file__).resolve().parents[1] / "src"
        for name in ("blender_builder.py", "config.py", "layout_solver.py", "frame_cache.py"):
            (project / "src" / name).write_bytes((original_source / name).read_bytes())
        with patch("src.main.PROJECT_ROOT", project), patch("src.main.render_midi_audio"):
            with patch("src.main.subprocess.run", side_effect=self.fake_run):
                run_pipeline(self.args())
            camera_source = project / "src" / "blender_builder.py"
            camera_source.write_text(camera_source.read_text() + "\n# Camera changed\n")
            with patch("src.main.subprocess.run", side_effect=self.fake_run) as run:
                run_pipeline(self.args("--overwrite"))
                self.assertTrue(any("--scene-json" in call.args[0] for call in run.call_args_list))
            midi = pretty_midi.PrettyMIDI(str(self.midi))
            midi.instruments[0].notes[0].pitch += 1
            midi.write(str(self.midi))
            with patch("src.main.subprocess.run", side_effect=self.fake_run) as run:
                run_pipeline(self.args("--overwrite"))
                self.assertTrue(any("--scene-json" in call.args[0] for call in run.call_args_list))
        self.assertEqual(len(list((self.root / ".marble-cache").glob("*/cache.json"))), 3)

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
        self.assertEqual(len(list((self.root / ".marble-cache").glob("*/frames/*.png"))), 135)

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
            if "--scene-json" in command or "-framerate" in command:
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

    @unittest.skipUnless(os.environ.get("SOUNDFONT"), "Defina SOUNDFONT para testar síntese real")
    def test_real_fluidsynth_only_synthesizes_excerpt_of_long_midi(self):
        midi = pretty_midi.PrettyMIDI(resolution=960)
        part = pretty_midi.Instrument(0)
        part.notes = [pretty_midi.Note(100, 60, i * 0.5, i * 0.5 + 0.2) for i in range(1601)]
        midi.instruments.append(part)
        midi.write(str(self.midi))
        original_synth = pretty_midi.PrettyMIDI.fluidsynth

        def synthesize_excerpt(midi, **kwargs):
            notes = [note for part in midi.instruments for note in part.notes]
            # A nota em 15 s é uma âncora visual; seu som começaria após o corte.
            self.assertEqual(len(notes), 30)
            self.assertTrue(all(note.start < 15 and note.end <= 15 for note in notes))
            return original_synth(midi, **kwargs)

        output = self.root / "real_excerpt.wav"
        with patch.object(pretty_midi.PrettyMIDI, "fluidsynth", autospec=True, side_effect=synthesize_excerpt):
            render_midi_audio(self.midi, os.environ["SOUNDFONT"], output, duration=15)
        info = sf.info(str(output))
        self.assertEqual((info.frames, info.samplerate, info.subtype), (720_000, 48_000, "PCM_24"))


if __name__ == "__main__":
    unittest.main()
