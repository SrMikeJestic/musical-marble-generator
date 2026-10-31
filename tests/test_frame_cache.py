import tempfile
import unittest
from pathlib import Path

from png_helpers import png_bytes
from src.frame_cache import FrameCache, cache_identity, valid_png


class FrameCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache = FrameCache(self.root / "job", 3, 18, 32)
        self.identity = {"key": "job", "version": 1}
        self.cache.prepare(self.identity)

    def fill(self):
        for frame in range(1, 4):
            self.cache.frame_path(frame).write_bytes(png_bytes(18, 32))

    def test_only_exact_complete_sequence_is_valid(self):
        self.fill()
        self.assertTrue(self.cache.inspect().complete)
        self.cache.frame_path(2).rename(self.cache.frame_path(4))
        self.assertFalse(self.cache.inspect().complete)  # Três PNGs, mas falta o frame 2.

    def test_corruption_wrong_dimensions_and_truncation_are_rejected(self):
        path = self.cache.frame_path(1)
        good = png_bytes(18, 32)
        for data in (good[:-12], good[:45] + b"bad" + good[48:], png_bytes(20, 32), b""):
            with self.subTest(size=len(data)):
                path.write_bytes(data)
                self.assertFalse(valid_png(path, 18, 32))
        path.write_bytes(good)
        self.assertTrue(valid_png(path, 18, 32))

    def test_extra_png_invalidates_even_with_all_expected_frames(self):
        self.fill()
        (self.cache.frames / "extra.png").write_bytes(png_bytes(18, 32))
        status = self.cache.inspect()
        self.assertFalse(status.complete)
        self.cache.prune_invalid(status)
        self.assertTrue(self.cache.inspect().complete)

    def test_partially_written_frame_is_never_counted(self):
        self.fill()
        self.cache.frame_path(3).rename(self.cache.frame_path(3).with_suffix(".png.part"))
        self.assertEqual(self.cache.inspect().valid, {1, 2})
        self.assertFalse(self.cache.inspect().complete)

    def test_prepare_preserves_valid_cache_but_force_resets_only_current_job(self):
        self.fill()
        unrelated = self.root / "other-job"
        unrelated.mkdir()
        (unrelated / "keep.txt").write_text("keep")
        self.cache.prepare(self.identity)
        self.assertTrue(self.cache.inspect().complete)
        self.cache.prepare(self.identity, force=True)
        self.assertEqual(self.cache.inspect().valid, set())
        self.assertEqual((unrelated / "keep.txt").read_text(), "keep")

    def test_missing_or_invalid_manifest_never_reuses_frames(self):
        self.fill()
        (self.cache.directory / "cache.json").write_text("incomplete JSON")
        self.cache.prepare(self.identity)
        self.assertFalse(self.cache.inspect().complete)
        self.assertEqual(list(self.cache.frames.iterdir()), [])

    def test_midi_camera_and_plan_changes_invalidate_identity(self):
        midi = self.root / "input.mid"
        midi.write_bytes(b"midi one")
        source = self.root / "src"
        source.mkdir()
        for name in ("blender_builder.py", "config.py", "layout_solver.py", "frame_cache.py"):
            (source / name).write_text("original source")
        plan = {"frame_end": 900, "descent": 0.35}
        original = cache_identity(midi, plan, source)["key"]
        self.assertEqual(original, cache_identity(midi, plan, source)["key"])
        midi.write_bytes(b"midi two")
        self.assertNotEqual(original, cache_identity(midi, plan, source)["key"])
        midi.write_bytes(b"midi one")
        (source / "blender_builder.py").write_text("camera changed")
        self.assertNotEqual(original, cache_identity(midi, plan, source)["key"])
        (source / "blender_builder.py").write_text("original source")
        self.assertNotEqual(original, cache_identity(midi, {**plan, "frame_end": 600}, source)["key"])


if __name__ == "__main__":
    unittest.main()
