"""Verifica seleção CUDA e limites sem exigir bpy/GPU no Python do CLI."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from png_helpers import png_bytes
from src.frame_cache import valid_png


class CyclesGpuTests(unittest.TestCase):
    def setUp(self):
        self.cpu = NS(id="cpu", name="CPU", type="CPU", use=True)
        self.cuda = NS(id="cuda-current", name="NVIDIA CUDA", type="CUDA", use=False)
        self.stale = NS(id="cuda-old", name="Old GPU", type="CUDA", use=True)
        self.optix = NS(id="optix", name="OptiX", type="OPTIX", use=True)
        self.preferences = NS(
            compute_device_type="NONE",
            devices=[self.cpu, self.cuda, self.stale, self.optix],
            get_devices_for_type=Mock(return_value=[self.cuda, self.cpu]),
        )
        self.addons = {"cycles": NS(preferences=self.preferences)}
        self.bpy = NS(
            types=NS(Material=object, Object=object, Scene=object),
            context=NS(preferences=NS(addons=self.addons)),
            ops=NS(preferences=NS(addon_enable=Mock())),
        )
        source = Path(__file__).resolve().parents[1] / "src" / "blender_builder.py"
        spec = importlib.util.spec_from_file_location("src._cycles_test", source)
        self.builder = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"bpy": self.bpy, "mathutils": NS(Vector=object)}):
            spec.loader.exec_module(self.builder)
        self.scene = NS(
            render=NS(engine="", use_compositing=True),
            cycles=NS(device="CPU", samples=4096, use_denoising=True, use_preview_denoising=True),
            view_layers=[
                NS(cycles=NS(use_denoising=True, denoising_store_passes=True)) for _ in range(2)
            ],
            use_nodes=True,
        )

    def test_denoising_is_disabled_for_render_preview_layers_and_compositor(self):
        self.builder.configure_cycles_gpu(self.scene)
        self.assertFalse(self.scene.cycles.use_denoising)
        self.assertFalse(self.scene.cycles.use_preview_denoising)
        self.assertTrue(all(not layer.cycles.use_denoising for layer in self.scene.view_layers))
        self.assertTrue(all(not layer.cycles.denoising_store_passes for layer in self.scene.view_layers))
        self.assertFalse(self.scene.use_nodes)
        self.assertFalse(self.scene.render.use_compositing)

    def test_only_current_cuda_devices_are_enabled(self):
        self.builder.configure_cycles_gpu(self.scene)
        self.assertEqual(self.scene.render.engine, "CYCLES")
        self.assertEqual(self.scene.cycles.device, "GPU")
        self.assertEqual(self.preferences.compute_device_type, "CUDA")
        self.preferences.get_devices_for_type.assert_called_once_with("CUDA")
        self.assertTrue(self.cuda.use)
        self.assertFalse(self.cpu.use)
        self.assertFalse(self.stale.use)
        self.assertFalse(self.optix.use)
        self.assertEqual(self.scene.cycles.samples, 64)

    def test_samples_are_capped_and_preview_samples_are_preserved(self):
        for requested, expected in ((1, 1), (8, 8), (64, 64), (512, 64)):
            with self.subTest(requested=requested):
                self.builder.configure_cycles_gpu(self.scene, requested)
                self.assertEqual(self.scene.cycles.samples, expected)

    def test_missing_cuda_never_falls_back_to_cpu_or_stale_gpu(self):
        self.preferences.get_devices_for_type.return_value = [self.cpu]
        with self.assertRaisesRegex(RuntimeError, "Nenhuma GPU CUDA"):
            self.builder.configure_cycles_gpu(self.scene)
        self.assertTrue(all(not device.use for device in self.preferences.devices))

    def test_cuda_detection_error_aborts(self):
        self.preferences.get_devices_for_type.side_effect = RuntimeError("driver unavailable")
        with self.assertRaisesRegex(RuntimeError, "Cycles CUDA indisponível"):
            self.builder.configure_cycles_gpu(self.scene)

    def test_cycles_addon_is_enabled_if_needed(self):
        self.addons.clear()
        self.bpy.ops.preferences.addon_enable.side_effect = lambda **kwargs: self.addons.update(
            cycles=NS(preferences=self.preferences)
        )
        self.builder.configure_cycles_gpu(self.scene)
        self.bpy.ops.preferences.addon_enable.assert_called_once_with(module="cycles")
        self.assertEqual(self.scene.cycles.device, "GPU")

    def test_invalid_samples_fail_before_changing_preferences(self):
        for value in (0, -1, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.builder.configure_cycles_gpu(self.scene, value)
        self.assertEqual(self.preferences.compute_device_type, "NONE")

    def test_draft_uses_optix_without_requiring_openimagedenoise(self):
        self.builder.configure_cycles_gpu(self.scene)
        self.preferences.get_devices_for_type.return_value = [self.optix, self.cpu]
        with patch.dict(sys.modules, {"_cycles": NS(with_openimagedenoise=False)}):
            self.builder.enable_draft_denoising(self.scene)
        self.assertEqual(self.scene.cycles.denoiser, "OPTIX")
        self.assertTrue(self.scene.cycles.use_denoising)
        self.assertTrue(all(layer.cycles.use_denoising for layer in self.scene.view_layers))
        self.assertEqual(self.scene.cycles.device, "GPU")
        self.assertEqual(self.preferences.compute_device_type, "CUDA")
        self.assertTrue(self.cuda.use)
        self.assertFalse(self.optix.use)
        self.assertFalse(self.cpu.use)
        self.assertFalse(self.scene.render.use_compositing)

    def test_draft_uses_oidn_only_if_build_supports_it(self):
        self.preferences.get_devices_for_type.return_value = []
        with patch.dict(sys.modules, {"_cycles": NS(with_openimagedenoise=True)}):
            self.builder.enable_draft_denoising(self.scene)
        self.assertEqual(self.scene.cycles.denoiser, "OPENIMAGEDENOISE")
        self.assertTrue(self.scene.cycles.use_denoising)

    def test_draft_reports_missing_denoiser_instead_of_silent_disable(self):
        self.preferences.get_devices_for_type.return_value = []
        with patch.dict(sys.modules, {"_cycles": NS(with_openimagedenoise=False)}), \
             self.assertRaisesRegex(RuntimeError, "nenhum dispositivo OptiX"):
            self.builder.enable_draft_denoising(self.scene)
        self.assertFalse(self.scene.cycles.use_denoising)

    def test_frame_writer_does_not_disable_draft_denoising(self):
        with tempfile.TemporaryDirectory() as directory:
            self.scene.frame_end = 1
            self.scene.frame_set = Mock()
            self.scene.render.resolution_x = 18
            self.scene.render.resolution_y = 32
            self.scene.render.image_settings = NS()

            def denoise(scene):
                scene.cycles.use_denoising = True

            def render(**kwargs):
                self.assertTrue(self.scene.cycles.use_denoising)
                Path(self.scene.render.filepath).write_bytes(png_bytes(18, 32))

            self.bpy.ops.render = NS(render=Mock(side_effect=render))
            with patch.object(self.builder, "enable_draft_denoising", side_effect=denoise) as configure:
                self.builder.render_frame_sequence(self.scene, Path(directory) / "frames", draft=True)
                configure.assert_called_once_with(self.scene)

    def test_draft_json_caps_profile_even_for_direct_builder_calls(self):
        from dataclasses import asdict
        from src.layout_solver import solve_layout

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = {"schema_version": 1, "fps": 60, "timestamps": [0, 1],
                    "segments": [asdict(arc) for arc in solve_layout([0, 1])],
                    "frame_end": 120, "descent": 0.35, "resolution_percentage": 100,
                    "render_samples": 512, "max_duration": 2, "render_mode": "draft"}
            path.write_text(json.dumps(plan))
            frames = Path(directory) / "frames"
            with patch.object(self.builder, "build_scene", return_value=self.scene) as build, \
                 patch.object(self.builder, "render_frame_sequence") as render:
                self.builder.render_scene_plan(path, frames_dir=frames)
                self.assertEqual(build.call_args.kwargs["render_samples"], 16)
                self.assertEqual((self.scene.render.resolution_x, self.scene.render.resolution_y), (270, 480))
                render.assert_called_once_with(self.scene, frames, draft=True)

    def test_frame_sequence_resumes_only_missing_or_corrupt_images(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory) / "frames"
            frames.mkdir()
            first = frames / "frame_000001.png"
            first.write_bytes(png_bytes(18, 32, 80))
            (frames / "frame_000002.png").write_bytes(b"truncated")
            self.scene.frame_end = 3
            self.scene.frame_set = Mock()
            self.scene.render.resolution_x = 18
            self.scene.render.resolution_y = 32
            self.scene.render.image_settings = NS()

            def render(**kwargs):
                self.assertEqual(kwargs, {"write_still": True})
                path = Path(self.scene.render.filepath)
                self.assertEqual(path.suffix, ".part")
                path.write_bytes(png_bytes(18, 32))

            self.bpy.ops.render = NS(render=Mock(side_effect=render))
            self.builder.render_frame_sequence(self.scene, frames)
            self.assertEqual([call.args[0] for call in self.scene.frame_set.call_args_list], [2, 3])
            self.assertEqual(first.read_bytes(), png_bytes(18, 32, 80))
            self.assertTrue(all(valid_png(frames / f"frame_{i:06d}.png", 18, 32) for i in range(1, 4)))
            self.assertEqual(list(frames.glob("*.part")), [])
            self.bpy.ops.render.render.reset_mock()
            self.builder.render_frame_sequence(self.scene, frames)
            self.bpy.ops.render.render.assert_not_called()

    def test_interruption_keeps_finished_frames_and_never_publishes_partial_png(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory) / "frames"
            self.scene.frame_end = 3
            self.scene.frame_set = Mock()
            self.scene.render.resolution_x = 18
            self.scene.render.resolution_y = 32
            self.scene.render.image_settings = NS()

            def interrupt(**kwargs):
                path = Path(self.scene.render.filepath)
                if self.scene.frame_set.call_args.args[0] == 2:
                    path.write_bytes(b"incomplete PNG")
                    raise RuntimeError("Colab disconnected")
                path.write_bytes(png_bytes(18, 32))

            self.bpy.ops.render = NS(render=Mock(side_effect=interrupt))
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                self.builder.render_frame_sequence(self.scene, frames)
            self.assertTrue(valid_png(frames / "frame_000001.png", 18, 32))
            self.assertFalse((frames / "frame_000002.png").exists())
            self.assertTrue((frames / "frame_000002.png.part").exists())

    def test_preview_renders_one_fresh_frame_without_using_animation_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame150.png"
            output.write_bytes(png_bytes(18, 32, 80))
            self.scene.frame_start = 1
            self.scene.frame_end = 900
            self.scene.frame_set = Mock()
            self.scene.render.resolution_x = 18
            self.scene.render.resolution_y = 32
            self.scene.render.image_settings = NS()

            def render(**kwargs):
                Path(self.scene.render.filepath).write_bytes(png_bytes(18, 32))

            self.bpy.ops.render = NS(render=Mock(side_effect=render))
            with patch.object(self.builder.FrameCache, "inspect", side_effect=AssertionError("cache used")):
                self.builder.render_test_frame(self.scene, 150, output)
            self.bpy.ops.render.render.assert_called_once_with(write_still=True)
            self.scene.frame_set.assert_called_once_with(150)
            self.assertEqual(output.read_bytes(), png_bytes(18, 32))

    def test_json_preview_never_dispatches_animation_renderer(self):
        from dataclasses import asdict
        from src.layout_solver import solve_layout

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            plan = {"schema_version": 1, "fps": 60, "timestamps": [0, 3],
                    "segments": [asdict(arc) for arc in solve_layout([0, 3])],
                    "frame_end": 900, "descent": 0.35, "resolution_percentage": 10,
                    "render_samples": 64, "max_duration": None}
            path.write_text(json.dumps(plan))
            output = Path(directory) / "preview.png"
            with patch.object(self.builder, "build_scene", return_value=self.scene), \
                 patch.object(self.builder, "render_test_frame") as preview, \
                 patch.object(self.builder, "render_frame_sequence", side_effect=AssertionError("animation used")):
                self.builder.render_scene_plan(path, test_frame=150, test_output=output)
                preview.assert_called_once_with(self.scene, 150, output)


if __name__ == "__main__":
    unittest.main()
