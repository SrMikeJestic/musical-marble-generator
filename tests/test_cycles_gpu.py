"""Verifica seleção CUDA e limites sem exigir bpy/GPU no Python do CLI."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch


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


if __name__ == "__main__":
    unittest.main()
