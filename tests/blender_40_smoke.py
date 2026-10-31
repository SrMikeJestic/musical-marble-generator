"""Teste de integração para o binário exato usado no Google Colab.

blender --background --python-exit-code 1 --python tests/blender_40_smoke.py

Exige GPU NVIDIA CUDA. Constrói materiais/geometria, verifica contatos e renderiza um MP4
curto via JSON. Os arquivos de teste ficam em um diretório temporário.
"""

import json
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blender_builder import BALL_RADIUS, build_scene, render_scene_plan
from src.layout_solver import solve_layout


def main() -> None:
    if bpy.app.version != (4, 0, 2):
        raise RuntimeError(f"Este teste exige Blender 4.0.2; encontrado: {bpy.app.version_string}")

    times = [0.0, 0.025, 0.05]
    # Um JSON antigo pode conter a música inteira; o builder também deve filtrar
    # antes de instanciar os mais de 1600 contatos fora da prévia.
    full_times = times + [float(i) for i in range(1, 1602)]
    arcs = solve_layout(full_times)
    scene = build_scene(full_times, segments=arcs, frame_end=5, max_duration=0.05)
    assert scene.frame_end == 3
    assert len([obj for obj in scene.objects if obj.name.startswith("Platform_")]) == 3
    assert len([obj for obj in scene.objects if obj.name.startswith("Pin_")]) == 3
    with tempfile.TemporaryDirectory(prefix="blender-40-smoke-") as directory:
        root = Path(directory)
        plan = {
            "schema_version": 1,
            "fps": 60,
            "timestamps": full_times,
            "segments": [asdict(arc) for arc in arcs],
            "frame_end": 5,
            "descent": 0.35,
            "resolution_percentage": 10,
            "render_samples": 8,
            "max_duration": 0.05,
        }
        plan_path = root / "scene.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        video_path = root / "video.mp4"
        render_scene_plan(plan_path, video_path)
        assert video_path.is_file() and video_path.stat().st_size > 0
        scene = bpy.context.scene
        assert scene.frame_end == 3
        assert len([obj for obj in scene.objects if obj.name.startswith("Platform_")]) == 3
        assert len([obj for obj in scene.objects if obj.name.startswith("Pin_")]) == 3
        assert scene.render.engine == "CYCLES"
        assert scene.cycles.device == "GPU"
        assert scene.cycles.samples == 8
        preferences = bpy.context.preferences.addons["cycles"].preferences
        assert preferences.compute_device_type == "CUDA"
        assert any(device.type == "CUDA" and device.use for device in preferences.devices)
        assert not any(device.type != "CUDA" and device.use for device in preferences.devices)
        assert (scene.render.resolution_x, scene.render.resolution_y) == (108, 192)
        assert scene.render.fps == 60 and scene.render.fps_base == 1.0
        assert scene.render.ffmpeg.codec == "H264"
        assert scene.camera.constraints[0].target == scene.objects["Marble"]
        assert all(obj.rigid_body is None for obj in scene.objects)

        ball = scene.objects["Marble"]
        glass = ball.data.materials[0]
        assert any(node.type == "BSDF_TRANSPARENT" for node in glass.node_tree.nodes)
        assert all(polygon.use_smooth for polygon in ball.data.polygons)
        chrome = scene.objects["Pin_000"].data.materials[0]
        shader = next(node for node in chrome.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        assert shader.inputs["Metallic"].default_value == 1.0
        for i, timestamp in enumerate(times):
            frame = 1 + timestamp * 60
            scene.frame_set(math.floor(frame), subframe=frame % 1)
            platform = scene.objects[f"Platform_{i:03d}"]
            assert abs(ball.location.x - platform.location.x) < 1e-5
            top = platform.location.z + platform.dimensions.z / 2
            assert abs(ball.location.z - BALL_RADIUS - top) < 1e-5
        for obj in (ball, scene.camera):
            for curve in obj.animation_data.action.fcurves:
                assert all(key.interpolation == "LINEAR" for key in curve.keyframe_points)
                assert any(abs(key.co.x - 2.5) < 1e-5 for key in curve.keyframe_points)
        bpy.ops.wm.save_as_mainfile(filepath=str(root / "scene.blend"))
        assert (root / "scene.blend").is_file()
    print("Blender 4.0.2 CUDA OK: materiais, geometria, câmera, frames fracionários, .blend e H.264.")


if __name__ == "__main__":
    main()
