"""Constrói e anima uma cena Cycles CUDA inteiramente procedural, sem Rigid Body.

Execute da raiz do projeto com Blender 4.0.2 e uma GPU NVIDIA CUDA disponível::

    blender --background --python src/blender_builder.py

O exemplo salva ``test_scene.blend`` na raiz do projeto, sem renderizar vídeo.
Com ``-- --scene-json plano.json --video-output video.mp4``, consome o plano do
orquestrador e renderiza a animação H.264, sem criar o arquivo de teste.
Os tempos absolutos do solver são preservados: frame = 1 + timestamp * FPS.
Há chaves em todos os frames e nos contatos, inclusive em frames fracionários.

O solver fornece pontos de contato. O builder soma o raio ao Z da esfera e
aplica uma inclinação configurável: z_visual = z_solver - descent * (x - x0).
Como X é linear no tempo, essa transformação mantém os arcos parabólicos e a
aceleração vertical. Use descent=0 para visualizar o layout horizontal original.
"""

import argparse
import json
import math
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector

# --python executa este arquivo sem contexto de pacote. Permite também importá-lo
# normalmente com `from src.blender_builder import build_scene` dentro do Blender.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import FPS, RESOLUTION_PERCENTAGE, RESOLUTION_X, RESOLUTION_Y
    from src.layout_solver import TrajectorySegment, solve_layout
else:
    from .config import FPS, RESOLUTION_PERCENTAGE, RESOLUTION_X, RESOLUTION_Y
    from .layout_solver import TrajectorySegment, solve_layout


BALL_RADIUS = 0.16
WALL_Y = 1.4
PLATFORM_DEPTH = 0.8
PLATFORM_THICKNESS = 0.18
MAX_RENDER_SAMPLES = 64
PALETTE = (
    (0.95, 0.20, 0.16, 1.0),
    (1.00, 0.61, 0.10, 1.0),
    (0.13, 0.64, 0.54, 1.0),
    (0.18, 0.44, 0.85, 1.0),
    (0.60, 0.28, 0.78, 1.0),
)


def disable_denoising(scene: bpy.types.Scene) -> None:
    """Dispensa denoisers no Blender 4.0.2, inclusive builds sem OpenImageDenoise."""
    scene.cycles.use_denoising = False
    scene.cycles.use_preview_denoising = False
    for layer in scene.view_layers:
        layer.cycles.use_denoising = False
        layer.cycles.denoising_store_passes = False
    # Ignora também árvores de composição herdadas do arquivo de inicialização,
    # inclusive nós Denoise dentro de grupos. Nenhum compositor será executado.
    scene.use_nodes = False
    scene.render.use_compositing = False


def configure_cycles_gpu(scene: bpy.types.Scene, samples: int = MAX_RENDER_SAMPLES) -> None:
    """Exige CUDA, desativa CPU e limita o Cycles a no máximo 64 samples."""
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("O número de amostras deve ser um inteiro positivo.")
    if "cycles" not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module="cycles")
    if "cycles" not in bpy.context.preferences.addons:
        raise RuntimeError("O Blender precisa ter o addon Cycles disponível.")
    scene.render.engine = "CYCLES"
    disable_denoising(scene)
    preferences = bpy.context.preferences.addons["cycles"].preferences
    try:
        preferences.compute_device_type = "CUDA"
        cuda_devices = [
            device for device in preferences.get_devices_for_type("CUDA") if device.type == "CUDA"
        ]
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("Cycles CUDA indisponível. Ative uma GPU NVIDIA no runtime do Colab.") from exc

    # Preferências salvas podem conter GPUs de outra sessão. Ative somente IDs
    # detectados agora, excluindo CPU, outros backends e dispositivos antigos.
    available_ids = {device.id for device in cuda_devices}
    for device in preferences.devices:
        device.use = device.type == "CUDA" and device.id in available_ids
    if not cuda_devices:
        raise RuntimeError("Nenhuma GPU CUDA detectada pelo Cycles. Ative uma GPU NVIDIA no Colab.")
    scene.cycles.device = "GPU"
    scene.cycles.samples = min(samples, MAX_RENDER_SAMPLES)
    names = ", ".join(device.name for device in cuda_devices)
    print(f"Cycles CUDA: {names}; máximo de {scene.cycles.samples} samples.", flush=True)


def create_pbr_material(
    name: str,
    color: tuple[float, float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.5,
    transmission: float = 0.0,
) -> bpy.types.Material:
    """Material PBR do Blender 4.0.2, com transparência de superfície para vidro.

    ``transmission`` mistura o Principled com Transparent BSDF, aproximando
    vidro translúcido sem refração óptica.
    """
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    material.diffuse_color = color
    nodes = material.node_tree.nodes
    nodes.clear()
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["IOR"].default_value = 1.45
    output = nodes.new("ShaderNodeOutputMaterial")
    if transmission:
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = 0.7 * transmission
        material.node_tree.links.new(shader.outputs["BSDF"], mix.inputs[1])
        material.node_tree.links.new(transparent.outputs["BSDF"], mix.inputs[2])
        material.node_tree.links.new(mix.outputs[0], output.inputs["Surface"])
    else:
        material.node_tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return material


def assign_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def _linear_keyframes(obj: bpy.types.Object) -> None:
    """Evita overshoot Bezier entre amostras e nos impactos da bolinha."""
    for curve in obj.animation_data.action.fcurves:
        for key in curve.keyframe_points:
            key.interpolation = "LINEAR"


def _make_platform(
    index: int,
    x: float,
    contact_z: float,
    width: float,
    material: bpy.types.Material,
    chrome: bpy.types.Material,
) -> None:
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=(x, 0.0, contact_z - PLATFORM_THICKNESS / 2)
    )
    platform = bpy.context.object
    platform.name = f"Platform_{index:03d}"
    platform.dimensions = (width, PLATFORM_DEPTH, PLATFORM_THICKNESS)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = platform.modifiers.new("Soft edges", "BEVEL")
    bevel.width = min(0.025, width / 8)
    bevel.segments = 3
    assign_material(platform, material)

    # Cilindro alinhado em Y, da face traseira da tecla até a parede XZ.
    pin_start_y = PLATFORM_DEPTH / 2
    pin_length = WALL_Y - pin_start_y
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32,
        radius=min(0.045, width / 4),
        depth=pin_length,
        location=(x, (WALL_Y + pin_start_y) / 2, contact_z - PLATFORM_THICKNESS / 2),
        rotation=(math.pi / 2, 0.0, 0.0),
    )
    pin = bpy.context.object
    pin.name = f"Pin_{index:03d}"
    for polygon in pin.data.polygons:
        polygon.use_smooth = len(polygon.vertices) == 4
    assign_material(pin, chrome)


def _setup_lighting(scene: bpy.types.Scene) -> None:
    world = bpy.data.worlds.new("MarbleWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.55, 0.65, 0.80, 1.0
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    scene.world = world

    bpy.ops.object.light_add(type="SUN", location=(-4.0, -6.0, 10.0))
    sun = bpy.context.object
    sun.name = "KeySun"
    sun.data.energy = 3.0
    sun.data.angle = math.radians(1.0)
    sun.data.use_shadow = True
    # Raios viajam para a parede (+Y) e para baixo (-Z).
    sun.rotation_euler = Vector((0.45, 1.0, -0.7)).to_track_quat("-Z", "Y").to_euler()

    # Preenchimento amplo acompanha a câmera e cria reflexos no vidro/cromo.
    bpy.ops.object.light_add(type="AREA")
    fill = bpy.context.object
    fill.name = "CameraFill"
    fill.data.energy = 500.0
    fill.data.shape = "DISK"
    fill.data.size = 5.0


def build_scene(
    timestamps: Iterable[float],
    *,
    descent: float = 0.35,
    segments: Iterable[TrajectorySegment] | None = None,
    frame_end: int | None = None,
    max_duration: float | None = None,
    render_samples: int = MAX_RENDER_SAMPLES,
) -> bpy.types.Scene:
    """Limpa a cena ativa e constrói o layout a partir de timestamps em segundos.

    ``descent`` é a queda em Z por metro em X, finita e não negativa. A câmera
    acompanha X e a descida de base em Z, sem copiar os saltos verticais; uma
    constraint mantém a esfera enquadrada. ``segments`` recebe a matemática já
    calculada no JSON; ``frame_end`` inclui a duração das notas e a cauda sonora.
    ``max_duration`` descarta contatos e segmentos posteriores ao limite antes
    de construir qualquer geometria e limita também o intervalo de renderização.
    ``render_samples`` limita as amostras do Cycles CUDA a no máximo 64 por frame.
    Nenhum corpo rígido é criado.
    """
    if not math.isfinite(descent) or descent < 0:
        raise ValueError("A inclinação deve ser finita e não negativa.")
    if max_duration is not None and (
        not math.isfinite(max_duration) or max_duration < 1 / FPS
    ):
        raise ValueError("A duração máxima deve ser finita e permitir pelo menos um frame.")
    times = tuple(timestamps)
    arcs = list(segments) if segments is not None else solve_layout(times, max_duration=max_duration)
    if max_duration is not None:
        times = tuple(t for t in times if t <= max_duration)
        arcs = [arc for arc in arcs if arc.end_time <= max_duration]
    if not arcs or (arcs[0].start_time, *(arc.end_time for arc in arcs)) != times:
        raise ValueError("Os segmentos devem corresponder aos timestamps do plano.")
    minimum_frame_end = math.ceil(arcs[-1].end_time * FPS) + 1
    if max_duration is not None:
        # O contato no instante exato do corte pode ter uma chave logo após o
        # último frame renderizado. Ela ancora o arco, sem acrescentar vídeo.
        max_frame_end = math.floor(max_duration * FPS)
        minimum_frame_end = min(minimum_frame_end, max_frame_end)
        if isinstance(frame_end, int):
            frame_end = min(frame_end, max_frame_end)
    if frame_end is not None and (not isinstance(frame_end, int) or frame_end < minimum_frame_end):
        raise ValueError("O frame final precisa incluir todos os contatos.")
    scene = bpy.context.scene
    configure_cycles_gpu(scene, render_samples)
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    scene.render.fps = FPS
    scene.render.fps_base = 1.0
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.resolution_percentage = RESOLUTION_PERCENTAGE
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.image_settings.file_format = "PNG"
    scene.unit_settings.system = "METRIC"
    scene.view_settings.view_transform = "AgX"
    scene.frame_start = 1
    scene.frame_end = minimum_frame_end if frame_end is None else frame_end

    glass = create_pbr_material(
        "BallGlass", (0.62, 0.87, 0.95, 1.0), roughness=0.10, transmission=1.0
    )
    chrome = create_pbr_material(
        "PinChrome", (0.72, 0.76, 0.82, 1.0), metallic=1.0, roughness=0.18
    )
    wall_material = create_pbr_material(
        "PastelWall", (0.64, 0.73, 0.80, 1.0), roughness=0.90
    )
    plastics = [
        create_pbr_material(f"PlatformPlastic_{i}", color, roughness=0.65)
        for i, color in enumerate(PALETTE)
    ]

    origin_x = arcs[0].start_x

    def contact_z(x: float, z: float) -> float:
        return z - descent * (x - origin_x)

    contacts = [(arcs[0].start_time, arcs[0].start_x, arcs[0].platform_z)]
    contacts.extend((arc.end_time, arc.end_x, arc.platform_z) for arc in arcs)
    for i, (_, x, z) in enumerate(contacts):
        gaps = []
        if i:
            gaps.append(x - contacts[i - 1][1])
        if i + 1 < len(contacts):
            gaps.append(contacts[i + 1][1] - x)
        width = min(0.9, 0.6 * min(gaps))
        _make_platform(i, x, contact_z(x, z), width, plastics[i % len(plastics)], chrome)

    # Margens cobrem o enquadramento da câmera e as sombras no início/fim.
    min_z = min(contact_z(x, z) for _, x, z in contacts) - 10.0
    max_z = max(
        contact_z(arc.start_x, arc.platform_z) + arc.gravity * arc.duration**2 / 8
        for arc in arcs
    ) + 10.0
    width = arcs[-1].end_x - origin_x + 20.0
    bpy.ops.mesh.primitive_plane_add(
        size=1.0,
        location=((origin_x + arcs[-1].end_x) / 2, WALL_Y, (min_z + max_z) / 2),
        rotation=(math.pi / 2, 0.0, 0.0),
    )
    wall = bpy.context.object
    wall.name = "BackdropWall"
    wall.scale = (width, max_z - min_z, 1.0)
    assign_material(wall, wall_material)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=BALL_RADIUS)
    ball = bpy.context.object
    ball.name = "Marble"
    for polygon in ball.data.polygons:
        polygon.use_smooth = True
    assign_material(ball, glass)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "FollowCamera"
    camera.data.type = "PERSP"
    camera.data.lens = 40.0
    camera.data.clip_start = 0.05
    camera.data.clip_end = max(200.0, width * 2, max_z - min_z)
    camera.data.dof.use_dof = False
    tracking = camera.constraints.new("TRACK_TO")
    tracking.target = ball
    tracking.track_axis = "TRACK_NEGATIVE_Z"
    tracking.up_axis = "UP_Y"
    scene.camera = camera

    _setup_lighting(scene)
    fill = scene.objects["CameraFill"]
    fill.parent = camera
    fill.location = (-3.0, 3.0, -1.0)
    fill.rotation_euler = (0.0, 0.0, 0.0)

    # Incluir contatos fracionários evita deslocar notas que não caem em 1/60 s.
    keyframes = {float(frame) for frame in range(scene.frame_start, scene.frame_end + 1)}
    keyframes.update(1 + t * FPS for t, _, _ in contacts)
    end_times = [arc.end_time for arc in arcs]
    for frame in sorted(keyframes):
        timestamp = min(max((frame - 1) / FPS, arcs[0].start_time), arcs[-1].end_time)
        arc: TrajectorySegment = arcs[min(bisect_right(end_times, timestamp), len(arcs) - 1)]
        x, z = arc.position_at(timestamp)
        ball.location = (x, 0.0, contact_z(x, z) + BALL_RADIUS)
        ball.keyframe_insert(data_path="location", frame=frame)
        camera.location = (x, -10.0, contact_z(x, arc.platform_z) + 3.0)
        camera.keyframe_insert(data_path="location", frame=frame)
    _linear_keyframes(ball)
    _linear_keyframes(camera)

    for i, (timestamp, _, _) in enumerate(contacts):
        # Marcadores são inteiros no Blender; as chaves mantêm o tempo exato.
        scene.timeline_markers.new(f"Note_{i:03d}", frame=round(1 + timestamp * FPS))
    scene.frame_set(scene.frame_start)
    bpy.ops.object.select_all(action="DESELECT")
    ball.select_set(True)
    bpy.context.view_layer.objects.active = ball
    return scene


def render_scene_plan(plan_path: Path, video_path: Path) -> None:
    """Consome a matemática calculada pelo CLI e renderiza um MP4 sem áudio."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["schema_version"] != 1 or plan["fps"] != FPS:
        raise ValueError("Versão do plano ou FPS incompatíveis com este builder.")
    percentage = plan["resolution_percentage"]
    samples = plan["render_samples"]
    if not isinstance(percentage, int) or not 1 <= percentage <= 100:
        raise ValueError("A escala de resolução deve ser um inteiro entre 1 e 100.")
    if not isinstance(samples, int) or samples < 1:
        raise ValueError("O número de amostras deve ser um inteiro positivo.")
    scene = build_scene(
        plan["timestamps"], descent=plan["descent"],
        segments=[TrajectorySegment(**segment) for segment in plan["segments"]],
        frame_end=plan["frame_end"],
        max_duration=plan.get("max_duration"),
        render_samples=samples,
    )
    # H.264 exige dimensões pares. Múltiplos de 18x32 preservam também o 9:16
    # em escalas pequenas, cuja porcentagem pode produzir dimensões ímpares.
    scale_unit = max(2, 2 * round((RESOLUTION_X / 9) * percentage / 200))
    scene.render.resolution_x = 9 * scale_unit
    scene.render.resolution_y = 16 * scale_unit
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.audio_codec = "NONE"
    video_path = video_path.expanduser().resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(video_path)
    disable_denoising(scene)
    bpy.ops.render.render(animation=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrói cena de teste ou renderiza plano JSON.")
    parser.add_argument("--scene-json", type=Path)
    parser.add_argument("--video-output", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.scene_json is not None or args.video_output is not None:
        if args.scene_json is None or args.video_output is None:
            parser.error("--scene-json e --video-output devem ser usados juntos.")
        render_scene_plan(args.scene_json, args.video_output)
    else:
        build_scene([0.0, 0.5, 1.1, 1.55, 2.2, 2.7, 3.25, 3.8])
        output_path = Path(__file__).resolve().parents[1] / "test_scene.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
        print(f"Cena de teste salva em: {output_path}")


if __name__ == "__main__":
    main()
