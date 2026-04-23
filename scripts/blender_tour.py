"""Blender-driven ARTEMIS capability tour.

Invoked as::

    blender -b --python scripts/blender_tour.py -- --config /tmp/tour.json

The config JSON carries everything Blender needs: VRM path, output
frame dir, capability timeline, and the per-frame mouth envelope.
The outer harness (``scripts/run_avatar_blender_tour.py``) produces
the JSON from the XTTS audio + the timeline definition, runs this
script, and muxes the rendered frames with the audio via NVENC.

No external dependencies beyond Blender's bundled bpy. Runs headless.
"""

import argparse
import json
import os
import sys

import bpy

# ─────────────────────────────────────────────────────────────────
# VRM shape-key + bone names (VRoid standard, Fcl_* prefix).
# ─────────────────────────────────────────────────────────────────

FACE_MESH = "Face"
KEY_PREFIX = "Face_Blendshape.Fcl_"

EMOTION_KEYS = {
    "neutral": None,  # zero everything
    "happy": "ALL_Joy",
    "sad": "ALL_Sorrow",
    "angry": "ALL_Angry",
    "surprised": "ALL_Surprised",
    "relaxed": "ALL_Fun",
}

VISEME_KEYS = {
    "aa": "MTH_A",
    "ih": "MTH_I",
    "ou": "MTH_U",
    "ee": "MTH_E",
    "oh": "MTH_O",
}

BLINK_KEY = "EYE_Close"

# Bone rotations are applied to these VRM humanoid bones. Values are
# *delta* rotations from rest (pose-bone local space, Euler XYZ).
BONE = {
    "head": "J_Bip_C_Head",
    "neck": "J_Bip_C_Neck",
    "spine": "J_Bip_C_Spine",
    "leftUpperArm": "J_Bip_L_UpperArm",
    "rightUpperArm": "J_Bip_R_UpperArm",
    "leftLowerArm": "J_Bip_L_LowerArm",
    "rightLowerArm": "J_Bip_R_LowerArm",
}

# 5 canonical poses — same intent as the Playwright side, but with
# angles tuned to VRM rest pose (T-pose, arms straight out). Positive
# Z on an upper-arm bone rotates the arm DOWN from the side; negative
# Z rotates it UP above the head.
POSES = {
    "mountain": {
        "leftUpperArm": (0, 0, 1.3),
        "rightUpperArm": (0, 0, -1.3),
    },
    "reach": {
        "leftUpperArm": (0, 0, -0.8),
        "rightUpperArm": (0, 0, 0.8),
        "spine": (-0.08, 0, 0),
        "neck": (-0.12, 0, 0),
    },
    "warrior": {
        "leftUpperArm": (0, 0, 0),  # stays at T-pose arms-out
        "rightUpperArm": (0, 0, 0),
        "spine": (0, 0.12, 0),
    },
    "prayer": {
        "leftUpperArm": (0, 0, 0.6),
        "rightUpperArm": (0, 0, -0.6),
        "leftLowerArm": (0, -1.1, 0),
        "rightLowerArm": (0, 1.1, 0),
        "spine": (0.06, 0, 0),
        "head": (0.15, 0, 0),
    },
    "ping": {
        "leftUpperArm": (0, 0, -0.6),
        "rightUpperArm": (0, 0, -1.1),
        "spine": (0, -0.1, 0.08),
        "head": (0, 0, -0.12),
    },
}


# ─────────────────────────────────────────────────────────────────
# Scene setup — camera, lighting, world
# ─────────────────────────────────────────────────────────────────


def _setup_scene(cfg: dict) -> None:
    scn = bpy.context.scene
    scn.render.resolution_x = cfg["width"]
    scn.render.resolution_y = cfg["height"]
    scn.render.resolution_percentage = 100
    scn.render.fps = cfg["fps"]
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGBA"
    scn.render.filepath = cfg["frames_pattern"]

    # Eevee with low samples — capability-tour quality, not film.
    scn.render.engine = "BLENDER_EEVEE"
    scn.eevee.taa_render_samples = 4
    scn.eevee.use_bloom = True

    # World background — matches the table UI slate-blue-black.
    if scn.world is None:
        scn.world = bpy.data.worlds.new("World")
    scn.world.use_nodes = True
    bg = scn.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.02, 0.035, 0.08, 1.0)
    bg.inputs["Strength"].default_value = 1.0

    # Camera — head/torso framing.
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Cam", cam_data)
    scn.collection.objects.link(cam)
    cam.location = (0.0, -2.4, 1.5)
    cam.rotation_euler = (1.4, 0.0, 0.0)
    cam_data.lens = 50
    scn.camera = cam

    # 3-point lighting.
    for name, kind, loc, energy, color in [
        ("Key", "AREA", (2.0, -2.0, 3.0), 500.0, (1, 1, 1)),
        ("Fill", "AREA", (-3.0, -1.0, 2.0), 200.0, (0.8, 0.9, 1.0)),
        ("Rim", "AREA", (0.0, 2.0, 3.0), 400.0, (1.0, 0.85, 0.7)),
    ]:
        d = bpy.data.lights.new(name, type=kind)
        d.energy = energy
        d.color = color
        o = bpy.data.objects.new(name, d)
        o.location = loc
        scn.collection.objects.link(o)


# ─────────────────────────────────────────────────────────────────
# Keyframing helpers
# ─────────────────────────────────────────────────────────────────


def _face() -> bpy.types.Object:
    face = bpy.data.objects.get(FACE_MESH)
    if face is None:
        raise RuntimeError(f"VRM import produced no {FACE_MESH!r} mesh")
    return face


def _keyframe_shape(face, key: str, value: float, frame: int) -> None:
    """Set + keyframe a single shape key by its Fcl_ suffix."""
    sk = face.data.shape_keys.key_blocks.get(KEY_PREFIX + key)
    if sk is None:
        return  # missing on this VRM, skip
    sk.value = float(value)
    sk.keyframe_insert(data_path="value", frame=int(frame))


def _keyframe_bone(arm, bone_alias: str, euler, frame: int) -> None:
    name = BONE.get(bone_alias)
    if name is None:
        return
    b = arm.pose.bones.get(name)
    if b is None:
        return
    b.rotation_mode = "XYZ"
    b.rotation_euler = euler
    b.keyframe_insert(data_path="rotation_euler", frame=int(frame))


def _apply_pose_keyframe(arm, pose_name: str, frame: int) -> None:
    """Keyframe every bone in the target pose AND zero the ones
    not in it (so we don't accumulate deltas from prior poses)."""
    target = POSES.get(pose_name, {})
    for alias in BONE:
        euler = target.get(alias, (0.0, 0.0, 0.0))
        _keyframe_bone(arm, alias, euler, frame)


def _apply_expression_keyframe(face, name: str, frame: int) -> None:
    """Binary emotion: set the named one to 1.0, others to 0.0."""
    for k, suffix in EMOTION_KEYS.items():
        if suffix is None:
            continue
        val = 1.0 if k == name else 0.0
        _keyframe_shape(face, suffix, val, frame)


# ─────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────


def _build_animation(cfg: dict) -> None:
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    face = _face()
    fps = cfg["fps"]

    # Start from a known-neutral baseline at frame 1.
    _apply_expression_keyframe(face, "neutral", 1)
    for suffix in VISEME_KEYS.values():
        _keyframe_shape(face, suffix, 0.0, 1)
    _keyframe_shape(face, BLINK_KEY, 0.0, 1)
    _apply_pose_keyframe(arm, "mountain", 1)

    # Timeline events.
    for ev in cfg.get("timeline", []):
        t, kind, arg = ev["t"], ev["kind"], ev["arg"]
        frame = max(1, int(round(t * fps)))
        if kind == "expression":
            _apply_expression_keyframe(face, str(arg), frame)
        elif kind == "pose":
            if arg is None:
                continue
            if isinstance(arg, int):
                name = list(POSES.keys())[arg % len(POSES)]
            else:
                name = str(arg)
            _apply_pose_keyframe(arm, name, frame)
        elif kind == "viseme":
            suffix = VISEME_KEYS.get(arg[0])
            if suffix:
                _keyframe_shape(face, suffix, float(arg[1]), frame)

    # Mouth envelope per frame — drive the 'A' viseme amplitude.
    aa_suffix = VISEME_KEYS["aa"]
    envelope = cfg.get("envelope", [])
    for i, level in enumerate(envelope):
        _keyframe_shape(face, aa_suffix, float(level), i + 1)

    # Randomized blinks on a coarse ~3 s cadence.
    import random as _r

    _r.seed(cfg.get("blink_seed", 42))
    t_blink = 1.5
    total_s = len(envelope) / fps if envelope else cfg.get("total_s", 30)
    while t_blink < total_s:
        f = int(round(t_blink * fps))
        _keyframe_shape(face, BLINK_KEY, 0.0, max(1, f - 1))
        _keyframe_shape(face, BLINK_KEY, 1.0, f)
        _keyframe_shape(face, BLINK_KEY, 0.0, f + int(fps * 0.15))
        t_blink += _r.uniform(3.0, 5.0)


def _configure_gpu() -> None:
    cprefs = bpy.context.preferences.addons.get("cycles")
    if cprefs is None:
        return
    prefs = cprefs.preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type in ("CUDA", "OPTIX")


def main() -> int:
    # Blender strips its own args; ours start after the first "--".
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = json.load(f)

    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.import_scene.vrm(filepath=cfg["vrm_path"])
    _setup_scene(cfg)
    _configure_gpu()
    _build_animation(cfg)

    scn = bpy.context.scene
    n_frames = max(1, len(cfg.get("envelope", [])))
    scn.frame_start = 1
    scn.frame_end = n_frames

    # Ensure the output dir exists.
    out_dir = os.path.dirname(cfg["frames_pattern"])
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"BLENDER_TOUR: rendering {n_frames} frames "
        f"@ {cfg['width']}x{cfg['height']} {cfg['fps']}fps → {out_dir}",
        flush=True,
    )
    bpy.ops.render.render(animation=True)
    print("BLENDER_TOUR: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
