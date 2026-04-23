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
import math
import os
import sys

import bpy
from mathutils import Quaternion, Vector
from mathutils import noise as bnoise

# ─────────────────────────────────────────────────────────────────
# VRM shape-key + bone names (VRoid standard, Fcl_* prefix).
# ─────────────────────────────────────────────────────────────────

FACE_MESH = "Face"
# Default prefix — VRMs from the pixiv three-vrm sample use
# "Face_Blendshape.Fcl_"; plain VRoid Hub exports just use "Fcl_".
# Resolved at runtime in _resolve_key_prefix().
KEY_PREFIX = "Fcl_"

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

# Bone rotations are applied to these VRM humanoid bones via the
# world-space helper below — bone-roll independent, so the same POSES
# dict works for any VRoid-exported VRM.
BONE = {
    "head": "J_Bip_C_Head",
    "neck": "J_Bip_C_Neck",
    "spine": "J_Bip_C_Spine",
    "hips": "J_Bip_C_Hips",
    "leftUpperArm": "J_Bip_L_UpperArm",
    "rightUpperArm": "J_Bip_R_UpperArm",
    "leftLowerArm": "J_Bip_L_LowerArm",
    "rightLowerArm": "J_Bip_R_LowerArm",
}

# Split arm bones (pose-controlled) from idle bones (continuous micro-
# motion). Head/neck/spine/hips never get pose keyframes — their motion
# comes from _build_idle_micromotion so she's never dead-still.
POSE_BONES = ("leftUpperArm", "rightUpperArm", "leftLowerArm", "rightLowerArm")
IDLE_BONES = ("head", "neck", "spine", "hips")

# Pose = per-arm-bone (world_axis, angle_rad). In Blender's right-hand
# rule, rotating around world +Y by +angle sends world +X → -Z, so a
# LEFT upper arm (rest along +X) goes DOWN. The right arm rests along
# -X and takes the opposite-sign angle around +Y.
POSES = {
    # Relaxed arms-at-sides.
    "mountain": {
        "leftUpperArm": ((0, 1, 0), 1.30),
        "rightUpperArm": ((0, 1, 0), -1.30),
    },
    # Both arms overhead.
    "reach": {
        "leftUpperArm": ((0, 1, 0), -1.45),
        "rightUpperArm": ((0, 1, 0), 1.45),
    },
    # Front arm forward, back arm back — warrior-II silhouette.
    "warrior": {
        "leftUpperArm": ((0, 0, 1), 0.35),
        "rightUpperArm": ((0, 0, 1), 0.35),
    },
    # Arms drawn in toward chest.
    "prayer": {
        "leftUpperArm": ((0, 1, 0), 0.80),
        "rightUpperArm": ((0, 1, 0), -0.80),
    },
    # One arm raised (checking-watch style), other relaxed.
    "ping": {
        "leftUpperArm": ((0, 1, 0), -1.10),
        "rightUpperArm": ((0, 1, 0), -0.20),
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


def _resolve_key_prefix(face) -> str:
    """Pick the shape-key prefix that actually matches this VRM.

    pixiv-sample VRMs prefix every Fcl_ key with ``Face_Blendshape.``;
    plain VRoid Hub exports don't. We probe by looking up ALL_Joy in
    both forms and return whichever resolves. Falls back to ``Fcl_``
    so the rest of the pipeline silently-skips rather than crashes on
    an unfamiliar rig.
    """
    sk = face.data.shape_keys
    if sk is None:
        return "Fcl_"
    for candidate in ("Fcl_", "Face_Blendshape.Fcl_"):
        if (candidate + "ALL_Joy") in sk.key_blocks:
            return candidate
    return "Fcl_"


def _keyframe_shape(face, key: str, value: float, frame: int) -> None:
    """Set + keyframe a single shape key by its Fcl_ suffix."""
    sk = face.data.shape_keys.key_blocks.get(KEY_PREFIX + key)
    if sk is None:
        return  # missing on this VRM, skip
    sk.value = float(value)
    sk.keyframe_insert(data_path="value", frame=int(frame))


def _world_rotate_keyframe(arm, bone_name: str, axis, angle: float, frame: int) -> None:
    """Keyframe a pose bone with a world-space rotation.

    Independent of bone-roll conventions — converts the world-space
    axis-angle to the equivalent local-basis quaternion via the bone's
    rest rotation. Safe to use on any VRoid-exported VRM.
    """
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        return
    if angle == 0.0:
        local_q = Quaternion()
    else:
        world_q = Quaternion(axis, angle)
        # Bone rest in world space = armature world rot ∘ bone rest-local rot.
        arm_rot = arm.matrix_world.to_quaternion()
        bone_rest = pb.bone.matrix_local.to_quaternion()
        rest_world = arm_rot @ bone_rest
        local_q = rest_world.inverted() @ world_q @ rest_world
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = local_q
    pb.keyframe_insert(data_path="rotation_quaternion", frame=int(frame))


def _apply_pose_keyframe(arm, pose_name: str, frame: int) -> None:
    """Keyframe arm bones for the target pose; zero bones not in it
    so deltas from prior poses don't accumulate. Does NOT touch
    head/neck/spine — those are driven by idle micro-motion."""
    target = POSES.get(pose_name, {})
    for alias in POSE_BONES:
        bone_name = BONE[alias]
        spec = target.get(alias)
        if spec is None:
            _world_rotate_keyframe(arm, bone_name, (0, 1, 0), 0.0, frame)
        else:
            axis, angle = spec
            _world_rotate_keyframe(arm, bone_name, axis, angle, frame)


def _apply_expression_keyframe(face, name: str, frame: int) -> None:
    """Binary emotion: set the named one to 1.0, others to 0.0."""
    for k, suffix in EMOTION_KEYS.items():
        if suffix is None:
            continue
        val = 1.0 if k == name else 0.0
        _keyframe_shape(face, suffix, val, frame)


def _build_idle_micromotion(arm, total_frames: int, fps: int) -> None:
    """Coordinated Perlin-noise sway on head+neck+spine+hips.

    Each bone samples three decorrelated Perlin channels (pitch, yaw,
    roll) at its own time-scale, so motion reads as organic rather
    than metronomic. Spine gets a slow breathing rhythm on top; hips
    get a subtle weight-shift sway. Runs in LOCAL Euler space — axis
    direction matters less than continuity and smallness.
    """
    step = max(1, fps // 3)  # ~10 Hz sampling
    # (pitch_amp, yaw_amp, roll_amp, time_scale_s, phase_offset)
    # Amplitudes ~3x the first pass — earlier values were too subtle
    # to read visually at the current camera framing.
    specs = {
        BONE["head"]: (0.15, 0.21, 0.09, 5.0, 0.0),
        BONE["neck"]: (0.075, 0.105, 0.045, 4.0, 11.3),
        BONE["spine"]: (0.06, 0.054, 0.075, 3.5, 23.7),
        BONE["hips"]: (0.03, 0.045, 0.066, 6.0, 41.1),
    }
    for bone_name in specs:
        pb = arm.pose.bones.get(bone_name)
        if pb is not None:
            pb.rotation_mode = "XYZ"

    for f in range(1, total_frames + 1, step):
        t = f / fps
        for bone_name, (p_amp, y_amp, r_amp, scale, phase) in specs.items():
            pb = arm.pose.bones.get(bone_name)
            if pb is None:
                continue
            u = (t + phase) / scale
            # Sample Perlin on three decorrelated lines in noise space.
            pitch = p_amp * bnoise.noise(Vector((u, 0.0, 0.0)))
            yaw = y_amp * bnoise.noise(Vector((0.0, u, 0.0)))
            roll = r_amp * bnoise.noise(Vector((0.0, 0.0, u)))
            if bone_name == BONE["spine"]:
                # Gentle breathing — ~0.13 Hz, modulates pitch.
                pitch += 0.012 * math.sin(0.80 * t)
            pb.rotation_euler = (pitch, yaw, roll)
            pb.keyframe_insert(data_path="rotation_euler", frame=f)


# ─────────────────────────────────────────────────────────────────
# Main build
# ─────────────────────────────────────────────────────────────────


def _build_animation(cfg: dict) -> None:
    global KEY_PREFIX
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    face = _face()
    KEY_PREFIX = _resolve_key_prefix(face)
    print(f"BLENDER_TOUR: shape-key prefix resolved to {KEY_PREFIX!r}", flush=True)
    fps = cfg["fps"]

    # Zero EVERY Fcl_* shape key on the face at frame 1 so there's no
    # import-time residual or auxiliary mouth-opening key stacking.
    sk_blocks = face.data.shape_keys.key_blocks if face.data.shape_keys else []
    for kb in sk_blocks:
        if kb.name.startswith(KEY_PREFIX) and kb.name != KEY_PREFIX + "ALL_Neutral":
            kb.value = 0.0
            kb.keyframe_insert(data_path="value", frame=1)

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
    # During the explicit emotion-demo window ALL_Surprised opens the
    # mouth on its own; suppress MTH_A there so the two don't stack
    # into a yawning look.
    aa_suffix = VISEME_KEYS["aa"]
    envelope = cfg.get("envelope", [])
    emotion_windows = [
        ev["t"]
        for ev in cfg.get("timeline", [])
        if ev["kind"] == "expression" and ev["arg"] not in (None, "neutral")
    ]
    for i, level in enumerate(envelope):
        t_s = i / fps
        # Zero MTH_A for ~0.7s around every non-neutral expression beat.
        if any(abs(t_s - et) < 0.7 for et in emotion_windows):
            level = 0.0
        _keyframe_shape(face, aa_suffix, float(level), i + 1)

    # Randomized blinks on a coarse ~3 s cadence.
    import random as _r

    _r.seed(cfg.get("blink_seed", 42))
    t_blink = 1.5
    total_s = len(envelope) / fps if envelope else cfg.get("total_s", 30)
    total_frames = max(1, int(round(total_s * fps)))
    while t_blink < total_s:
        f = int(round(t_blink * fps))
        _keyframe_shape(face, BLINK_KEY, 0.0, max(1, f - 1))
        _keyframe_shape(face, BLINK_KEY, 1.0, f)
        _keyframe_shape(face, BLINK_KEY, 0.0, f + int(fps * 0.15))
        t_blink += _r.uniform(3.0, 5.0)

    # Continuous head + neck + spine micro-motion so she's never
    # dead-still. Keyframed at ~10 Hz for the whole clip.
    _build_idle_micromotion(arm, total_frames, fps)

    # Force LINEAR interpolation on shape-key f-curves. Default Bezier
    # auto-handles overshoot between rapidly-oscillating envelope
    # samples and progressively pushes MTH_A beyond its keyframed
    # ceiling — reads as "mouth keeps getting wider until yawning."
    _force_linear_shape_keys(face)


def _force_linear_shape_keys(face) -> None:
    sk_data = face.data.shape_keys
    if sk_data is None or sk_data.animation_data is None:
        return
    action = sk_data.animation_data.action
    if action is None:
        return
    # Dump every MTH_A f-curve keyframe to stderr so we can verify the
    # animation is actually bounded at NATURAL_MAX. Debug instrumentation.
    for fc in action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
        # Set extrapolation to CONSTANT at both ends — default is
        # CONSTANT anyway, but be explicit so no f-curve can coast
        # above its last keyframe value.
        fc.extrapolation = "CONSTANT"
        if "MTH_A" in fc.data_path and "ALL" not in fc.data_path:
            vals = [kp.co[1] for kp in fc.keyframe_points]
            if vals:
                print(
                    "BLENDER_TOUR: MTH_A fcurve: "
                    f"n={len(vals)} min={min(vals):.3f} max={max(vals):.3f} "
                    f"first3={vals[:3]} last3={vals[-3:]}",
                    flush=True,
                )


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
