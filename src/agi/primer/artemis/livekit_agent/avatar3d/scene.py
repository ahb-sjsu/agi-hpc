# AGI-HPC Project - High-Performance Computing Architecture for AGI
# Copyright (c) 2025-2026 Andrew H. Bond
# Contact: agi.hpc@gmail.com
#
# Licensed under the AGI-HPC Responsible AI License v1.0.

"""HTML + Three.js scene template for the ARTEMIS 3D avatar.

The scene is a self-contained HTML page — no build step, no
filesystem dependencies. The renderer drops it into a temp file,
points headless Chromium at it, and captures the canvas. Three.js
and the GLTFLoader are pulled from a versioned CDN; the avatar GLB
is loaded from whatever URL we pass in.

The scene exposes a small JS API on ``window.artemis``:

  - ``window.artemis.ready`` — true once the model is loaded
  - ``window.artemis.setMouthOpen(level)`` — 0..1 mouth amplitude
  - ``window.artemis.setExpression(name)`` — "neutral" / "listening"
  - ``window.artemis.setIdle(on)`` — toggle idle micro-motion

Phase A only renders the scene + spins the model. Phases B/D wire
the audio amplitude and expression hooks up.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template

# Pinned CDN versions so the scene is reproducible even if upstream
# ships breaking changes. Bumped here, not silently picked up.
THREE_VERSION = "0.165.0"
# three-vrm version compatible with three 0.165. three-vrm 3.x uses
# the current three API. Bump together.
THREE_VRM_VERSION = "3.2.0"

# Default model — pixiv's public VRM sample (anime VTuber style,
# hosted via GitHub Pages). Replaces RobotExpressive as the placeholder
# after the user asked for anime-style instead of realistic-humanoid.
# Final ARTEMIS avatar URL will come from a VRoid Hub pick; swap via
# ``model_url`` or the ``ARTEMIS_AVATAR_MODEL_URL`` env var.
DEFAULT_MODEL_URL = (
    "https://pixiv.github.io/three-vrm/packages/three-vrm/examples/models/"
    "VRM1_Constraint_Twist_Sample.vrm"
)


@dataclass(frozen=True)
class SceneConfig:
    """Render-time scene configuration.

    Width/height match the avatar's video track at publish time.
    Background is a solid hex color (no alpha channel — LiveKit
    Opus video doesn't carry transparency).
    """

    model_url: str = DEFAULT_MODEL_URL
    width: int = 1280
    height: int = 720
    background: str = "#050914"  # matches the table UI bg-deep
    fps: int = 30
    # Animation playback scaling. 1.0 = native speed from the GLB;
    # 0.3 = the dance plays at a third speed (easier to see what the
    # rig is doing during smoke tests). Independent of capture fps.
    animation_speed: float = 0.3

    def to_template_vars(self) -> dict[str, str]:
        return {
            "MODEL_URL": self.model_url,
            "WIDTH": str(self.width),
            "HEIGHT": str(self.height),
            "BACKGROUND": self.background,
            "FPS": str(self.fps),
            "ANIMATION_SPEED": str(self.animation_speed),
            "THREE_VERSION": THREE_VERSION,
            "THREE_VRM_VERSION": THREE_VRM_VERSION,
        }


# Three.js lives at unpkg/jsdelivr; the importmap pins the version so
# Chromium loads exactly what we tested against. The animation loop
# is intentionally minimal in Phase A — it spins the model so we can
# verify capture works end-to-end. Phase D layers on idle motion +
# lip-sync via window.artemis.setMouthOpen.
_HTML_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>ARTEMIS Avatar</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; }
  body { background: $BACKGROUND; }
  canvas { display: block; }
</style>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@$THREE_VERSION/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@$THREE_VERSION/examples/jsm/",
    "@pixiv/three-vrm": "https://unpkg.com/@pixiv/three-vrm@$THREE_VRM_VERSION/lib/three-vrm.module.js"
  }
}
</script>
</head>
<body>
<script type="module">
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const W = $WIDTH, H = $HEIGHT, FPS = $FPS;

window.artemis = {
  ready: false,
  // Target values set by the caller. The rig is driven from these
  // (possibly via a smoother) inside tick().
  mouthOpen: 0,
  _mouthSmoothed: 0,
  visemes: { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 },
  expression: "neutral",
  emotionBlend: { happy: 0, sad: 0, angry: 0, surprised: 0, relaxed: 0 },
  lookAt: { x: 0, y: 0 },  // -1..+1 each axis
  idle: true,
  pose: null,  // null = auto-cycle through POSES; int = hold that index
  _nextBlinkAt: 0,
  setMouthOpen(v) { this.mouthOpen = Math.max(0, Math.min(1, +v || 0)); },
  // Direct per-viseme control (mouthOpen just drives 'aa').
  setViseme(name, v) {
    if (name in this.visemes) {
      this.visemes[name] = Math.max(0, Math.min(1, +v || 0));
    }
  },
  setExpression(n) { this.expression = String(n || "neutral"); },
  setEmotion(name, v) {
    if (name in this.emotionBlend) {
      this.emotionBlend[name] = Math.max(0, Math.min(1, +v || 0));
    }
  },
  setLookAt(x, y) {
    this.lookAt.x = Math.max(-1, Math.min(1, +x || 0));
    this.lookAt.y = Math.max(-1, Math.min(1, +y || 0));
  },
  setIdle(on) { this.idle = !!on; },
  setPose(i) {
    this.pose = i == null ? null : Math.max(0, Math.min(POSES.length - 1, i | 0));
  },
};

// ─────────────────────────────────────────────────────────────────
// Yoga-ish pose library. Each entry is a dict of VRM humanoid bone
// name → [xRad, yRad, zRad] Euler rotation. Bones are always local
// to the rig so poses retarget cleanly onto any VRM avatar.
// Keep them simple + readable — the Edward-from-Bebop vibe is odd
// pose, not biomechanical accuracy.
// ─────────────────────────────────────────────────────────────────
const P = Math.PI;
const POSES = [
  // 0 — Mountain / at attention
  { name: "mountain", bones: {
    leftUpperArm:  [0, 0,  0.08],  leftLowerArm:  [0, 0, 0],
    rightUpperArm: [0, 0, -0.08],  rightLowerArm: [0, 0, 0],
    spine: [0, 0, 0], neck: [0, 0, 0], head: [0, 0, 0],
  }},
  // 1 — Arms overhead
  { name: "reach", bones: {
    leftUpperArm:  [0, 0,  2.8],   leftLowerArm:  [0, 0, 0],
    rightUpperArm: [0, 0, -2.8],   rightLowerArm: [0, 0, 0],
    spine: [-0.08, 0, 0], neck: [-0.12, 0, 0], head: [-0.1, 0, 0],
  }},
  // 2 — T-pose / Warrior II-ish
  { name: "warrior", bones: {
    leftUpperArm:  [0, 0,  P * 0.5],  leftLowerArm:  [0, 0, 0],
    rightUpperArm: [0, 0, -P * 0.5],  rightLowerArm: [0, 0, 0],
    spine: [0, 0.12, 0], neck: [0, -0.12, 0], head: [0, 0, 0],
  }},
  // 3 — Prayer / thinking
  { name: "prayer", bones: {
    leftUpperArm:  [0,  0.1,  1.45], leftLowerArm:  [0,  1.1, 0],
    rightUpperArm: [0, -0.1, -1.45], rightLowerArm: [0, -1.1, 0],
    spine: [0.04, 0, 0], neck: [0.12, 0, 0], head: [0.15, 0, 0],
  }},
  // 4 — One-arm-up lean (Edward ping)
  { name: "ping", bones: {
    leftUpperArm:  [0, 0, 2.6],
    rightUpperArm: [0, 0, -0.4],
    leftLowerArm: [0, 0, 0], rightLowerArm: [0, 0, 0],
    spine: [0, -0.1, 0.08], neck: [0, 0, -0.08], head: [0, 0, -0.12],
  }},
];

const POSE_HOLD_S = 10.0;    // how long we linger on a pose
const POSE_BLEND_S = 3.0;    // SLERP duration between poses (eased)
const VRM_MOUTH_VISEME = "aa";
// Smoothing:
// - Pose blend factor runs through smoothstep (cubic) so velocity
//   starts+ends at zero, killing the jerk at blend-window edges.
// - Mouth amplitude runs through a one-pole IIR low-pass: a scalar
//   Kalman-ish EMA. Alpha 0.22 ≈ 5-frame window at 30 fps —
//   enough to smooth per-frame RMS jitter without lagging the
//   audio visibly.
const MOUTH_SMOOTH_ALPHA = 0.22;
function _smoothstep(t) { return t * t * (3 - 2 * t); }

// Module-scoped state used by tick helpers. Declared here so later
// function references don't hit the let/const temporal-dead-zone.
let _lookTarget = null;

const scene = new THREE.Scene();
scene.background = new THREE.Color("$BACKGROUND");

const camera = new THREE.PerspectiveCamera(35, W / H, 0.1, 100);
camera.position.set(0, 1.55, 2.6);
camera.lookAt(0, 1.45, 0);

const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(2, 4, 3);
scene.add(key);
const fill = new THREE.DirectionalLight(0x80c0ff, 0.4);
fill.position.set(-3, 2, 2);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xffd0a0, 0.6);
rim.position.set(0, 3, -3);
scene.add(rim);
scene.add(new THREE.AmbientLight(0xffffff, 0.25));

const renderer = new THREE.WebGLRenderer({
  antialias: true, preserveDrawingBuffer: true,
});
renderer.setSize(W, H);
renderer.setPixelRatio(1);
document.body.appendChild(renderer.domElement);

let model = null;
let mixer = null;
let vrm = null;  // set when a VRM file is loaded (spring bones + expressions)
let clock = new THREE.Clock();

const loader = new GLTFLoader();
// Handle .vrm files through the same loader — VRMLoaderPlugin reads
// the VRM extensions from the GLB container and attaches a VRM
// instance to gltf.userData.vrm. Plain GLB files without the VRM
// extensions still load normally (userData.vrm is undefined).
loader.register((parser) => new VRMLoaderPlugin(parser));

loader.load(
  "$MODEL_URL",
  (gltf) => {
    vrm = gltf.userData && gltf.userData.vrm ? gltf.userData.vrm : null;
    model = vrm ? vrm.scene : gltf.scene;

    // VRM models come pre-posed facing -Z; rotate to face the camera.
    if (vrm) {
      VRMUtils.rotateVRM0(vrm);
    }

    // Center + scale so any rig frames head-to-torso consistently.
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3()).y || 1;
    const scale = 1.7 / size;
    model.scale.setScalar(scale);
    box.setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.x -= center.x;
    model.position.z -= center.z;
    const newBox = new THREE.Box3().setFromObject(model);
    model.position.y -= newBox.min.y;
    scene.add(model);

    // Baked GLB clips only — VRM uses humanoid bone animation via
    // expressions / look-at / manual pose instead of clips, so we
    // skip mixer on VRM models.
    if (!vrm && gltf.animations && gltf.animations.length) {
      mixer = new THREE.AnimationMixer(model);
      const action = mixer.clipAction(gltf.animations[0]);
      action.timeScale = $ANIMATION_SPEED;
      action.play();
    }

    // Clark Kent disguise — attach simple glasses to the head bone
    // so this isn't obviously the pixiv test character. Two lens
    // rings + a bridge, dark frame + semi-transparent lenses.
    if (vrm) {
      const head = vrm.humanoid.getNormalizedBoneNode("head");
      if (head) {
        const glasses = _buildGlasses();
        // Scale/offset are in meters in the head's local frame —
        // VRM spec canonical head height ~0.23m tall, eyes ~0.05m
        // out from the head origin. Tweak if they float off-face.
        glasses.position.set(0, 0.08, 0.11);
        glasses.scale.setScalar(0.09);
        head.add(glasses);
      }
    }
    window.artemis.ready = true;
  },
  undefined,
  (err) => {
    console.error("model load failed:", err);
    const geo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const mat = new THREE.MeshStandardMaterial({ color: 0x50dcf0 });
    model = new THREE.Mesh(geo, mat);
    model.position.set(0, 1.5, 0);
    scene.add(model);
    window.artemis.ready = true;
    window.artemis.error = String(err);
  },
);

// Cached Euler → Quaternion scratch so tick() allocates nothing.
const _eulerScratch = new THREE.Euler();
const _quatA = new THREE.Quaternion();
const _quatB = new THREE.Quaternion();
const _quatOut = new THREE.Quaternion();

function _applyPoseBlend(tick_s) {
  if (!vrm || !vrm.humanoid) return;
  const a = window.artemis;
  let curIdx, prevIdx, blend;
  if (a.pose != null) {
    // Caller pinned a pose — blend from whatever we were on to it
    // over POSE_BLEND_S from the moment setPose was called.
    if (a._pinnedAt == null || a._pinnedPose !== a.pose) {
      a._pinnedAt = tick_s;
      a._pinnedFrom = a._lastIdx ?? 0;
      a._pinnedPose = a.pose;
    }
    const raw = Math.min(1.0, (tick_s - a._pinnedAt) / POSE_BLEND_S);
    blend = _smoothstep(raw);
    prevIdx = a._pinnedFrom;
    curIdx = a.pose;
    a._lastIdx = a.pose;
  } else {
    a._pinnedAt = null;
    const cycle = POSE_HOLD_S + POSE_BLEND_S;
    const phase = tick_s % cycle;
    const slot = Math.floor(tick_s / cycle);
    const inBlend = phase < POSE_BLEND_S;
    const rawBlend = inBlend ? phase / POSE_BLEND_S : 1.0;
    blend = _smoothstep(rawBlend);
    curIdx = slot % POSES.length;
    prevIdx = (slot + POSES.length - 1) % POSES.length;
    a._lastIdx = curIdx;
  }

  const prev = POSES[prevIdx].bones;
  const cur = POSES[curIdx].bones;

  // Walk every bone mentioned in either pose and SLERP it.
  const allBones = new Set([...Object.keys(prev), ...Object.keys(cur)]);
  for (const boneName of allBones) {
    const node = vrm.humanoid.getNormalizedBoneNode(boneName);
    if (!node) continue;
    const fromE = prev[boneName] || [0, 0, 0];
    const toE = cur[boneName] || [0, 0, 0];
    _eulerScratch.set(fromE[0], fromE[1], fromE[2], "XYZ");
    _quatA.setFromEuler(_eulerScratch);
    _eulerScratch.set(toE[0], toE[1], toE[2], "XYZ");
    _quatB.setFromEuler(_eulerScratch);
    _quatOut.copy(_quatA).slerp(_quatB, blend);
    node.quaternion.copy(_quatOut);
  }
}

function _applyMouthAndFace(tick_s) {
  if (!vrm || !vrm.expressionManager) return;
  const em = vrm.expressionManager;
  const a = window.artemis;

  // Mouth: IIR low-pass on 'aa' (driven by mouthOpen) plus any
  // direct per-viseme overrides callers set via setViseme().
  a._mouthSmoothed =
    MOUTH_SMOOTH_ALPHA * a.mouthOpen + (1 - MOUTH_SMOOTH_ALPHA) * a._mouthSmoothed;
  em.setValue("aa", Math.max(a._mouthSmoothed, a.visemes.aa));
  em.setValue("ih", a.visemes.ih);
  em.setValue("ou", a.visemes.ou);
  em.setValue("ee", a.visemes.ee);
  em.setValue("oh", a.visemes.oh);

  // Emotion: explicit setEmotion() values win, else setExpression(n)
  // binary-sets n to 1.0 and others to 0.
  const emotions = ["happy", "sad", "angry", "surprised", "relaxed"];
  for (const n of emotions) {
    const blended = a.emotionBlend[n];
    const picked = a.expression === n ? 1.0 : 0.0;
    em.setValue(n, Math.max(blended, picked));
  }

  // Blink — every ~3–5 s, 150 ms closed.
  if (tick_s > a._nextBlinkAt) {
    em.setValue("blink", 1.0);
    if (tick_s > a._nextBlinkAt + 0.15) {
      em.setValue("blink", 0.0);
      a._nextBlinkAt = tick_s + 3 + Math.random() * 2;
    }
  }
  em.update();

  // Look-at: map normalized (-1..1) gaze onto a distant target.
  if (vrm.lookAt) {
    if (!_lookTarget) {
      _lookTarget = new THREE.Object3D();
      scene.add(_lookTarget);
      vrm.lookAt.target = _lookTarget;
    }
    _lookTarget.position.set(a.lookAt.x * 2, 1.5 + a.lookAt.y * 1.5, 2.0);
  }
}

function _buildGlasses() {
  const group = new THREE.Group();
  const frameMat = new THREE.MeshStandardMaterial({
    color: 0x1a1a22, metalness: 0.3, roughness: 0.4,
  });
  const lensMat = new THREE.MeshStandardMaterial({
    color: 0x0e2633, metalness: 0.2, roughness: 0.2,
    transparent: true, opacity: 0.35,
  });
  const lensRing = new THREE.TorusGeometry(0.45, 0.06, 12, 32);
  const lensDisc = new THREE.CircleGeometry(0.42, 24);
  const bridge = new THREE.CylinderGeometry(0.04, 0.04, 0.22, 8);
  const temple = new THREE.CylinderGeometry(0.04, 0.04, 1.1, 8);
  // Left lens
  const lL = new THREE.Mesh(lensRing, frameMat);
  lL.position.set(-0.55, 0, 0);
  group.add(lL);
  const discL = new THREE.Mesh(lensDisc, lensMat);
  discL.position.set(-0.55, 0, 0);
  group.add(discL);
  // Right lens
  const lR = new THREE.Mesh(lensRing, frameMat);
  lR.position.set(0.55, 0, 0);
  group.add(lR);
  const discR = new THREE.Mesh(lensDisc, lensMat);
  discR.position.set(0.55, 0, 0);
  group.add(discR);
  // Bridge
  const br = new THREE.Mesh(bridge, frameMat);
  br.rotation.z = Math.PI / 2;
  br.position.set(0, 0, 0);
  group.add(br);
  // Temples (earpieces)
  const tL = new THREE.Mesh(temple, frameMat);
  tL.rotation.x = Math.PI / 2;
  tL.position.set(-1.05, 0, -0.55);
  group.add(tL);
  const tR = new THREE.Mesh(temple, frameMat);
  tR.rotation.x = Math.PI / 2;
  tR.position.set(1.05, 0, -0.55);
  group.add(tR);
  return group;
}

function tick() {
  const dt = clock.getDelta();
  const t = clock.elapsedTime;
  if (mixer) mixer.update(dt);
  // Drive pose blending BEFORE vrm.update so spring bones follow
  // the posed body rather than the last frame's.
  _applyPoseBlend(t);
  _applyMouthAndFace(t);
  if (vrm) vrm.update(dt);
  if (model && window.artemis.idle) {
    // Tiny body sway so the avatar doesn't look frozen between
    // pose transitions.
    model.rotation.y = Math.sin(t * 0.15) * 0.04;
  }
  renderer.render(scene, camera);
  setTimeout(tick, 1000 / FPS);
}
tick();
</script>
</body>
</html>
""")


def build_scene_html(config: SceneConfig | None = None) -> str:
    """Render the scene's HTML for ``config``.

    Pure function — no I/O. Tests inspect the output structure.
    """
    cfg = config or SceneConfig()
    return _HTML_TEMPLATE.substitute(cfg.to_template_vars())
