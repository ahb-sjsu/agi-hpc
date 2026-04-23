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

# Default model — Three.js's RobotExpressive. Stylized humanoid robot
# with a real face (eyes, mouth, brow morph targets for Angry / Sad /
# Happy / Surprised) and packaged idle/walk/dance clips. Chosen over
# Michelle (racial identification) and CesiumMan (no face) for an AI
# NPC: clearly artificial, unambiguously non-human, still expressive.
# Final ARTEMIS avatar will be an RPM/VRoid pick once S3-1 lands.
DEFAULT_MODEL_URL = (
    "https://threejs.org/examples/models/gltf/RobotExpressive/RobotExpressive.glb"
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
    "three/addons/": "https://unpkg.com/three@$THREE_VERSION/examples/jsm/"
  }
}
</script>
</head>
<body>
<script type="module">
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const W = $WIDTH, H = $HEIGHT, FPS = $FPS;

window.artemis = {
  ready: false,
  mouthOpen: 0,
  expression: "neutral",
  idle: true,
  setMouthOpen(v) { this.mouthOpen = Math.max(0, Math.min(1, +v || 0)); },
  setExpression(n) { this.expression = String(n || "neutral"); },
  setIdle(on) { this.idle = !!on; },
};

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
let clock = new THREE.Clock();

const loader = new GLTFLoader();
loader.load(
  "$MODEL_URL",
  (gltf) => {
    model = gltf.scene;
    // Center + scale conservatively so we frame the head/torso even
    // for a wide variety of model rigs.
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3()).y || 1;
    const scale = 1.7 / size;
    model.scale.setScalar(scale);
    box.setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.x -= center.x;
    model.position.z -= center.z;
    // Foot on the ground.
    const newBox = new THREE.Box3().setFromObject(model);
    model.position.y -= newBox.min.y;
    scene.add(model);
    if (gltf.animations && gltf.animations.length) {
      mixer = new THREE.AnimationMixer(model);
      const action = mixer.clipAction(gltf.animations[0]);
      action.timeScale = $ANIMATION_SPEED;
      action.play();
    }
    window.artemis.ready = true;
  },
  undefined,
  (err) => {
    console.error("model load failed:", err);
    // Fall back to a placeholder cube so the renderer still produces
    // frames the operator can see.
    const geo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const mat = new THREE.MeshStandardMaterial({ color: 0x50dcf0 });
    model = new THREE.Mesh(geo, mat);
    model.position.set(0, 1.5, 0);
    scene.add(model);
    window.artemis.ready = true;
    window.artemis.error = String(err);
  },
);

function tick() {
  const dt = clock.getDelta();
  if (mixer) mixer.update(dt);
  if (model && window.artemis.idle) {
    // Subtle breathing — chest scale + head tilt.
    const t = clock.elapsedTime;
    model.rotation.y = Math.sin(t * 0.25) * 0.08;
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
