import * as THREE from './vendor/three.module.min.js';

const viewport = document.querySelector('#viewport');
const styleSelect = document.querySelector('#style-select');
const statusElement = document.querySelector('#status');
const planInfo = document.querySelector('#plan-info');
const targetInfo = document.querySelector('#target-info');
const testResult = document.querySelector('#test-result');
const showAllTargets = document.querySelector('#show-all-targets');
const resetCamera = document.querySelector('#reset-camera');
const query = new URLSearchParams(location.search);

const state = {
  meta: null, session: '', motion: null, targets: null, playhead: 0,
  lastTime: performance.now(), move: [0, 0], facing: [0, 1], keys: new Set(),
  pending: false, seed: 10, style: '', rig: null, targetRigs: [],
  generatedPath: null, targetPath: null,
};

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x090b10, 0.045);
const camera = new THREE.PerspectiveCamera(42, 1, 0.02, 100);
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.append(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xcde4ff, 0x253226, 2.5));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
keyLight.position.set(3, 7, 4);
keyLight.castShadow = true;
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x67e8c3, 1.2);
rimLight.position.set(-4, 3, -3);
scene.add(rimLight);
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.MeshStandardMaterial({color: 0x0d121a, roughness: 0.96, metalness: 0.02}),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.012;
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(24, 48, 0x3a4b5d, 0x202b38);
grid.position.y = 0;
scene.add(grid);

const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 8, 1, false);
const sphereGeometry = new THREE.SphereGeometry(1, 14, 10);
const diamondGeometry = new THREE.OctahedronGeometry(1, 0);
const up = new THREE.Vector3(0, 1, 0);
const startPoint = new THREE.Vector3();
const endPoint = new THREE.Vector3();
const direction = new THREE.Vector3();
const midpoint = new THREE.Vector3();

function labelSprite(text, color, opacity) {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  context.font = '700 30px system-ui';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = color;
  context.fillText(text, 64, 32);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: texture, transparent: true, opacity, depthTest: false}));
  sprite.scale.set(0.32, 0.16, 1);
  sprite.renderOrder = 20;
  return sprite;
}

class SkeletonRig {
  constructor(joints, options) {
    this.bones = [];
    this.jointMeshes = [];
    this.segments = [];
    this.group = new THREE.Group();
    this.radius = options.radius;
    this.jointRadius = options.jointRadius;
    this.rootRadius = options.rootRadius;
    this.labelHeight = options.labelHeight ?? 0.34;
    this.boneMaterial = new THREE.MeshStandardMaterial({
      color: options.color, emissive: options.emissive, emissiveIntensity: options.emissiveIntensity,
      roughness: 0.42, transparent: options.opacity < 1, opacity: options.opacity,
      depthWrite: options.opacity >= 0.95,
    });
    this.jointMaterial = new THREE.MeshStandardMaterial({
      color: options.jointColor, emissive: options.emissive, emissiveIntensity: options.emissiveIntensity,
      roughness: 0.35, transparent: options.opacity < 1, opacity: options.opacity,
      depthWrite: options.opacity >= 0.95,
    });
    this.rootMaterial = new THREE.MeshStandardMaterial({
      color: options.rootColor, emissive: options.rootEmissive, emissiveIntensity: 1.4,
      transparent: options.opacity < 1, opacity: options.opacity, depthWrite: options.opacity >= 0.95,
    });
    for (let index = 0; index < joints.length; index++) {
      const joint = joints[index];
      const bone = new THREE.Bone();
      bone.name = joint.name;
      if (joint.parent < 0) {
        bone.position.set(0, 0, 0);
      } else {
        const parent = joints[joint.parent].position;
        bone.position.set(joint.position[0] - parent[0], joint.position[1] - parent[1], joint.position[2] - parent[2]);
        this.bones[joint.parent].add(bone);
      }
      this.bones.push(bone);
      const geometry = options.diamonds ? diamondGeometry : sphereGeometry;
      const marker = new THREE.Mesh(geometry, index === 0 ? this.rootMaterial : this.jointMaterial);
      marker.castShadow = !options.diamonds;
      marker.renderOrder = options.renderOrder;
      this.jointMeshes.push(marker);
      this.group.add(marker);
      if (joint.parent >= 0) {
        const segment = new THREE.Mesh(cylinderGeometry, this.boneMaterial);
        segment.castShadow = !options.diamonds;
        segment.renderOrder = options.renderOrder;
        this.segments.push({mesh: segment, child: index, parent: joint.parent});
        this.group.add(segment);
      }
    }
    scene.add(this.bones[0]);
    scene.add(this.group);
    this.label = options.label ? labelSprite(options.label, options.labelColor, options.opacity) : null;
    if (this.label) scene.add(this.label);
  }

  pose(roots, rotations, frame, frameCount) {
    const index = Math.min(frameCount - 1, Math.max(0, frame));
    this.bones[0].position.fromArray(roots, index * 3);
    for (let joint = 0; joint < this.bones.length; joint++) {
      this.bones[joint].quaternion.fromArray(rotations, (index * this.bones.length + joint) * 4);
    }
    this.updateGeometry();
  }

  updateGeometry() {
    this.bones[0].updateMatrixWorld(true);
    for (let index = 0; index < this.bones.length; index++) {
      this.bones[index].getWorldPosition(endPoint);
      const marker = this.jointMeshes[index];
      marker.position.copy(endPoint);
      const radius = index === 0 ? this.rootRadius : this.jointRadius;
      marker.scale.setScalar(radius);
    }
    for (const segment of this.segments) {
      this.bones[segment.parent].getWorldPosition(startPoint);
      this.bones[segment.child].getWorldPosition(endPoint);
      direction.subVectors(endPoint, startPoint);
      const length = direction.length();
      midpoint.addVectors(startPoint, endPoint).multiplyScalar(0.5);
      segment.mesh.position.copy(midpoint);
      segment.mesh.quaternion.setFromUnitVectors(up, direction.normalize());
      segment.mesh.scale.set(this.radius, length, this.radius);
    }
    if (this.label) {
      this.bones[0].getWorldPosition(endPoint);
      this.label.position.set(endPoint.x, endPoint.y + this.labelHeight, endPoint.z);
    }
  }

  setVisible(visible) {
    this.group.visible = visible;
    this.bones[0].visible = visible;
    if (this.label) this.label.visible = visible;
  }
}

function makeLine(color, dashed = false) {
  const material = dashed
    ? new THREE.LineDashedMaterial({color, dashSize: 0.09, gapSize: 0.055, transparent: true, opacity: 0.9})
    : new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.72});
  const line = new THREE.Line(new THREE.BufferGeometry(), material);
  line.renderOrder = 1;
  scene.add(line);
  return line;
}

function setGroundPath(line, roots, frames) {
  const points = [];
  for (let frame = 0; frame < frames; frame++) {
    points.push(new THREE.Vector3(roots[frame * 3], 0.018, roots[frame * 3 + 2]));
  }
  line.geometry.dispose();
  line.geometry = new THREE.BufferGeometry().setFromPoints(points);
  if (line.material.isLineDashedMaterial) line.computeLineDistances();
}

function makeSkeletons(joints) {
  state.rig = new SkeletonRig(joints, {
    color: 0x55efc4, jointColor: 0xd9fff3, rootColor: 0xffd166,
    emissive: 0x0c5b49, rootEmissive: 0x6a3b00, emissiveIntensity: 0.8,
    radius: 0.022, jointRadius: 0.034, rootRadius: 0.062,
    opacity: 1, diamonds: false, renderOrder: 5,
  });
  const opacities = [0.16, 0.25, 0.42, 0.9];
  for (let frame = 0; frame < 4; frame++) {
    state.targetRigs.push(new SkeletonRig(joints, {
      color: 0xff763b, jointColor: 0xffb06b, rootColor: 0xff4f8b,
      emissive: 0x7b1f00, rootEmissive: 0x790025, emissiveIntensity: 1,
      radius: frame === 3 ? 0.018 : 0.012, jointRadius: frame === 3 ? 0.032 : 0.023,
      rootRadius: frame === 3 ? 0.056 : 0.042, opacity: opacities[frame], diamonds: true,
      renderOrder: 8 + frame, label: `T${frame}`, labelColor: frame === 3 ? '#ffb477' : '#c97354',
      labelHeight: 0.34 + (3 - frame) * 0.11,
    }));
  }
  state.generatedPath = makeLine(0x43e8bf);
  state.targetPath = makeLine(0xff6b3d, true);
}

function updateTargetVisibility() {
  for (let frame = 0; frame < state.targetRigs.length; frame++) {
    const exists = Boolean(state.targets) && frame < state.targets.frames;
    state.targetRigs[frame].setVisible(exists && (showAllTargets.checked || frame === state.targets.frames - 1));
  }
  state.targetPath.visible = Boolean(state.targets) && showAllTargets.checked;
}

const cameraView = {yaw: 0.68, pitch: 0.24, distance: 4.8, dragging: false, x: 0, y: 0};
const focus = new THREE.Vector3();
const desiredCamera = new THREE.Vector3();
const visibleBounds = new THREE.Box3();
function resetCameraView() {
  cameraView.yaw = 0.68;
  cameraView.pitch = 0.24;
  cameraView.distance = 4.8;
}
resetCamera.addEventListener('click', resetCameraView);
renderer.domElement.addEventListener('pointerdown', event => {
  cameraView.dragging = true; cameraView.x = event.clientX; cameraView.y = event.clientY;
  renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointermove', event => {
  if (!cameraView.dragging) return;
  cameraView.yaw -= (event.clientX - cameraView.x) * 0.006;
  cameraView.pitch = THREE.MathUtils.clamp(cameraView.pitch + (event.clientY - cameraView.y) * 0.004, -0.05, 1.05);
  cameraView.x = event.clientX; cameraView.y = event.clientY;
});
renderer.domElement.addEventListener('pointerup', event => {
  cameraView.dragging = false; renderer.domElement.releasePointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointercancel', () => { cameraView.dragging = false; });
renderer.domElement.addEventListener('wheel', event => {
  cameraView.distance = THREE.MathUtils.clamp(cameraView.distance * Math.exp(event.deltaY * 0.001), 2.1, 8);
  event.preventDefault();
}, {passive: false});
renderer.domElement.addEventListener('dblclick', resetCameraView);

function resize() {
  const width = Math.max(1, viewport.clientWidth), height = Math.max(1, viewport.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
resize();

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? 'POST' : 'GET', headers: {'Content-Type': 'application/json'},
    body: body ? JSON.stringify(body) : undefined,
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `${response.status} ${response.statusText}`);
  return value;
}

function installStyles(styles) {
  for (const style of styles) {
    const option = document.createElement('option');
    option.value = style.name;
    option.textContent = `${style.name.replaceAll('_', ' ')} · ${style.speed.toFixed(1)} m/s`;
    styleSelect.append(option);
  }
  state.style = styles.some(item => item.name === 'walk') ? 'walk' : styles[0].name;
  styleSelect.value = state.style;
  styleSelect.addEventListener('change', () => { state.style = styleSelect.value; void requestPlan(); });
}

function useMotion(response) {
  state.session = response.session;
  state.style = response.style;
  styleSelect.value = response.style;
  state.motion = response.motion;
  state.targets = response.targets;
  state.playhead = 0;
  state.lastTime = performance.now();
  planInfo.textContent = `${response.motion.frames} frames · ${response.style.replaceAll('_', ' ')}`;
  targetInfo.textContent = `${response.targets.frames} placed constraints`;
  statusElement.textContent = 'Playing';
  document.documentElement.dataset.planSequence = String(Number(document.documentElement.dataset.planSequence || 0) + 1);
  setGroundPath(state.generatedPath, response.motion.roots, response.motion.frames);
  setGroundPath(state.targetPath, response.targets.roots, response.targets.frames);
  for (let frame = 0; frame < response.targets.frames; frame++) {
    state.targetRigs[frame].pose(response.targets.roots, response.targets.rotations, frame, response.targets.frames);
  }
  updateTargetVisibility();
}

async function requestPlan(advance = Math.floor(state.playhead)) {
  if (state.pending || !state.session) return null;
  state.pending = true;
  statusElement.textContent = 'Planning…';
  try {
    const response = await api('/api/plan', {
      session: state.session, style: state.style, move: state.move, facing: state.facing,
      seed: state.seed++, advance,
    });
    useMotion(response);
    return response;
  } finally {
    state.pending = false;
  }
}

function updateControl() {
  let x = 0, z = 0;
  if (state.keys.has('w')) z += 1;
  if (state.keys.has('s')) z -= 1;
  if (state.keys.has('a')) x -= 1;
  if (state.keys.has('d')) x += 1;
  const length = Math.hypot(x, z);
  if (length > 0) { x /= length; z /= length; state.facing = [x, z]; }
  state.move = [x, z];
  document.querySelectorAll('.pad button').forEach(button => button.classList.toggle('active', state.keys.has(button.dataset.key)));
}

let controlTimer = 0;
function schedulePlan() { clearTimeout(controlTimer); controlTimer = setTimeout(() => void requestPlan(), 70); }
addEventListener('keydown', event => {
  const key = event.key.toLowerCase();
  if (['w', 'a', 's', 'd'].includes(key)) {
    state.keys.add(key); updateControl(); schedulePlan(); event.preventDefault();
  }
  if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
    const angle = Math.atan2(state.facing[0], state.facing[1]) + (event.key === 'ArrowLeft' ? 0.25 : -0.25);
    state.facing = [Math.sin(angle), Math.cos(angle)]; schedulePlan(); event.preventDefault();
  }
});
addEventListener('keyup', event => {
  const key = event.key.toLowerCase();
  if (state.keys.delete(key)) { updateControl(); schedulePlan(); }
});
for (const button of document.querySelectorAll('.pad button')) {
  const down = event => { state.keys.add(button.dataset.key); updateControl(); schedulePlan(); event.preventDefault(); };
  const upHandler = event => { state.keys.delete(button.dataset.key); updateControl(); schedulePlan(); event.preventDefault(); };
  button.addEventListener('pointerdown', down);
  button.addEventListener('pointerup', upHandler);
  button.addEventListener('pointercancel', upHandler);
  button.addEventListener('pointerleave', upHandler);
}
showAllTargets.addEventListener('change', updateTargetVisibility);

function updateCamera() {
  visibleBounds.makeEmpty();
  for (const marker of state.rig.jointMeshes) visibleBounds.expandByPoint(marker.position);
  for (const rig of state.targetRigs) {
    if (rig.group.visible) for (const marker of rig.jointMeshes) visibleBounds.expandByPoint(marker.position);
  }
  visibleBounds.getCenter(focus);
  const horizontal = Math.cos(cameraView.pitch) * cameraView.distance;
  desiredCamera.set(
    focus.x + Math.sin(cameraView.yaw) * horizontal,
    focus.y + Math.sin(cameraView.pitch) * cameraView.distance,
    focus.z + Math.cos(cameraView.yaw) * horizontal,
  );
  if (camera.userData.positioned) camera.position.lerp(desiredCamera, 0.12);
  else { camera.position.copy(desiredCamera); camera.userData.positioned = true; }
  camera.lookAt(focus);
}

function renderMotion(frame) {
  if (!state.motion) return;
  const index = Math.min(state.motion.frames - 1, Math.max(0, frame));
  state.rig.pose(state.motion.roots, state.motion.rotations, index, state.motion.frames);
  updateCamera();
}

function animate(now) {
  requestAnimationFrame(animate);
  const delta = Math.min(0.1, (now - state.lastTime) / 1000);
  state.lastTime = now;
  if (state.motion) {
    state.playhead += delta * state.meta.fps;
    if (state.playhead >= state.motion.frames - 5 && !state.pending) void requestPlan(Math.floor(state.playhead));
    renderMotion(Math.floor(state.playhead));
  }
  renderer.render(scene, camera);
}

async function selfTest() {
  const alternate = state.meta.styles.find(item => item.name === 'walk_zombie') || state.meta.styles.find(item => item.name !== 'walk');
  if (!alternate) throw new Error('no alternate upstream style available');
  state.style = alternate.name;
  styleSelect.value = alternate.name;
  state.move = [1, 0];
  state.facing = [1, 0];
  const response = await requestPlan(3);
  if (!response || response.style !== alternate.name) throw new Error('style change was not applied');
  if (response.motion.joints !== 34 || response.motion.frames < 24 || response.motion.rotations.length !== response.motion.frames * 34 * 4) {
    throw new Error('invalid skeletal animation response');
  }
  if (response.targets.frames !== 4 || response.targets.joints !== 34 || response.targets.roots.length !== 12 || response.targets.rotations.length !== 4 * 34 * 4) {
    throw new Error('invalid placed target-keyframe response');
  }
  renderMotion(2);
  renderer.render(scene, camera);
  const visibleTargets = state.targetRigs.filter(rig => rig.group.visible).length;
  if (!renderer.domElement.width || state.rig.bones.length !== 34 || visibleTargets !== 4) {
    throw new Error('animated and target skeletons were not rendered');
  }
  document.documentElement.dataset.animatedJoints = String(state.rig.bones.length);
  document.documentElement.dataset.targetFrames = String(visibleTargets);
  document.documentElement.dataset.testStatus = 'passed';
  testResult.textContent = `Headless check passed: solid model + ${visibleTargets} target ghosts, ${alternate.name}, right turn`;
}

async function start() {
  try {
    state.meta = await api('/api/meta');
    installStyles(state.meta.styles);
    makeSkeletons(state.meta.joints);
    const initial = await api('/api/session', {style: state.style});
    useMotion(initial);
    renderMotion(0);
    requestAnimationFrame(animate);
    if (query.get('test') === '1') await selfTest();
    else document.documentElement.dataset.testStatus = 'ready';
  } catch (error) {
    console.error(error);
    statusElement.textContent = 'Error';
    testResult.textContent = error.message;
    document.documentElement.dataset.testStatus = 'failed';
  }
}

await start();
