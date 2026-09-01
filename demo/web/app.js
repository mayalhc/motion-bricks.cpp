import * as THREE from './vendor/three.module.min.js';

const viewport = document.querySelector('#viewport');
const styleSelect = document.querySelector('#style-select');
const statusElement = document.querySelector('#status');
const planInfo = document.querySelector('#plan-info');
const testResult = document.querySelector('#test-result');
const query = new URLSearchParams(location.search);

const state = {
  meta: null, session: '', motion: null, playhead: 0, lastTime: performance.now(),
  move: [0, 0], facing: [0, 1], keys: new Set(), pending: false, seed: 10,
  style: '', bones: [], helper: null,
};

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x090b10, 0.075);
const camera = new THREE.PerspectiveCamera(48, 1, 0.02, 100);
camera.position.set(3.5, 2.2, 4.8);
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.append(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xbad8ff, 0x243026, 2.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
keyLight.position.set(3, 6, 4); scene.add(keyLight);
const grid = new THREE.GridHelper(24, 48, 0x344253, 0x1b2430);
grid.position.y = 0; scene.add(grid);

function resize() {
  const width = Math.max(1, viewport.clientWidth), height = Math.max(1, viewport.clientHeight);
  renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

async function api(path, body) {
  const response = await fetch(path, {method: body ? 'POST' : 'GET', headers: {'Content-Type':'application/json'}, body: body ? JSON.stringify(body) : undefined});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `${response.status} ${response.statusText}`);
  return value;
}

function makeSkeleton(joints) {
  for (const joint of joints) {
    const bone = new THREE.Bone(); bone.name = joint.name;
    if (joint.parent < 0) bone.position.set(0, 0, 0);
    else {
      const parent = joints[joint.parent].position;
      bone.position.set(joint.position[0]-parent[0], joint.position[1]-parent[1], joint.position[2]-parent[2]);
      state.bones[joint.parent].add(bone);
    }
    state.bones.push(bone);
  }
  scene.add(state.bones[0]);
  state.helper = new THREE.SkeletonHelper(state.bones[0]);
  state.helper.material.color.setHex(0x71f1c2);
  state.helper.material.depthTest = false;
  state.helper.renderOrder = 3;
  scene.add(state.helper);
  const rootMarker = new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 10), new THREE.MeshStandardMaterial({color:0xffca68, emissive:0x4d2a00}));
  state.bones[0].add(rootMarker);
}

function installStyles(styles) {
  for (const style of styles) {
    const option = document.createElement('option'); option.value = style.name;
    option.textContent = `${style.name.replaceAll('_',' ')} · ${style.speed.toFixed(1)} m/s`;
    styleSelect.append(option);
  }
  state.style = styles.some(item => item.name === 'walk') ? 'walk' : styles[0].name;
  styleSelect.value = state.style;
  styleSelect.addEventListener('change', () => { state.style = styleSelect.value; void requestPlan(); });
}

function useMotion(response) {
  state.session = response.session; state.style = response.style; styleSelect.value = response.style;
  state.motion = response.motion; state.playhead = 0; state.lastTime = performance.now();
  planInfo.textContent = `${response.motion.frames} frames · ${response.style.replaceAll('_',' ')}`;
  statusElement.textContent = 'Playing';
}

async function requestPlan(advance = Math.floor(state.playhead)) {
  if (state.pending || !state.session) return null;
  state.pending = true; statusElement.textContent = 'Planning…';
  try {
    const response = await api('/api/plan', {session:state.session, style:state.style, move:state.move, facing:state.facing, seed:state.seed++, advance});
    useMotion(response); return response;
  } finally { state.pending = false; }
}

function updateControl() {
  let x = 0, z = 0;
  if (state.keys.has('w')) z += 1; if (state.keys.has('s')) z -= 1;
  if (state.keys.has('a')) x -= 1; if (state.keys.has('d')) x += 1;
  const length = Math.hypot(x,z); if (length > 0) { x/=length; z/=length; state.facing=[x,z]; }
  state.move=[x,z];
  document.querySelectorAll('.pad button').forEach(button => button.classList.toggle('active',state.keys.has(button.dataset.key)));
}

let controlTimer = 0;
function schedulePlan() { clearTimeout(controlTimer); controlTimer=setTimeout(()=>void requestPlan(),70); }
addEventListener('keydown',event=>{
  const key=event.key.toLowerCase();
  if (['w','a','s','d'].includes(key)) { state.keys.add(key); updateControl(); schedulePlan(); event.preventDefault(); }
  if (event.key==='ArrowLeft'||event.key==='ArrowRight') {
    const angle=Math.atan2(state.facing[0],state.facing[1])+(event.key==='ArrowLeft' ? 0.25 : -0.25);
    state.facing=[Math.sin(angle),Math.cos(angle)]; schedulePlan(); event.preventDefault();
  }
});
addEventListener('keyup',event=>{const key=event.key.toLowerCase();if(state.keys.delete(key)){updateControl();schedulePlan();}});
for (const button of document.querySelectorAll('.pad button')) {
  const down=event=>{state.keys.add(button.dataset.key);updateControl();schedulePlan();event.preventDefault();};
  const up=event=>{state.keys.delete(button.dataset.key);updateControl();schedulePlan();event.preventDefault();};
  button.addEventListener('pointerdown',down); button.addEventListener('pointerup',up); button.addEventListener('pointercancel',up); button.addEventListener('pointerleave',up);
}

function renderMotion(frame) {
  const motion=state.motion;if(!motion)return;
  const index=Math.min(motion.frames-1,Math.max(0,frame));
  state.bones[0].position.fromArray(motion.roots,index*3);
  for(let joint=0;joint<motion.joints;joint++) state.bones[joint].quaternion.fromArray(motion.rotations,(index*motion.joints+joint)*4);
  const root=state.bones[0].position;
  const target=new THREE.Vector3(root.x+3.3,root.y+1.45,root.z+4.2);
  camera.position.lerp(target,.025);camera.lookAt(root.x,root.y+.85,root.z);
}

function animate(now) {
  requestAnimationFrame(animate);
  const delta=Math.min(.1,(now-state.lastTime)/1000);state.lastTime=now;
  if(state.motion){state.playhead+=delta*state.meta.fps;if(state.playhead>=state.motion.frames-5&&!state.pending)void requestPlan(Math.floor(state.playhead));renderMotion(Math.floor(state.playhead));}
  renderer.render(scene,camera);
}

async function selfTest() {
  const alternate=state.meta.styles.find(item=>item.name==='walk_zombie')||state.meta.styles.find(item=>item.name!=='walk');
  if(!alternate)throw new Error('no alternate upstream style available');
  state.style=alternate.name;styleSelect.value=alternate.name;state.move=[1,0];state.facing=[1,0];
  const response=await requestPlan(3);
  if(!response||response.style!==alternate.name)throw new Error('style change was not applied');
  if(response.motion.joints!==34||response.motion.frames<24||response.motion.rotations.length!==response.motion.frames*34*4)throw new Error('invalid skeletal animation response');
  renderMotion(2);renderer.render(scene,camera);
  if(!renderer.domElement.width||state.bones.length!==34)throw new Error('Three.js skeleton was not rendered');
  document.documentElement.dataset.testStatus='passed';testResult.textContent=`Headless check passed: ${alternate.name}, right turn, ${response.motion.frames} frames`;
}

async function start() {
  try {
    state.meta=await api('/api/meta');installStyles(state.meta.styles);makeSkeleton(state.meta.joints);
    const initial=await api('/api/session',{style:state.style});useMotion(initial);requestAnimationFrame(animate);
    if(query.get('test')==='1')await selfTest();else document.documentElement.dataset.testStatus='ready';
  } catch(error) {
    console.error(error);statusElement.textContent='Error';testResult.textContent=error.message;document.documentElement.dataset.testStatus='failed';
  }
}
await start();
