let currentMethod = null; 
let scene, camera, renderer, composer, dead = false;
let camAngle = {th: 0.6, ph: 0.2, d: 35};
let mouse = {dn:false, lx:0, ly:0};
let sceneObjects = { layers: [], extras: [], annotations: [], lasers: null };
let isExploded = false;
let trainingProgress = 0;

const ARCH = [
  { id: 'embed', name: 'Embedding', baseZ: -10, color: 0x475569, w: 8, h: 8 },
  { id: 'qkv', name: 'Q, K, V Matrices', baseZ: -4, color: 0x3B82F6, w: 12, h: 12 },
  { id: 'proj', name: 'Output Proj', baseZ: 2, color: 0x8B5CF6, w: 10, h: 10 },
  { id: 'ffn1', name: 'FFN Gate/Up', baseZ: 8, color: 0xEC4899, w: 16, h: 8 },
  { id: 'ffn2', name: 'FFN Down', baseZ: 14, color: 0x10B981, w: 10, h: 10 }
];

function toggleExplode() {
  isExploded = !isExploded;
  const btn = document.getElementById('explode-btn');
  if (isExploded) {
    btn.classList.add('text-brand-accent', 'border-brand-accent', 'bg-brand-accent/20');
    camAngle.d = 50;
  } else {
    btn.classList.remove('text-brand-accent', 'border-brand-accent', 'bg-brand-accent/20');
    camAngle.d = 35;
  }
}

function updateTrainingProgress(val) {
  trainingProgress = parseFloat(val);
  document.getElementById('epoch-val').textContent = `EPOCH: ${(trainingProgress * 3).toFixed(2)}`;
}

function pickMethod(id) {
  currentMethod = METHODS.find(m => m.id === id);
  document.querySelectorAll('#method-btns button').forEach(b => {
    const isActive = b.dataset.id === id;
    b.className = `hud-btn group ${isActive ? 'active' : ''}`;
    const nameEl = b.querySelector('span:last-child');
    if (nameEl) nameEl.className = `text-xs font-bold tracking-tight transition-colors ${isActive ? 'text-brand-accent' : 'text-brand-text2'}`;
  });

  const titleEl = document.getElementById('overlay-title');
  titleEl.innerHTML = `${currentMethod.emoji} ${currentMethod.name}`;
  titleEl.style.textShadow = `0 0 30px ${currentMethod.color}`;

  document.getElementById('desc-text').textContent = currentMethod.desc;
  document.getElementById('insight-text').textContent = currentMethod.insight;
  const formulaText = document.getElementById('formula-text');
  formulaText.textContent = currentMethod.formula;
  formulaText.style.color = currentMethod.color;
  rebuildScene();
}

function initThree() {
  const cvs = document.getElementById('three-canvas');
  if (!cvs) return;
  const container = cvs.parentElement;
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x020408, 0.015);
  camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 1000);
  renderer = new THREE.WebGLRenderer({canvas:cvs, antialias:false, powerPreference: "high-performance"});
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio > 1 ? 1.5 : 1);
  renderer.toneMapping = THREE.ReinhardToneMapping;
  renderer.toneMappingExposure = 1.2;
  
  const renderScene = new THREE.RenderPass(scene, camera);
  const bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(container.clientWidth, container.clientHeight), 2.0, 0.5, 0.1);
  composer = new THREE.EffectComposer(renderer);
  composer.addPass(renderScene);
  composer.addPass(bloomPass);

  cvs.addEventListener('mousedown', e=>{mouse={dn:true,lx:e.clientX,ly:e.clientY}});
  window.addEventListener('mousemove', e=>{
    if(!mouse.dn) return;
    camAngle.th-=(e.clientX-mouse.lx)*0.005;
    camAngle.ph=Math.max(-0.5,Math.min(0.8,camAngle.ph+(e.clientY-mouse.ly)*0.005));
    mouse.lx=e.clientX; mouse.ly=e.clientY;
  });
  window.addEventListener('mouseup', ()=>{mouse.dn=false});
  cvs.addEventListener('wheel', e=>{
    camAngle.d=Math.max(20,Math.min(80,camAngle.d+e.deltaY*0.05));
    e.preventDefault();
  },{passive:false});

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
    composer.setSize(container.clientWidth, container.clientHeight);
  });
}

function clearScene(){
  while(scene.children.length>0){
    const o = scene.children[0];
    if(o.geometry) o.geometry.dispose();
    if(o.material) {
        if(Array.isArray(o.material)) o.material.forEach(m=>m.dispose());
        else o.material.dispose();
    }
    scene.remove(o);
  }
  const annotationsContainer = document.getElementById('annotations');
  if (annotationsContainer) annotationsContainer.innerHTML = '';
  sceneObjects = { layers: [], extras: [], annotations: [], lasers: null };
}

function createAnnotation(text, obj3D) {
  const el = document.createElement('div');
  el.className = 'annotation-tag';
  el.innerHTML = `<span class="opacity-50 mr-1">///</span> ${text}`;
  const annotationsContainer = document.getElementById('annotations');
  if (annotationsContainer) annotationsContainer.appendChild(el);
  sceneObjects.annotations.push({ el, obj: obj3D });
}

function createTensorMatrix(w, h, color, isEmissive) {
  const group = new THREE.Group();
  const geo = new THREE.BoxGeometry(0.3, 0.3, 0.3);
  const mat = new THREE.MeshStandardMaterial({
    color: color, emissive: color, emissiveIntensity: isEmissive ? 1.5 : 0.05,
    metalness: 0.9, roughness: 0.1, transparent: true, opacity: isEmissive ? 0.9 : 0.3
  });
  const imesh = new THREE.InstancedMesh(geo, mat, w * h);
  let idx = 0; const dummy = new THREE.Object3D();
  for(let x=0; x<w; x++){
    for(let y=0; y<h; y++){
      dummy.position.set((x - w/2)*0.5, (y - h/2)*0.5, 0);
      dummy.updateMatrix(); imesh.setMatrixAt(idx++, dummy.matrix);
    }
  }
  group.add(imesh);
  return { group, imesh, mat };
}

function rebuildScene(){
  clearScene();
  const mc = new THREE.Color(currentMethod.color);
  scene.add(new THREE.AmbientLight(0x0c1018, 2.0));
  const dl = new THREE.DirectionalLight(0xffffff, 1.0); dl.position.set(10, 20, 10); scene.add(dl);
  const gridHelper = new THREE.GridHelper(200, 100, 0x1A2030, 0x06080F); gridHelper.position.y = -8; scene.add(gridHelper);

  ARCH.forEach((layerDef) => {
    const isLayerEmissive = currentMethod.id === 'full' || (currentMethod.id === 'prefix' && layerDef.id === 'qkv');
    const { group, mat } = createTensorMatrix(layerDef.w, layerDef.h, isLayerEmissive ? mc : layerDef.color, isLayerEmissive);
    group.position.z = layerDef.baseZ; scene.add(group);
    const anchor = new THREE.Object3D(); anchor.position.set(0, (layerDef.h/2 * 0.5) + 1, 0); group.add(anchor);
    createAnnotation(layerDef.name, anchor);
    sceneObjects.layers.push({ group, baseZ: layerDef.baseZ, mat, isEmissive: isLayerEmissive });
  });

  if (['lora', 'qlora', 'dora'].includes(currentMethod.id)) {
    const qkvLayer = sceneObjects.layers.find(l => l.baseZ === -4);
    const loraGroup = new THREE.Group();
    const { group: matA } = createTensorMatrix(8, 2, mc, true); matA.position.set(4, 0, 0.5);
    const { group: matB } = createTensorMatrix(2, 8, new THREE.Color(currentMethod.accent), true); matB.position.set(4, 0, -0.5);
    loraGroup.add(matA); loraGroup.add(matB);
    qkvLayer.group.add(loraGroup);
    const loraAnchor = new THREE.Object3D(); loraAnchor.position.set(4, 2.5, 0); loraGroup.add(loraAnchor);
    createAnnotation(`LoRA Adapter`, loraAnchor);
  }

  const laserGeo = new THREE.BoxGeometry(0.05, 0.05, 2.0);
  const laserMat = new THREE.MeshBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.9});
  const lasers = new THREE.InstancedMesh(laserGeo, laserMat, 100);
  const dummy = new THREE.Object3D(); const laserData = [];
  for(let i=0; i<100; i++) laserData.push({ x:(Math.random()-0.5)*8, y:(Math.random()-0.5)*8, z:-15+Math.random()*30, speed:0.2+Math.random()*0.3 });
  scene.add(lasers); sceneObjects.lasers = { mesh: lasers, data: laserData, dummy };
}

function updateAnnotations() {
  sceneObjects.annotations.forEach(a => {
    const worldPos = new THREE.Vector3(); a.obj.getWorldPosition(worldPos);
    worldPos.project(camera);
    if (worldPos.z > 1) { a.el.style.display = 'none'; } else {
      a.el.style.display = 'block';
      const x = (worldPos.x + 1) * renderer.domElement.clientWidth / 2;
      const y = (-worldPos.y + 1) * renderer.domElement.clientHeight / 2;
      a.el.style.transform = `translate(-50%, -50%) translate(${x}px, ${y}px)`;
    }
  });
}

function animate(){
  if(dead) return; requestAnimationFrame(animate);
  const t = performance.now()/1000;
  if(!mouse.dn) camAngle.th += 0.002;
  camera.position.x = Math.sin(camAngle.th)*Math.cos(camAngle.ph)*camAngle.d;
  camera.position.y = Math.sin(camAngle.ph)*camAngle.d + 2;
  camera.position.z = Math.cos(camAngle.th)*Math.cos(camAngle.ph)*camAngle.d;
  camera.lookAt(0,0,0);

  const explodeMult = isExploded ? 3.0 : 1.0;
  sceneObjects.layers.forEach((layer, i) => {
      layer.group.position.z += (layer.baseZ * explodeMult - layer.group.position.z) * 0.1;
      if (layer.isEmissive) layer.mat.emissiveIntensity = 1.0 + Math.sin(t * (5 + trainingProgress * 15) + i) * (0.5 + trainingProgress);
  });

  if (sceneObjects.lasers) {
      const {mesh, data, dummy} = sceneObjects.lasers;
      for(let i=0; i<data.length; i++) {
          data[i].z += data[i].speed * (1 + trainingProgress * 4);
          if (data[i].z > 15) data[i].z = -15;
          dummy.position.set(data[i].x, data[i].y, data[i].z);
          dummy.updateMatrix(); mesh.setMatrixAt(i, dummy.matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
  }
  updateAnnotations(); composer.render();
}

function initExplore() {
    initThree();
    pickMethod('lora');
    animate();
}

document.addEventListener('configReady', initExplore);
