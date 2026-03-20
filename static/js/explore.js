/**
 * Elite Engineering Laboratory - llmviz.studio
 * High-fidelity neural architecture visualization with drill-down logic.
 */

let currentMethod = null; 
let scene, camera, renderer, composer, dead = false;
let camAngle = {th: 0.6, ph: 0.2, d: 35};
let mouse = {dn:false, lx:0, ly:0, x:0, y:0};
let raycaster = new THREE.Raycaster();
let sceneObjects = { layers: [], extras: [], annotations: [], forwardParticles: null, backwardParticles: null };
let isExploded = false;
let trainingProgress = 0;
let focusedLayerId = null;
let rgbShiftPass;
let time = 0;

const THEME = {
    base: 0x05070a,
    glass: 0x1a1e2e,
    wireframe: 0x00f2ff,
    accent: 0x00f2ff,
    gradient: 0x7000ff,
    grid: 0x161b22,
};

const ARCH = [
  { 
    id: 'embed', name: 'EMBEDDING_VECT', baseZ: -12, color: 0x475569, w: 8, h: 8, slices: 4,
    desc: 'Transforms discrete tokens (words) into continuous vectors. This is where the model "conceptualizes" the meaning of text.',
    math: 'E = Input_{one-hot} × W_{embed}',
    specs: { 'Type': 'Lookup Table', 'Input': 'Vocab ID', 'Output': 'Hidden State' }
  },
  { 
    id: 'qkv', name: 'ATTN_QKV_HEADS', baseZ: -4, color: 0x3B82F6, w: 12, h: 12, slices: 32,
    desc: 'Query, Key, and Value matrices. The core of Self-Attention that allows the model to find relationships between words.',
    math: 'Attention(Q,K,V) = softmax(QK^T / √d_k)V',
    specs: { 'Complexity': 'O(n^2)', 'Heads': '32', 'Mechanism': 'GQA/MHA' }
  },
  { 
    id: 'proj', name: 'DENSE_PROJECTION', baseZ: 4, color: 0x8B5CF6, w: 10, h: 10, slices: 8,
    desc: 'Combines outputs from multiple attention heads back into a unified vector for the next stage of processing.',
    math: 'Output = Concat(head_1, ..., head_n)W_O',
    specs: { 'Type': 'Linear', 'Params': 'd_model^2', 'Ops': 'Matrix Mul' }
  },
  { 
    id: 'ffn1', name: 'FFN_UP_GATE', baseZ: 12, color: 0xEC4899, w: 16, h: 8, slices: 16,
    desc: 'Feed-Forward Network Up-Projection. Expands dimensionality to allow the model to learn complex non-linear features.',
    math: 'Intermediate = SiLU(xW_1) ⊗ xW_2',
    specs: { 'Expansion': '4x-8x', 'Activation': 'SwiGLU', 'Compute': 'Heavy' }
  },
  { 
    id: 'ffn2', name: 'FFN_DOWN_PROJ', baseZ: 20, color: 0x10B981, w: 10, h: 10, slices: 8,
    desc: 'Projects the expanded data back to the model dimension, summarizing the newly learned high-level features.',
    math: 'FFN(x) = Intermediate × W_3',
    specs: { 'Type': 'Linear', 'Role': 'Aggregation', 'Output': 'Resid. Add' }
  }
];

function initThree() {
  const cvs = document.getElementById('three-canvas');
  if (!cvs) return;
  const container = cvs.parentElement;
  
  scene = new THREE.Scene();
  scene.background = new THREE.Color(THEME.base);
  scene.fog = new THREE.FogExp2(THEME.base, 0.015);

  camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 1000);
  
  renderer = new THREE.WebGLRenderer({canvas:cvs, antialias:true, alpha: true});
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio for performance
  
  // High-End Environment
  scene.add(new THREE.AmbientLight(0xffffff, 0.3));
  
  const mainLight = new THREE.DirectionalLight(0xffffff, 1.5);
  mainLight.position.set(15, 25, 15);
  scene.add(mainLight);

  const accentLight = new THREE.PointLight(THEME.accent, 3, 100);
  accentLight.position.set(-15, 10, 10);
  scene.add(accentLight);

  const gradientLight = new THREE.PointLight(THEME.gradient, 2, 100);
  gradientLight.position.set(15, -10, 20);
  scene.add(gradientLight);

  // Infinite Mirror Floor
  const floorGeo = new THREE.PlaneGeometry(200, 200);
  const floorMat = new THREE.MeshStandardMaterial({
      color: 0x020305,
      metalness: 0.9,
      roughness: 0.1,
  });
  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -12;
  scene.add(floor);

  // Server Chassis Wireframe
  const chassisGeo = new THREE.BoxGeometry(40, 30, 60);
  const chassisMat = new THREE.LineBasicMaterial({ color: 0x1e293b, transparent: true, opacity: 0.3 });
  const chassis = new THREE.LineSegments(new THREE.EdgesGeometry(chassisGeo), chassisMat);
  chassis.position.y = 3;
  scene.add(chassis);

  // Post-Processing
  const renderScene = new THREE.RenderPass(scene, camera);
  const bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(container.clientWidth, container.clientHeight), 1.2, 0.5, 0.85);
  
  composer = new THREE.EffectComposer(renderer);
  composer.addPass(renderScene);
  composer.addPass(bloomPass);

  // Digital Aberration (RGB Shift)
  if (THREE.RGBShiftShader) {
      rgbShiftPass = new THREE.ShaderPass(THREE.RGBShiftShader);
      rgbShiftPass.uniforms['amount'].value = 0.0015;
      composer.addPass(rgbShiftPass);
  }

  cvs.addEventListener('mousedown', e=>{mouse={...mouse, dn:true,lx:e.clientX,ly:e.clientY}});
  cvs.addEventListener('click', e => {
      if (Math.abs(e.clientX - mouse.lx) > 5 || Math.abs(e.clientY - mouse.ly) > 5) return;
      checkClick(e);
  });

  window.addEventListener('mousemove', e=>{
    const rect = cvs.getBoundingClientRect();
    // Normalize mouse coordinates for Three.js (-1 to +1)
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    if(!mouse.dn) {
        checkHover(e);
        return;
    }
    camAngle.th-=(e.clientX-mouse.lx)*0.005;
    camAngle.ph=Math.max(-0.5,Math.min(0.8,camAngle.ph+(e.clientY-mouse.ly)*0.005));
    mouse.lx=e.clientX; mouse.ly=e.clientY;
  });
  window.addEventListener('mouseup', ()=>{mouse.dn=false});
  cvs.addEventListener('wheel', e=>{
    camAngle.d=Math.max(15,Math.min(100,camAngle.d+e.deltaY*0.05));
    e.preventDefault();
  },{passive:false});

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
    composer.setSize(container.clientWidth, container.clientHeight);
  });
}

function createWaferLayer(def, methodColor) {
    const group = new THREE.Group();
    
    // Wafer Instancing
    const sliceZ = 0.1;
    const gap = 0.2;
    const totalDepth = def.slices * (sliceZ + gap);
    
    const geo = new THREE.BoxGeometry(def.w * 0.5, def.h * 0.5, sliceZ);
    const mat = new THREE.MeshStandardMaterial({
        color: THEME.glass,
        metalness: 0.9,
        roughness: 0.1,
        transparent: true,
        opacity: 0.5,
        emissive: methodColor,
        emissiveIntensity: 0
    });

    const imesh = new THREE.InstancedMesh(geo, mat, def.slices);
    const dummy = new THREE.Object3D();
    for(let i=0; i<def.slices; i++) {
        dummy.position.set(0, 0, (i - def.slices/2) * (sliceZ + gap));
        dummy.updateMatrix();
        imesh.setMatrixAt(i, dummy.matrix);
    }
    group.add(imesh);

    // Bounding Wireframe
    const boundGeo = new THREE.BoxGeometry(def.w * 0.5 + 0.2, def.h * 0.5 + 0.2, totalDepth + 0.2);
    const wireMat = new THREE.LineBasicMaterial({ color: THEME.wireframe, transparent: true, opacity: 0.2 });
    const wire = new THREE.LineSegments(new THREE.EdgesGeometry(boundGeo), wireMat);
    group.add(wire);

    return { group, imesh, mat, wire, slices: def.slices, sliceZ, gap };
}

function rebuildScene() {
    if (!scene) return;
    if (!currentMethod) currentMethod = METHODS[0];

    // Clear specific objects, keep lights/floor/chassis
    const keep = [];
    scene.children.forEach(c => {
        if (c.type === 'AmbientLight' || c.type === 'DirectionalLight' || c.type === 'PointLight' || c.geometry?.type === 'PlaneGeometry' || c.type === 'LineSegments') {
            keep.push(c);
        }
    });
    
    while(scene.children.length > 0){
        const o = scene.children[0];
        if(!keep.includes(o)) {
            if(o.geometry) o.geometry.dispose();
            if(o.material) {
                if(Array.isArray(o.material)) o.material.forEach(m=>m.dispose());
                else o.material.dispose();
            }
            scene.remove(o);
        } else {
            scene.remove(o); // Re-add later
        }
    }
    keep.forEach(k => scene.add(k));
    
    document.getElementById('annotations').innerHTML = '';
    sceneObjects = { layers: [], extras: [], annotations: [], forwardParticles: null, backwardParticles: null };

    const mc = new THREE.Color(currentMethod.color);

    ARCH.forEach(def => {
        const isEmissive = currentMethod.id === 'full' || (currentMethod.id === 'prefix' && def.id === 'qkv');
        const layer = createWaferLayer(def, isEmissive ? mc : 0x000000);
        layer.group.position.z = def.baseZ;
        scene.add(layer.group);
        
        const anchor = new THREE.Object3D();
        anchor.position.set(0, (def.h * 0.25) + 2.0, 0);
        layer.group.add(anchor);
        createAnnotation(def.name, anchor);
        
        sceneObjects.layers.push({ ...layer, baseZ: def.baseZ, isEmissive, id: def.id, def });
    });

    if (['lora', 'qlora', 'dora'].includes(currentMethod.id)) {
        const qkv = sceneObjects.layers.find(l => l.id === 'qkv');
        const loraGroup = new THREE.Group();
        const matA = new THREE.Mesh(new THREE.BoxGeometry(2, 8, 0.2), new THREE.MeshStandardMaterial({color: THEME.accent, emissive: THEME.accent, emissiveIntensity: 2}));
        matA.position.set(5, 0, 0.5);
        const matB = new THREE.Mesh(new THREE.BoxGeometry(0.2, 8, 3), new THREE.MeshStandardMaterial({color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 1}));
        matB.position.set(6, 0, -0.5);
        loraGroup.add(matA); loraGroup.add(matB);
        qkv.group.add(loraGroup);
        const loraAnchor = new THREE.Object3D(); loraAnchor.position.set(6, 5, 0); loraGroup.add(loraAnchor);
        createAnnotation('LORA_ADAPTER_CORE', loraAnchor);
    }

    initDataParticles();
}

function initDataParticles() {
    // Forward Pass (Inference/Compute)
    const fCount = 1500;
    const fGeo = new THREE.BufferGeometry();
    const fPos = new Float32Array(fCount * 3);
    for(let i=0; i<fCount; i++) {
        fPos[i*3] = (Math.random()-0.5) * 12;
        fPos[i*3+1] = (Math.random()-0.5) * 12;
        fPos[i*3+2] = -25 + Math.random() * 50;
    }
    fGeo.setAttribute('position', new THREE.BufferAttribute(fPos, 3));
    const fMat = new THREE.PointsMaterial({ color: THEME.accent, size: 0.08, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending });
    const fPoints = new THREE.Points(fGeo, fMat);
    scene.add(fPoints);

    // Backward Pass (Gradients)
    const bCount = 1000;
    const bGeo = new THREE.BufferGeometry();
    const bPos = new Float32Array(bCount * 3);
    for(let i=0; i<bCount; i++) {
        bPos[i*3] = (Math.random()-0.5) * 12;
        bPos[i*3+1] = (Math.random()-0.5) * 12;
        bPos[i*3+2] = -25 + Math.random() * 50;
    }
    bGeo.setAttribute('position', new THREE.BufferAttribute(bPos, 3));
    const bMat = new THREE.PointsMaterial({ color: THEME.gradient, size: 0.08, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending });
    const bPoints = new THREE.Points(bGeo, bMat);
    scene.add(bPoints);

    sceneObjects.forwardParticles = { points: fPoints, pos: fPos };
    sceneObjects.backwardParticles = { points: bPoints, pos: bPos };
}

function animate() {
    if(dead) return;
    requestAnimationFrame(animate);
    
    time += 0.005;
    const dt = 0.016;

    // Cinematic Camera Drift
    let targetTh = camAngle.th;
    let targetPh = camAngle.ph;
    
    if(!mouse.dn && !focusedLayerId) {
        targetTh += time * 0.02;
        // Drone-like wobble
        targetPh += Math.sin(time) * 0.01;
    }

    camera.position.x = Math.sin(targetTh) * Math.cos(targetPh) * camAngle.d;
    camera.position.y = Math.sin(targetPh) * camAngle.d + 2 + Math.sin(time*2)*0.5; // Vertical bob
    camera.position.z = Math.cos(targetTh) * Math.cos(targetPh) * camAngle.d;
    camera.lookAt(0, 0, 0);

    // RGB Shift intensity based on training
    if (rgbShiftPass) {
        rgbShiftPass.uniforms['amount'].value = 0.001 + (trainingProgress * 0.003);
    }

    // Explode Logic & Layer Animation
    const explodeMult = isExploded ? 2.5 : 1.0;
    sceneObjects.layers.forEach((layer, i) => {
        // Move entire group
        layer.group.position.z += (layer.baseZ * explodeMult - layer.group.position.z) * 0.1;
        
        // Unstack Wafers if focused
        const dummy = new THREE.Object3D();
        const unstackTarget = (focusedLayerId === layer.id && isExploded) ? 1.5 : 1.0;
        
        for(let j=0; j<layer.slices; j++) {
            // Read current matrix
            const mat = new THREE.Matrix4();
            layer.imesh.getMatrixAt(j, mat);
            const pos = new THREE.Vector3().setFromMatrixPosition(mat);
            
            // Target Z
            const targetZ = (j - layer.slices/2) * (layer.sliceZ + layer.gap) * unstackTarget;
            pos.z += (targetZ - pos.z) * 0.1;
            
            dummy.position.copy(pos);
            dummy.updateMatrix();
            layer.imesh.setMatrixAt(j, dummy.matrix);
        }
        layer.imesh.instanceMatrix.needsUpdate = true;

        // Weight Excitation (Pulsing when training)
        if (layer.isEmissive || trainingProgress > 0) {
            const baseGlow = layer.isEmissive ? 0.5 : 0;
            const trainGlow = Math.sin(time * 10 + i) * trainingProgress * 2;
            layer.mat.emissiveIntensity = baseGlow + trainGlow;
            layer.wire.material.opacity = 0.2 + (trainGlow * 0.2);
        }
    });

    // Bi-Directional Flow
    if (sceneObjects.forwardParticles) {
        const {pos} = sceneObjects.forwardParticles;
        for(let i=0; i<pos.length/3; i++) {
            pos[i*3+2] += 0.2 * (1 + trainingProgress * 2); // Forward +z
            if(pos[i*3+2] > 25) pos[i*3+2] = -25;
        }
        sceneObjects.forwardParticles.points.geometry.attributes.position.needsUpdate = true;
    }
    
    if (sceneObjects.backwardParticles) {
        const {pos, points} = sceneObjects.backwardParticles;
        points.visible = trainingProgress > 0; // Only show backwards on training
        if (points.visible) {
            for(let i=0; i<pos.length/3; i++) {
                pos[i*3+2] -= 0.15 * (trainingProgress * 4); // Backward -z
                if(pos[i*3+2] < -25) pos[i*3+2] = 25;
            }
            points.geometry.attributes.position.needsUpdate = true;
        }
    }

    updateAnnotations();
    composer.render();
}

function checkClick(e) {
    raycaster.setFromCamera(mouse, camera);
    
    // Only target objects that are part of our architecture layers
    const targetObjects = [];
    sceneObjects.layers.forEach(l => {
        targetObjects.push(l.imesh);
        targetObjects.push(l.wire);
        l.group.traverse(child => {
            if (child.isMesh) targetObjects.push(child);
        });
    });
    
    const intersects = raycaster.intersectObjects(targetObjects, true);
    
    if (intersects.length > 0) {
        let obj = intersects[0].object;
        let layer = null;
        
        // Find which layer this object belongs to
        let current = obj;
        while (current) {
            layer = sceneObjects.layers.find(l => l.group === current || l.imesh === current);
            if (layer) break;
            current = current.parent;
        }

        if (layer) {
            const arch = ARCH.find(a => a.id === layer.id);
            if (arch) {
                if (focusedLayerId === arch.id) unfocusLayer();
                else focusLayer(arch, layer);
            }
            return;
        }
    }
    if (focusedLayerId) unfocusLayer();
}

function focusLayer(arch, layer) {
    focusedLayerId = arch.id;
    const side = document.getElementById('side-insight');
    if (side) side.classList.remove('translate-x-[400px]');
    
    const titleEl = document.getElementById('side-title');
    if(titleEl) titleEl.textContent = arch.name;
    
    const descEl = document.getElementById('side-desc');
    if(descEl) descEl.textContent = arch.desc;
    
    const mathEl = document.getElementById('side-math');
    if(mathEl) mathEl.textContent = arch.math;
    
    const specsContainer = document.getElementById('side-specs');
    if (specsContainer) {
        specsContainer.innerHTML = Object.entries(arch.specs).map(([k, v]) => `
            <div class="p-3 rounded-xl bg-black/40 border border-brand-accent/20 shadow-[0_0_10px_rgba(0,242,255,0.05)]">
                <div class="text-[7px] text-brand-text3 uppercase mb-1 font-bold">${k}</div>
                <div class="text-[10px] font-bold text-white truncate">${v}</div>
            </div>
        `).join('');
    }

    // Isolate visually
    sceneObjects.layers.forEach(l => {
        const isTarget = l.id === arch.id;
        l.mat.opacity = isTarget ? 0.8 : 0.05;
        l.wire.material.opacity = isTarget ? 0.5 : 0.02;
    });
    
    camAngle.d = 20; // Zoom in
    if (!isExploded) toggleExplode(); // Auto-explode to see wafers
}

function unfocusLayer() {
    focusedLayerId = null;
    const side = document.getElementById('side-insight');
    if (side) side.classList.add('translate-x-[400px]');
    
    // Restore
    sceneObjects.layers.forEach(l => {
        l.mat.opacity = 0.5;
        l.wire.material.opacity = 0.2;
    });
    camAngle.d = 35; 
}

function toggleExplode() {
  isExploded = !isExploded;
  const btn = document.getElementById('explode-btn');
  if (isExploded) {
    btn?.classList.add('text-brand-bg', 'bg-brand-accent');
  } else {
    btn?.classList.remove('text-brand-bg', 'bg-brand-accent');
  }
}

function updateTrainingProgress(val) {
  trainingProgress = parseFloat(val);
  const epochVal = document.getElementById('epoch-val');
  if (epochVal) epochVal.textContent = `EPOCH: ${(trainingProgress * 3).toFixed(2)}`;
}

function pickMethod(id) {
  currentMethod = METHODS.find(m => m.id === id);
  if (!currentMethod) return;

  document.querySelectorAll('#method-btns button').forEach(b => {
    const isActive = b.dataset.id === id;
    b.classList.toggle('active', isActive);
    const nameEl = b.querySelector('span:last-child');
    if (nameEl) {
        nameEl.classList.toggle('text-brand-accent', isActive);
        nameEl.classList.toggle('text-brand-text2', !isActive);
    }
  });

  const titleEl = document.getElementById('overlay-title');
  if (titleEl) {
    titleEl.innerHTML = `${currentMethod.emoji} ${currentMethod.name}`;
    titleEl.style.textShadow = `0 0 30px ${currentMethod.color}`;
  }

  const descEl = document.getElementById('desc-text');
  if (descEl) descEl.textContent = currentMethod.desc;
  
  const insightEl = document.getElementById('insight-text');
  if (insightEl) insightEl.textContent = currentMethod.insight;
  
  const formulaText = document.getElementById('formula-text');
  if (formulaText) {
    formulaText.textContent = currentMethod.formula;
    formulaText.style.color = currentMethod.color;
  }
  rebuildScene();
}

function createAnnotation(text, obj3D) {
  const el = document.createElement('div');
  el.className = 'annotation-tag';
  el.innerHTML = `<span class="text-brand-accent opacity-50 mr-2">⌖</span> ${text}`;
  document.getElementById('annotations')?.appendChild(el);
  sceneObjects.annotations.push({ el, obj: obj3D });
}

function checkHover(e) {
    raycaster.setFromCamera(mouse, camera);
    const meshes = [];
    scene.traverse(o => { if (o.isMesh || o.isInstancedMesh) meshes.push(o); });
    const intersects = raycaster.intersectObjects(meshes, true);
    const tooltip = document.getElementById('hover-tooltip');
    if (!tooltip) return;
    
    // Don't show tooltip if sidebar is open
    if (focusedLayerId) {
        tooltip.classList.add('hidden');
        return;
    }

    if (intersects.length > 0) {
        let obj = intersects[0].object;
        let layer = null;
        sceneObjects.layers.forEach(l => {
            if (l.imesh === obj || l.group === obj.parent || l.group === obj.parent?.parent) layer = l;
        });
        if (layer) {
            tooltip.classList.remove('hidden');
            tooltip.style.left = `${e.clientX + 20}px`;
            tooltip.style.top = `${e.clientY + 20}px`;
            const arch = ARCH.find(a => a.id === layer.id);
            document.getElementById('tooltip-title').textContent = arch ? arch.name : 'ADAPTER';
            const dimText = arch ? `SLICES: ${layer.slices} (Depth)` : 'NODE: ACTIVE';
            document.getElementById('tooltip-dim').innerHTML = `${dimText}<br><span class="text-brand-text3 text-[8px] mt-1 block">CLICK TO ISOLATE</span>`;
            
            // Highlight cursor
            document.getElementById('three-canvas').style.cursor = 'pointer';
            return;
        }
    }
    tooltip.classList.add('hidden');
    document.getElementById('three-canvas').style.cursor = 'grab';
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

function initExplore() {
    initThree();
    rebuildScene();
    animate();
}

document.addEventListener('configReady', initExplore);
