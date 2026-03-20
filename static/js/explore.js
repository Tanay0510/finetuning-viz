let currentMethod = null; 
let scene, camera, renderer, composer, dead = false;
let camAngle = {th: 0.6, ph: 0.2, d: 35};
let mouse = {dn:false, lx:0, ly:0, x:0, y:0};
let raycaster = new THREE.Raycaster();
let sceneObjects = { layers: [], extras: [], annotations: [], dataParticles: null };
let isExploded = false;
let trainingProgress = 0;

const ARCH = [
  { 
    id: 'embed', name: 'EMBEDDING_VECT', baseZ: -10, color: 0x475569, w: 8, h: 8,
    desc: 'Transforms discrete tokens (words) into continuous vectors. This is where the model "conceptualizes" the meaning of text.'
  },
  { 
    id: 'qkv', name: 'ATTN_QKV_HEADS', baseZ: -4, color: 0x3B82F6, w: 12, h: 12,
    desc: 'Query, Key, and Value matrices. The core of Self-Attention that allows the model to find relationships between words.'
  },
  { 
    id: 'proj', name: 'DENSE_PROJECTION', baseZ: 2, color: 0x8B5CF6, w: 10, h: 10,
    desc: 'Combines outputs from multiple attention heads back into a unified vector for the next stage of processing.'
  },
  { 
    id: 'ffn1', name: 'FFN_UP_GATE', baseZ: 8, color: 0xEC4899, w: 16, h: 8,
    desc: 'Feed-Forward Network Up-Projection. Expands dimensionality to allow the model to learn complex non-linear features.'
  },
  { 
    id: 'ffn2', name: 'FFN_DOWN_PROJ', baseZ: 14, color: 0x10B981, w: 10, h: 10,
    desc: 'Projects the expanded data back to the model dimension, summarizing the newly learned high-level features.'
  }
];

function toggleExplode() {
  isExploded = !isExploded;
  const btn = document.getElementById('explode-btn');
  if (isExploded) {
    btn?.classList.add('text-brand-accent', 'border-brand-accent', 'bg-brand-accent/20');
    camAngle.d = 50;
  } else {
    btn?.classList.remove('text-brand-accent', 'border-brand-accent', 'bg-brand-accent/20');
    camAngle.d = 35;
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

function initThree() {
  const cvs = document.getElementById('three-canvas');
  if (!cvs) return;
  const container = cvs.parentElement;
  
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070a);
  scene.fog = new THREE.FogExp2(0x05070a, 0.01);

  camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 1000);
  
  renderer = new THREE.WebGLRenderer({canvas:cvs, antialias:true, alpha: true});
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  
  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
  mainLight.position.set(10, 20, 10);
  scene.add(mainLight);

  const accentLight = new THREE.PointLight(0x00f2ff, 2, 100);
  accentLight.position.set(-10, 10, 10);
  scene.add(accentLight);

  const grid = new THREE.GridHelper(200, 50, 0x1e293b, 0x0a0d14);
  grid.position.y = -10;
  scene.add(grid);

  // Composer for Bloom
  const renderScene = new THREE.RenderPass(scene, camera);
  const bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(container.clientWidth, container.clientHeight), 1.0, 0.4, 0.85);
  composer = new THREE.EffectComposer(renderer);
  composer.addPass(renderScene);
  composer.addPass(bloomPass);

  cvs.addEventListener('mousedown', e=>{mouse={...mouse, dn:true,lx:e.clientX,ly:e.clientY}});
  window.addEventListener('mousemove', e=>{
    const rect = cvs.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    if(!mouse.dn) { checkHover(e); return; }
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

function createTensorLayer(def, methodColor) {
    const group = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({
        color: 0x1a1e2e,
        metalness: 0.8,
        roughness: 0.2,
        transparent: true,
        opacity: 0.6,
        emissive: methodColor,
        emissiveIntensity: 0
    });

    const geo = new THREE.BoxGeometry(def.w * 0.5, def.h * 0.5, 0.4);
    const mesh = new THREE.Mesh(geo, mat);
    group.add(mesh);

    const wireGeo = new THREE.EdgesGeometry(geo);
    const wireMat = new THREE.LineBasicMaterial({ color: 0x00f2ff, transparent: true, opacity: 0.3 });
    const wire = new THREE.LineSegments(wireGeo, wireMat);
    group.add(wire);

    return { group, mesh, mat, wire };
}

function rebuildScene() {
    if (!scene) return;
    if (!currentMethod) currentMethod = METHODS[0];

    while(scene.children.length > 4) {
        const o = scene.children[4];
        if(o.geometry) o.geometry.dispose();
        if(o.material) {
            if(Array.isArray(o.material)) o.material.forEach(m=>m.dispose());
            else o.material.dispose();
        }
        scene.remove(o);
    }
    
    document.getElementById('annotations').innerHTML = '';
    sceneObjects = { layers: [], extras: [], annotations: [], dataParticles: null };

    const mc = new THREE.Color(currentMethod.color);

    ARCH.forEach(def => {
        const isEmissive = currentMethod.id === 'full' || (currentMethod.id === 'prefix' && def.id === 'qkv');
        const layer = createTensorLayer(def, isEmissive ? mc : 0x000000);
        layer.group.position.z = def.baseZ;
        scene.add(layer.group);
        
        const anchor = new THREE.Object3D();
        anchor.position.set(0, (def.h * 0.25) + 1.5, 0);
        layer.group.add(anchor);
        createAnnotation(def.name, anchor);
        
        sceneObjects.layers.push({ ...layer, baseZ: def.baseZ, isEmissive });
    });

    if (['lora', 'qlora', 'dora'].includes(currentMethod.id)) {
        const qkv = sceneObjects.layers.find(l => l.baseZ === -4);
        const loraGroup = new THREE.Group();
        const matA = new THREE.Mesh(new THREE.BoxGeometry(2, 6, 0.1), new THREE.MeshStandardMaterial({color: mc, emissive: mc, emissiveIntensity: 2}));
        matA.position.set(4, 0, 0.2);
        const matB = new THREE.Mesh(new THREE.BoxGeometry(0.1, 6, 2), new THREE.MeshStandardMaterial({color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 1}));
        matB.position.set(4.5, 0, 0);
        loraGroup.add(matA); loraGroup.add(matB);
        qkv.group.add(loraGroup);
        const loraAnchor = new THREE.Object3D(); loraAnchor.position.set(4.5, 4, 0); loraGroup.add(loraAnchor);
        createAnnotation('LORA_ADAPTER', loraAnchor);
    }

    initDataParticles();
}

function initDataParticles() {
    const count = 1000;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count);
    for(let i=0; i<count; i++) {
        pos[i*3] = (Math.random()-0.5) * 15;
        pos[i*3+1] = (Math.random()-0.5) * 15;
        pos[i*3+2] = -30 + Math.random() * 60;
        vel[i] = 0.1 + Math.random() * 0.2;
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color: 0x00f2ff, size: 0.05, transparent: true, opacity: 0.4 });
    const points = new THREE.Points(geo, mat);
    scene.add(points);
    sceneObjects.dataParticles = { points, pos, vel };
}

function animate() {
    if(dead) return;
    requestAnimationFrame(animate);
    const t = performance.now() * 0.001;

    if(!mouse.dn) camAngle.th += 0.002;
    camera.position.x = Math.sin(camAngle.th) * Math.cos(camAngle.ph) * camAngle.d;
    camera.position.y = Math.sin(camAngle.ph) * camAngle.d + 2;
    camera.position.z = Math.cos(camAngle.th) * Math.cos(camAngle.ph) * camAngle.d;
    camera.lookAt(0, 0, 0);

    const explodeMult = isExploded ? 3.0 : 1.0;
    sceneObjects.layers.forEach((layer, i) => {
        layer.group.position.z += (layer.baseZ * explodeMult - layer.group.position.z) * 0.05;
        if (layer.isEmissive) {
            layer.mat.emissiveIntensity = (0.5 + Math.sin(t * 3 + i) * 0.5) * (1 + trainingProgress * 2);
        }
    });

    if (sceneObjects.dataParticles) {
        const {pos} = sceneObjects.dataParticles;
        for(let i=0; i<pos.length/3; i++) {
            pos[i*3+2] += 0.2 * (1 + trainingProgress * 5);
            if(pos[i*3+2] > 30) pos[i*3+2] = -30;
        }
        sceneObjects.dataParticles.points.geometry.attributes.position.needsUpdate = true;
    }

    updateAnnotations();
    if (composer) composer.render();
    else renderer.render(scene, camera);
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
    const intersects = raycaster.intersectObjects(scene.children, true);
    const tooltip = document.getElementById('hover-tooltip');
    if (!tooltip) return;
    if (intersects.length > 0) {
        let obj = intersects[0].object;
        let layer = null;
        sceneObjects.layers.forEach(l => {
            if (l.group === obj.parent || l.group === obj.parent?.parent) layer = l;
        });
        if (layer) {
            tooltip.classList.remove('hidden');
            tooltip.style.left = `${e.clientX + 20}px`;
            tooltip.style.top = `${e.clientY + 20}px`;
            const arch = ARCH.find(a => a.baseZ === layer.baseZ);
            
            document.getElementById('tooltip-title').textContent = arch ? arch.name : 'ADAPTER_SCHEMA';
            
            // Refined Dimension & Logic Info
            const dimText = arch ? `DIM: ${arch.w * 512}x${arch.h * 512}` : 'TYPE: RANK_DECOMPOSED';
            const descText = arch ? `<div class="mt-2 pt-2 border-t border-white/10 text-[10px] text-brand-text2 leading-tight max-w-[200px] normal-case font-sans italic">${arch.desc}</div>` : '';
            
            document.getElementById('tooltip-dim').innerHTML = `${dimText}${descText}`;
            return;
        }
    }
    tooltip.classList.add('hidden');
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
