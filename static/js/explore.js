/**
 * Pro-Grade 3D Explorer - llmviz.studio
 * High-fidelity neural architecture visualization.
 */

let currentMethod = null; 
let scene, camera, renderer, composer, dead = false;
let camAngle = {th: 0.6, ph: 0.2, d: 35};
let mouse = {dn:false, lx:0, ly:0, x:0, y:0};
let raycaster = new THREE.Raycaster();
let sceneObjects = { layers: [], extras: [], annotations: [], dataParticles: null };
let isExploded = false;
let trainingProgress = 0;

// High-End Industrial Palette
const THEME = {
    base: 0x0a0c14,
    glass: 0x1a1e2e,
    wireframe: 0x3b82f6,
    accent: 0x60a5fa,
    grid: 0x161b22,
    particle: 0x3b82f6
};

const ARCH = [
  { id: 'embed', name: 'EMBEDDING_VECT', baseZ: -10, color: 0x475569, w: 8, h: 8 },
  { id: 'qkv', name: 'ATTN_QKV_HEADS', baseZ: -4, color: 0x3B82F6, w: 12, h: 12 },
  { id: 'proj', name: 'DENSE_PROJECTION', baseZ: 2, color: 0x8B5CF6, w: 10, h: 10 },
  { id: 'ffn1', name: 'FFN_UP_GATE', baseZ: 8, color: 0xEC4899, w: 16, h: 8 },
  { id: 'ffn2', name: 'FFN_DOWN_PROJ', baseZ: 14, color: 0x10B981, w: 10, h: 10 }
];

function initThree() {
  const cvs = document.getElementById('three-canvas');
  if (!cvs) return;
  const container = cvs.parentElement;
  
  scene = new THREE.Scene();
  scene.background = new THREE.Color(THEME.base);
  scene.fog = new THREE.FogExp2(THEME.base, 0.02);

  camera = new THREE.PerspectiveCamera(40, container.clientWidth/container.clientHeight, 0.1, 1000);
  
  renderer = new THREE.WebGLRenderer({canvas:cvs, antialias:true, powerPreference: "high-performance"});
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  
  // High-End Post Processing
  const renderScene = new THREE.RenderPass(scene, camera);
  const bloomPass = new THREE.UnrealBloomPass(new THREE.Vector2(container.clientWidth, container.clientHeight), 1.2, 0.4, 0.85);
  composer = new THREE.EffectComposer(renderer);
  composer.addPass(renderScene);
  composer.addPass(bloomPass);

  // Studio Lighting
  const ambient = new THREE.AmbientLight(0xffffff, 0.2);
  scene.add(ambient);

  const spot = new THREE.SpotLight(0xffffff, 100);
  spot.position.set(20, 40, 20);
  spot.angle = 0.3;
  spot.penumbra = 1;
  scene.add(spot);

  const rimLight = new THREE.DirectionalLight(0x3b82f6, 2);
  rimLight.position.set(-10, -10, -10);
  scene.add(rimLight);

  // Technical Grid
  const grid = new THREE.GridHelper(200, 80, 0x1e293b, 0x0f172a);
  grid.position.y = -10;
  scene.add(grid);

  // Interaction
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
    
    // Pro Material: Frosted Glass / Metallic mix
    const mat = new THREE.MeshStandardMaterial({
        color: 0x1a1e2e,
        metalness: 0.9,
        roughness: 0.2,
        transparent: true,
        opacity: 0.4,
        emissive: methodColor,
        emissiveIntensity: 0
    });

    const geo = new THREE.BoxGeometry(def.w * 0.5, def.h * 0.5, 0.4);
    const mesh = new THREE.Mesh(geo, mat);
    group.add(mesh);

    // Wireframe Overlay (Technical Detail)
    const wireGeo = new THREE.EdgesGeometry(geo);
    const wireMat = new THREE.LineBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.2 });
    const wire = new THREE.LineSegments(wireGeo, wireMat);
    group.add(wire);

    // Subtle Internal Nodes (Instanced)
    const nodeGeo = new THREE.BoxGeometry(0.1, 0.1, 0.1);
    const nodeMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.5 });
    const count = 16;
    const imesh = new THREE.InstancedMesh(nodeGeo, nodeMat, count);
    const dummy = new THREE.Object3D();
    for(let i=0; i<count; i++) {
        dummy.position.set((Math.random()-0.5)*def.w*0.4, (Math.random()-0.5)*def.h*0.4, 0);
        dummy.updateMatrix();
        imesh.setMatrixAt(i, dummy.matrix);
    }
    group.add(imesh);

    return { group, mesh, mat, wire };
}

function rebuildScene() {
    if (!scene) return;
    // Clear
    while(scene.children.length > 4) { // Keep lights/grid
        const o = scene.children[4];
        if(o.geometry) o.geometry.dispose();
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

    // High-End Adapter Visualization
    if (['lora', 'qlora', 'dora'].includes(currentMethod.id)) {
        const qkv = sceneObjects.layers.find(l => l.baseZ === -4);
        const loraGroup = new THREE.Group();
        
        // Matrix A & B as thin sleek slabs
        const matA = new THREE.Mesh(new THREE.BoxGeometry(2, 6, 0.1), new THREE.MeshStandardMaterial({color: mc, emissive: mc, emissiveIntensity: 2}));
        matA.position.set(4, 0, 0.2);
        
        const matB = new THREE.Mesh(new THREE.BoxGeometry(0.1, 6, 2), new THREE.MeshStandardMaterial({color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 1}));
        matB.position.set(4.5, 0, 0);
        
        loraGroup.add(matA);
        loraGroup.add(matB);
        qkv.group.add(loraGroup);
        
        const loraAnchor = new THREE.Object3D();
        loraAnchor.position.set(4.5, 4, 0);
        loraGroup.add(loraAnchor);
        createAnnotation('LORA_ADAPTER_CORE', loraAnchor);
    }

    // Pro Particle Data Flow
    initDataParticles();
}

function initDataParticles() {
    const count = 2000;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    const vel = new Float32Array(count);
    
    for(let i=0; i<count; i++) {
        pos[i*3] = (Math.random()-0.5) * 10;
        pos[i*3+1] = (Math.random()-0.5) * 10;
        pos[i*3+2] = -20 + Math.random() * 40;
        vel[i] = 0.1 + Math.random() * 0.2;
    }
    
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({
        color: 0x60a5fa,
        size: 0.05,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending
    });
    
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

    // Smoother Explode Transition
    const explodeMult = isExploded ? 3.0 : 1.0;
    sceneObjects.layers.forEach((layer, i) => {
        layer.group.position.z += (layer.baseZ * explodeMult - layer.group.position.z) * 0.05;
        if (layer.isEmissive) {
            const pulse = 0.5 + Math.sin(t * 3 + i) * 0.5;
            layer.mat.emissiveIntensity = pulse * (1 + trainingProgress * 2);
            layer.wire.material.opacity = 0.2 + pulse * 0.5;
        }
    });

    // Pro Particle Animation
    if (sceneObjects.dataParticles) {
        const {pos, vel} = sceneObjects.dataParticles;
        for(let i=0; i<pos.length/3; i++) {
            pos[i*3+2] += vel[i] * (1 + trainingProgress * 5);
            if(pos[i*3+2] > 20) pos[i*3+2] = -20;
        }
        sceneObjects.dataParticles.points.geometry.attributes.position.needsUpdate = true;
    }

    updateAnnotations();
    composer.render();
}

// Reuse existing annotation/hover logic but update styling
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
            document.getElementById('tooltip-dim').textContent = arch ? `DIM: ${arch.w * 512}x${arch.h * 512}` : 'TYPE: RANK_DECOMPOSED';
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
    pickMethod('lora');
    animate();
}

document.addEventListener('configReady', initExplore);
