/**
 * Comparison Engine - llmviz.studio
 * Dual-viewport synchronized comparison.
 */

const engineState = {
    a: { scene: null, camera: null, renderer: null, methodId: 'full', modelId: '7b', method: null, model: null },
    b: { scene: null, camera: null, renderer: null, methodId: 'lora', modelId: '7b', method: null, model: null }
};

let camAngle = { th: 0.6, ph: 0.2, d: 45 };
let mouse = { dn: false, lx: 0, ly: 0 };

function initCompare() {
    initEngine('a');
    initEngine('b');
    
    // Initial builds
    pickMethod('a', 'full');
    pickMethod('b', 'lora');
    
    animate();
}

function initEngine(id) {
    const cvs = document.getElementById(`canvas-${id}`);
    if (!cvs) return;
    const container = cvs.parentElement;
    
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x020408, 0.015);
    
    const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ canvas: cvs, antialias: false, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    
    scene.add(new THREE.AmbientLight(0xffffff, 0.8));
    const dl = new THREE.DirectionalLight(0xffffff, 1); dl.position.set(10, 20, 10); scene.add(dl);

    engineState[id].scene = scene;
    engineState[id].camera = camera;
    engineState[id].renderer = renderer;
    
    cvs.addEventListener('mousedown', e => { mouse.dn = true; mouse.lx = e.clientX; mouse.ly = e.clientY; });
}

function pickMethod(id, methodId) {
    engineState[id].methodId = methodId;
    engineState[id].method = METHODS.find(m => m.id === methodId);
    rebuildViewport(id);
}

function pickModel(id, modelId) {
    engineState[id].modelId = modelId;
    engineState[id].model = MODELS.find(m => m.id === modelId);
    
    // Update UI buttons
    document.querySelectorAll(`[id^="model-${id}-"]`).forEach(b => b.classList.remove('border-brand-accent', 'text-brand-accent'));
    document.getElementById(`model-${id}-${modelId}`).classList.add('border-brand-accent', 'text-brand-accent');
    
    rebuildViewport(id);
}

function rebuildViewport(id) {
    const engine = engineState[id];
    const method = engine.method;
    if (!method) return;

    // Clear existing
    while(engine.scene.children.length > 2) {
        const o = engine.scene.children[2];
        if(o.geometry) o.geometry.dispose();
        engine.scene.remove(o);
    }

    const mc = new THREE.Color(method.color);
    const isFull = ['full', 'rlhf', 'galore'].includes(engine.methodId);
    const group = new THREE.Group();

    for(let z=0; z<5; z++) {
        const box = new THREE.Mesh(new THREE.BoxGeometry(8, 8, 0.5), new THREE.MeshBasicMaterial({color: mc, wireframe: true, transparent: true, opacity: 0.05}));
        box.position.z = (z-2) * 6;
        group.add(box);
        
        if (isFull) {
            const geo = new THREE.BoxGeometry(0.5, 0.5, 0.5);
            const mat = new THREE.MeshStandardMaterial({
                color: 0x1a1e2e,
                metalness: 0.9,
                roughness: 0.2,
                emissive: mc,
                emissiveIntensity: 0.5,
                transparent: true,
                opacity: 0.8
            });
            const imesh = new THREE.InstancedMesh(geo, mat, 16);
            const dummy = new THREE.Object3D();
            for(let i=0; i<16; i++) {
                dummy.position.set((i%4-2)*1.5, (Math.floor(i/4)-2)*1.5, (z-2)*6);
                dummy.updateMatrix();
                imesh.setMatrixAt(i, dummy.matrix);
            }
            group.add(imesh);
        }
    }

    if (['lora', 'qlora', 'dora'].includes(engine.methodId)) {
        const adapterGeo = new THREE.BoxGeometry(2, 8, 0.1);
        const adapterMat = new THREE.MeshStandardMaterial({
            color: mc,
            emissive: mc,
            emissiveIntensity: 2,
            metalness: 1,
            roughness: 0
        });
        for(let z=0; z<5; z++) {
            const adapter = new THREE.Mesh(adapterGeo, adapterMat);
            adapter.position.set(5, 0, (z-2)*6);
            group.add(adapter);
        }
    }

    engine.scene.add(group);
    updateMetrics();
}

function updateMetrics() {
    ['a', 'b'].forEach(id => {
        const engine = engineState[id];
        const m = engine.method;
        const mdl = engine.model || MODELS.find(x => x.id === '7b');
        
        if (!m || !mdl) return;

        // Simplified calculation for comparison
        let vram = mdl.params * 2; // Default FP16
        if (m.id === 'full') vram *= 4;
        else if (m.id === 'qlora') vram = mdl.params * 0.8;
        else vram *= 1.2;

        document.getElementById(`metric-${id}-vram`).textContent = `${vram.toFixed(1)} GB`;
        document.getElementById(`metric-${id}-params`).textContent = m.id === 'full' ? `${mdl.params}B` : '4.2M';
        document.getElementById(`metric-${id}-cost`).textContent = `$${(vram * 0.15).toFixed(2)}`;
    });
}

window.addEventListener('mousemove', e => {
    if(!mouse.dn) return;
    camAngle.th -= (e.clientX - mouse.lx) * 0.005;
    camAngle.ph = Math.max(-0.5, Math.min(0.8, camAngle.ph + (e.clientY - mouse.ly) * 0.005));
    mouse.lx = e.clientX; mouse.ly = e.clientY;
});
window.addEventListener('mouseup', () => { mouse.dn = false; });

function animate() {
    requestAnimationFrame(animate);
    ['a', 'b'].forEach(id => {
        const engine = engineState[id];
        if(!engine.camera) return;
        engine.camera.position.x = Math.sin(camAngle.th) * Math.cos(camAngle.ph) * camAngle.d;
        engine.camera.position.y = Math.sin(camAngle.ph) * camAngle.d + 2;
        engine.camera.position.z = Math.cos(camAngle.th) * Math.cos(camAngle.ph) * camAngle.d;
        engine.camera.lookAt(0, 0, 0);
        engine.renderer.render(engine.scene, engine.camera);
    });
}

document.addEventListener('configReady', initCompare);
