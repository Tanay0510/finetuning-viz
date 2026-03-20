let scene, camera, renderer, dead = false;
let camAngle = {th: 0, ph: 0.1, d: 25};
let mouse = {dn:false, lx:0, ly:0};
let tokens = [];
let attentionLines = [];
let activeTokenIndex = -1;

function initThree() {
  const cvs = document.getElementById('three-canvas');
  if (!cvs) return;
  const container = cvs.parentElement;
  
  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x020408, 10, 50);

  camera = new THREE.PerspectiveCamera(40, container.clientWidth/container.clientHeight, 0.1, 1000);
  
  renderer = new THREE.WebGLRenderer({canvas:cvs, antialias:true, alpha: true});
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dl = new THREE.DirectionalLight(0xffffff, 1); dl.position.set(5, 10, 5); scene.add(dl);

  cvs.addEventListener('mousedown', e=>{mouse={dn:true,lx:e.clientX,ly:e.clientY}});
  window.addEventListener('mousemove', e=>{
    if(!mouse.dn) return;
    camAngle.th-=(e.clientX-mouse.lx)*0.005;
    camAngle.ph=Math.max(-0.5,Math.min(0.8,camAngle.ph+(e.clientY-mouse.ly)*0.005));
    mouse.lx=e.clientX; mouse.ly=e.clientY;
  });
  window.addEventListener('mouseup', ()=>{mouse.dn=false});

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
}

function visualizeAttention() {
    const inputEl = document.getElementById('attention-input');
    if (!inputEl) return;
    const text = inputEl.value;
    const words = text.split(' ');
    const container = document.getElementById('token-container');
    if (!container) return;
    
    // Clear
    container.innerHTML = '';
    tokens = [];
    attentionLines.forEach(l => scene.remove(l));
    attentionLines = [];

    const spacing = 3;
    const startX = -(words.length - 1) * spacing / 2;

    words.forEach((word, i) => {
        const pos = new THREE.Vector3(startX + i * spacing, 0, 0);
        
        // Visual Node in 3D (invisible, just for reference)
        const node = new THREE.Object3D();
        node.position.copy(pos);
        scene.add(node);

        // UI Element
        const el = document.createElement('div');
        el.className = 'token-node pointer-events-auto';
        el.textContent = word;
        el.onmouseenter = () => highlightAttention(i);
        el.onmouseleave = () => resetAttention();
        container.appendChild(el);

        tokens.push({ el, node, pos });
    });
}

function highlightAttention(index) {
    activeTokenIndex = index;
    tokens.forEach((t, i) => {
        t.el.classList.toggle('active', i === index);
    });

    // Remove old lines
    attentionLines.forEach(l => scene.remove(l));
    attentionLines = [];

    const source = tokens[index];
    
    tokens.forEach((target, i) => {
        if (i === index) return;

        // Simulate some fake attention weights (closer words or random)
        const weight = Math.random() * 0.8 + 0.1;
        
        const curve = new THREE.QuadraticBezierCurve3(
            source.pos,
            new THREE.Vector3((source.pos.x + target.pos.x)/2, 4 * weight, (source.pos.z + target.pos.z)/2 + 2),
            target.pos
        );

        const points = curve.getPoints(20);
        const geo = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({
            color: 0x3B82F6,
            transparent: true,
            opacity: weight * 0.6,
            linewidth: weight * 5
        });

        const line = new THREE.Line(geo, mat);
        scene.add(line);
        attentionLines.push(line);
    });
}

function resetAttention() {
    activeTokenIndex = -1;
    tokens.forEach(t => t.el.classList.remove('active'));
    attentionLines.forEach(l => scene.remove(l));
    attentionLines = [];
}

function animate() {
    if (dead) return;
    requestAnimationFrame(animate);
    
    if(!mouse.dn) camAngle.th += 0.001;
    camera.position.x = Math.sin(camAngle.th)*Math.cos(camAngle.ph)*camAngle.d;
    camera.position.y = Math.sin(camAngle.ph)*camAngle.d + 2;
    camera.position.z = Math.cos(camAngle.th)*Math.cos(camAngle.ph)*camAngle.d;
    camera.lookAt(0,0,0);

    // Update projections
    tokens.forEach(t => {
        const p = t.pos.clone().project(camera);
        if (p.z > 1) {
            t.el.style.display = 'none';
        } else {
            t.el.style.display = 'block';
            const x = (p.x + 1) * renderer.domElement.clientWidth / 2;
            const y = (-p.y + 1) * renderer.domElement.clientHeight / 2;
            t.el.style.transform = `translate(-50%, -50%) translate(${x}px, ${y}px)`;
        }
    });

    renderer.render(scene, camera);
}

function initAttention() {
    initThree();
    visualizeAttention();
    animate();
}

document.addEventListener('configReady', initAttention);
