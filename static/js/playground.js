let state = {
  method: 'lora', model: '7b', gpu: 'a100_80', gpu_count: 1, sharding: 'none',
  rank: 16, batch: 4, seq: 2048, precision: 'bf16', optimizer: 'paged_adamw_8bit', grad_checkpoint: true,
  dataset_size: 1000, epochs: 3
};

function setState(key, value, selectorPrefix) {
    state[key] = value;
    document.querySelectorAll(`${selectorPrefix} button`).forEach(b => {
        b.classList.toggle('active', b.dataset.id === value);
    });
    if (key === 'method') {
      const m = METHODS.find(x => x.id === value);
      document.getElementById('method-desc').textContent = m.desc;
    }
    updateCalc();
}

function toggleState(key, btnId) {
    state[key] = !state[key];
    document.getElementById(btnId).classList.toggle('active', state[key]);
    updateCalc();
}

function updateCalc() {
  state.rank = parseInt(document.getElementById('rank-slider').value);
  state.batch = parseInt(document.getElementById('batch-slider').value);
  state.seq = parseInt(document.getElementById('seq-slider').value);
  state.gpu_count = parseInt(document.getElementById('gpu-count-slider').value);
  state.dataset_size = parseInt(document.getElementById('dataset-slider').value);
  state.epochs = parseInt(document.getElementById('epochs-slider').value);
  
  document.getElementById('rank-val').textContent = state.rank;
  document.getElementById('batch-val').textContent = state.batch;
  document.getElementById('seq-val').textContent = state.seq;
  document.getElementById('gpu-count-val').textContent = state.gpu_count;
  document.getElementById('dataset-val').textContent = state.dataset_size;
  document.getElementById('epochs-val').textContent = state.epochs;
  
  const loraControls = document.getElementById('lora-controls');
  if (loraControls) {
    loraControls.style.display = ['lora','qlora','dora'].includes(state.method) ? 'block' : 'none';
  }
  
  const m = METHODS.find(x => x.id === state.method);
  const gpu = GPUS.find(x => x.id === state.gpu);
  const mdl = MODELS.find(x => x.id === state.model);

  const P = mdl.params, D = mdl.d, L = mdl.layers, R = state.rank, B = state.batch, S = state.seq, N = state.gpu_count;
  const precisionBytes = {'bf16': 2, 'fp16': 2, 'fp32': 4}[state.precision];
  const optimizerBytes = state.optimizer === 'adamw_32bit' ? 8 : 1;

  let weightsGB, gradsGB, optimGB, actGB, trainableParams;
  const totalParams = P * 1e9;

  if (['full', 'rlhf', 'galore'].includes(state.method)) {
    weightsGB = P * precisionBytes;
    gradsGB = P * precisionBytes;
    optimGB = (state.method === 'galore') ? (P * 0.25) : (P * optimizerBytes);
    trainableParams = totalParams;
  } else {
    weightsGB = state.method === 'qlora' ? P * 0.5 : P * precisionBytes;
    const adapterParams = (4 * L * 2 * D * R);
    trainableParams = adapterParams;
    gradsGB = (adapterParams * precisionBytes) / 1e9;
    optimGB = (adapterParams * optimizerBytes) / 1e9;
  }
  
  actGB = (B * S * D * L * precisionBytes) / 1e9;
  if(state.grad_checkpoint) actGB *= 0.55;
  actGB = Math.min(actGB, 25);

  if (N > 1) {
    if (state.sharding === 'zero1') optimGB /= N;
    else if (state.sharding === 'zero2') { optimGB /= N; gradsGB /= N; }
    else if (['zero3', 'fsdp'].includes(state.sharding)) { optimGB /= N; gradsGB /= N; weightsGB /= N; }
  }

  const totalVRAM = weightsGB + gradsGB + optimGB + actGB;

  // --- NEW: Logistics Math ---
  const totalTokens = state.dataset_size * state.seq * state.epochs;
  const tflopsNeeded = (totalParams * totalTokens * 6) / 1e12;
  const clusterTflops = gpu.tflops * N * 0.5; // 0.5 is efficiency (MFU)
  
  const totalSeconds = tflopsNeeded / clusterTflops;
  const totalHours = totalSeconds / 3600;
  const totalCost = totalHours * gpu.cost * N;

  updateUI(totalVRAM, { weightsGB, gradsGB, optimGB, actGB }, { trainableParams, totalParams }, gpu, { totalHours, totalCost });
}

function updateUI(totalVRAM, mem, params, gpu, logistics) {
    const gpuVRAM = gpu.vram;
    const fits = totalVRAM <= gpuVRAM;
    const tight = totalVRAM > gpuVRAM * 0.85 && fits;

    document.getElementById('vram-used').textContent = totalVRAM.toFixed(1);
    document.getElementById('vram-total').textContent = gpuVRAM;
    
    const maxVal = Math.max(totalVRAM, gpuVRAM) * 1.1;
    document.getElementById('vram-limit-line').style.left = `${(gpuVRAM / maxVal) * 100}%`;
    document.getElementById('seg-weights').style.width = `${(mem.weightsGB / maxVal) * 100}%`;
    document.getElementById('seg-grads').style.width = `${(mem.gradsGB / maxVal) * 100}%`;
    document.getElementById('seg-optim').style.width = `${(mem.optimGB / maxVal) * 100}%`;
    document.getElementById('seg-act').style.width = `${(mem.actGB / maxVal) * 100}%`;

    // Populate Breakdown
    const breakdown = document.getElementById('vram-breakdown');
    if (breakdown) {
        const items = [
            { label: 'Frozen Weights', val: mem.weightsGB, col: '#3B82F6' },
            { label: 'Gradients', val: mem.gradsGB, col: '#8B5CF6' },
            { label: 'Optimizer States', val: mem.optimGB, col: '#EC4899' },
            { label: 'Activations', val: mem.actGB, col: '#F59E0B' }
        ];
        breakdown.innerHTML = items.map(i => `
            <div class="flex justify-between items-center font-mono text-[10px]">
                <div class="flex items-center gap-2">
                    <span class="w-1.5 h-1.5 rounded-full" style="background: ${i.col}"></span>
                    <span class="text-brand-text2 uppercase">${i.label}</span>
                </div>
                <span class="text-white font-bold">${i.val.toFixed(2)} GB</span>
            </div>
        `).join('');
    }

    const resVRAM = document.getElementById('res-vram');
    resVRAM.textContent = totalVRAM.toFixed(1) + ' GB';
    resVRAM.style.color = fits ? (tight ? '#f59e0b' : '#3B82F6') : '#ef4444';

    const fmtParams = params.trainableParams >= 1e9 ? (params.trainableParams/1e9).toFixed(1)+'B' : (params.trainableParams/1e6).toFixed(1)+'M';
    document.getElementById('res-params').textContent = fmtParams;

    // Logistics Output
    const timeEl = document.getElementById('res-time');
    const costEl = document.getElementById('res-cost');
    if (timeEl && costEl) {
        const h = Math.floor(logistics.totalHours);
        const m = Math.round((logistics.totalHours - h) * 60);
        timeEl.textContent = `${h}h ${m}m`;
        costEl.textContent = `$${logistics.totalCost.toFixed(2)}`;
    }
    
    const v = document.getElementById('verdict');
    if (fits && !tight) {
      v.className = 'p-8 rounded-3xl border bg-green-500/5 border-green-500/20 text-green-400 text-sm shadow-[0_0_20px_rgba(34,197,94,0.1)]';
      v.innerHTML = `<span class="text-xl mr-2">✓</span> Configuration fits comfortably within target VRAM.`;
    } else if (tight) {
      v.className = 'p-8 rounded-3xl border bg-yellow-500/5 border-yellow-500/20 text-yellow-400 text-sm shadow-[0_0_20px_rgba(234,179,8,0.1)]';
      v.innerHTML = `<span class="text-xl mr-2">⚠</span> Memory usage is critical ( > 85% ). Expect potential OOM on long sequences.`;
    } else {
      v.className = 'p-8 rounded-3xl border bg-red-500/5 border-red-500/20 text-red-400 text-sm shadow-[0_0_20px_rgba(239,68,68,0.1)]';
      v.innerHTML = `<span class="text-xl mr-2">✗</span> Hardware allocation insufficient. Reduce rank or increase GPU nodes.`;
    }
}

// Recipe Framework
let currentRecipeFramework = 'axolotl';
function openRecipeModal() {
  document.getElementById('recipe-modal').classList.remove('hidden');
  switchRecipe('axolotl');
}
function closeRecipeModal() { document.getElementById('recipe-modal').classList.add('hidden'); }

function switchRecipe(framework) {
  currentRecipeFramework = framework;
  document.querySelectorAll('.recipe-tab').forEach(b => b.classList.toggle('active', b.id === `tab-${framework}`));
  document.getElementById('recipe-content').textContent = generateRecipe(framework);
}

function generateRecipe(fw) {
  const m = METHODS.find(x => x.id === state.method);
  return `base_model: meta-llama/Llama-2-7b-hf\nmethod: ${state.method.toUpperCase()}\nrank: ${state.rank}\nbatch_size: ${state.batch}\nseq_len: ${state.seq}\nprecision: ${state.precision}\noptimizer: ${state.optimizer}`;
}

function copyRecipe() {
  navigator.clipboard.writeText(document.getElementById('recipe-content').textContent);
}

function downloadRecipe() {
  const blob = new Blob([document.getElementById('recipe-content').textContent], { type: 'text/yaml' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `training_recipe.yaml`;
  a.click();
}

async function downloadDevOpsBundle() {
    // Generate Files
    const trainPy = `
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Config from llmviz.studio
model_id = "meta-llama/Llama-2-7b-hf"
peft_config = LoraConfig(
    r=${state.rank},
    lora_alpha=${state.rank * 2},
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model = get_peft_model(model, peft_config)

print(f"Training started: rank=${state.rank}, batch=${state.batch}")
# Add your training loop here
    `.trim();

    const dockerfile = `
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "train.py"]
    `.trim();

    const reqs = `
torch
transformers
peft
accelerate
bitsandbytes
    `.trim();

    // Use JSZip (assumes script loaded in base.html)
    const zip = new JSZip();
    zip.file("train.py", trainPy);
    zip.file("Dockerfile", dockerfile);
    zip.file("requirements.txt", reqs);
    zip.file("recipe.yaml", document.getElementById('recipe-content').textContent);

    const content = await zip.generateAsync({type:"blob"});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(content);
    a.download = "llmviz_devops_bundle.zip";
    a.click();
}

async function saveRecipe() {
    const name = prompt("Enter a name for this recipe:", "My Llama-2-7b LoRA");
    if (!name) return;
    
    const description = prompt("Enter a brief description (optional):", "Optimized for consumer GPUs.");
    const isPublic = confirm("Would you like to share this recipe with the community Zoo?");

    try {
        const response = await fetch('/api/recipes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                description: description,
                config: state,
                is_public: isPublic
            })
        });
        const result = await response.json();
        alert(isPublic ? "Recipe saved and shared to the Zoo!" : "Recipe saved to your profile.");
    } catch (err) {
        console.error(err);
        alert("Failed to save recipe.");
    }
}

async function loadClonedRecipe() {
    const params = new URLSearchParams(window.location.search);
    const cloneId = params.get('clone');
    if (!cloneId) return;

    try {
        const response = await fetch(`/api/recipes/${cloneId}`);
        const recipe = await response.json();
        if (recipe.config) {
            state = { ...state, ...recipe.config };
            // Refresh UI components
            initPlaygroundUI();
            updateCalc();
            alert(`Loaded community recipe: ${recipe.name}`);
        }
    } catch (err) {
        console.error("Clone failed", err);
    }
}

// Split UI initialization for reuse
function initPlaygroundUI() {
    Object.keys(state).forEach(key => {
        const value = state[key];
        if (typeof value === 'boolean') {
            const btn = document.getElementById(`${key.replace('_', '-')}-btn`);
            if (btn) btn.classList.toggle('active', value);
        } else {
            // Remove active from all siblings
            const activeBtns = document.querySelectorAll(`[data-id='${value}']`);
            activeBtns.forEach(b => {
                const parent = b.parentElement;
                if(parent && (parent.id.includes(key.replace('_', '-')) || parent.id === 'model-selector' || parent.id === 'gpu-selector')) {
                   // Clear siblings
                   Array.from(parent.children).forEach(s => s.classList.remove('active'));
                   b.classList.add('active');
                }
            });
        }
    });
    
    const currentMethod = METHODS.find(x => x.id === state.method);
    if (currentMethod) {
        document.getElementById('method-desc').textContent = currentMethod.desc;
    }
}

let simActive = false;
let simData = [];
let simInterval;

function toggleSimulation() {
    const btn = document.getElementById('sim-btn');
    const overlay = document.getElementById('sim-overlay');
    
    if (simActive) {
        simActive = false;
        clearInterval(simInterval);
        btn.innerHTML = '▶ Run Simulation';
        btn.classList.replace('bg-red-500', 'bg-brand-accent');
    } else {
        simActive = true;
        simData = [];
        overlay.style.opacity = '0';
        overlay.style.pointerEvents = 'none';
        btn.innerHTML = '⏹ Stop Test';
        btn.classList.replace('bg-brand-accent', 'bg-red-500');
        startSimulation();
    }
}

function startSimulation() {
    let loss = 4.5;
    let step = 0;
    const maxSteps = 100;
    
    // Heuristics based on state
    const learningRate = state.rank > 64 ? 0.15 : 0.08; 
    const noiseLevel = Math.max(0.02, 0.2 / state.batch);
    
    simInterval = setInterval(() => {
        if (step >= maxSteps) {
            clearInterval(simInterval);
            document.getElementById('sim-status').textContent = 'CONVERGED';
            return;
        }

        // Simulating exponential decay + noise
        const decay = Math.exp(-step * learningRate);
        const currentLoss = (loss * decay) + (Math.random() * noiseLevel);
        simData.push(currentLoss);
        
        updateSimUI(currentLoss, step);
        drawLossCurve();
        step++;
    }, 100);
}

function updateSimUI(loss, step) {
    document.getElementById('sim-loss').textContent = loss.toFixed(4);
    const statusEl = document.getElementById('sim-status');
    if (loss > 2.0) {
        statusEl.textContent = 'LEARNING';
        statusEl.className = 'text-sm font-mono text-blue-400';
    } else if (loss > 0.5) {
        statusEl.textContent = 'STABLE';
        statusEl.className = 'text-sm font-mono text-green-400';
    } else {
        statusEl.textContent = 'FINALIZING';
        statusEl.className = 'text-sm font-mono text-brand-accent';
    }
}

function drawLossCurve() {
    const canvas = document.getElementById('loss-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    // Set internal resolution if needed
    if (canvas.width !== canvas.offsetWidth) {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (simData.length < 2) return;

    ctx.beginPath();
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';

    const padding = 20;
    const graphWidth = canvas.width - padding * 2;
    const graphHeight = canvas.height - padding * 2;
    
    simData.forEach((val, i) => {
        const x = padding + (i / 100) * graphWidth;
        const y = padding + (1 - (val / 5)) * graphHeight; // Assuming loss starts at 5
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    
    ctx.stroke();

    // Fill area under curve
    ctx.lineTo(padding + ((simData.length-1) / 100) * graphWidth, canvas.height - padding);
    ctx.lineTo(padding, canvas.height - padding);
    const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
    grad.addColorStop(0, 'rgba(59, 130, 246, 0.2)');
    grad.addColorStop(1, 'rgba(59, 130, 246, 0)');
    ctx.fillStyle = grad;
    ctx.fill();
}

function initPlayground() {
    initPlaygroundUI();
    updateCalc();
    loadClonedRecipe();
}

document.addEventListener('configReady', initPlayground);
