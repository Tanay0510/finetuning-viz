let state = {
  method: 'lora', model: '7b', gpu: 'a100_80', gpu_count: 1, sharding: 'none',
  rank: 16, batch: 4, seq: 2048, precision: 'bf16', optimizer: 'paged_adamw_8bit', grad_checkpoint: true,
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
  
  document.getElementById('rank-val').textContent = state.rank;
  document.getElementById('batch-val').textContent = state.batch;
  document.getElementById('seq-val').textContent = state.seq;
  document.getElementById('gpu-count-val').textContent = state.gpu_count;
  
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
  updateUI(totalVRAM, { weightsGB, gradsGB, optimGB, actGB }, { trainableParams, totalParams }, gpu);
}

function updateUI(totalVRAM, mem, params, gpu) {
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

    const resVRAM = document.getElementById('res-vram');
    resVRAM.textContent = totalVRAM.toFixed(1) + ' GB';
    resVRAM.style.color = fits ? (tight ? '#f59e0b' : '#3B82F6') : '#ef4444';

    const fmtParams = params.trainableParams >= 1e9 ? (params.trainableParams/1e9).toFixed(1)+'B' : (params.trainableParams/1e6).toFixed(1)+'M';
    document.getElementById('res-params').textContent = fmtParams;
    
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

function initPlayground() {
    Object.keys(state).forEach(key => {
        const value = state[key];
        if (typeof value === 'boolean') {
            document.getElementById(`${key.replace('_', '-')}-btn`)?.classList.toggle('active', value);
        } else {
            document.querySelectorAll(`[data-id='${value}']`).forEach(b => {
                const parent = b.parentElement;
                if(parent && (parent.id.includes(key.replace('_', '-')) || parent.id === 'model-selector' || parent.id === 'gpu-selector')) {
                   b.classList.add('active');
                }
            });
        }
    });
    
    const currentMethod = METHODS.find(x => x.id === state.method);
    if (currentMethod) {
        document.getElementById('method-desc').textContent = currentMethod.desc;
    }
    updateCalc();
}

document.addEventListener('configReady', initPlayground);
