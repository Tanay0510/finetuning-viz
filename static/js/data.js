let currentMicroModel = 'llama2';
const colors = ["#3B82F6", "#8B5CF6", "#EC4899", "#F59E0B", "#10B981", "#60A5FA"];

function setMicroscopeModel(model) {
    currentMicroModel = model;
    document.querySelectorAll('[id^="micro-"]').forEach(b => b.classList.remove('active', 'border-brand-accent', 'text-brand-accent'));
    const btn = document.getElementById(`micro-${model}`);
    btn.classList.add('active', 'border-brand-accent', 'text-brand-accent');
    tokenize();
}

function tokenize() {
  const container = document.getElementById('token-stream');
  const input = document.getElementById('micro-input').value;
  if (!container || !input) return;
  
  container.innerHTML = '';
  
  let tokens = [];
  if (currentMicroModel === 'llama2') {
      // Simulation: Llama-2 is aggressive, often splitting words
      tokens = input.split(' ').flatMap(w => w.length > 6 ? [w.substring(0, 4), w.substring(4)] : [w]);
  } else if (currentMicroModel === 'gpt4') {
      // Simulation: GPT-4 has a huge vocab, efficient
      tokens = input.split(' ');
  } else {
      // Mistral: Middle ground
      tokens = input.match(/\b(\w+)\b/g) || [];
  }

  tokens.forEach((token, i) => {
      const chip = document.createElement('span');
      chip.className = 'token-chip animate-token';
      const color = colors[i % colors.length];
      chip.style.backgroundColor = `${color}15`;
      chip.style.color = color;
      chip.style.borderColor = `${color}30`;
      chip.textContent = token;
      container.appendChild(chip);
  });

  const words = input.trim().split(/\s+/).length;
  const efficiency = (tokens.length / words).toFixed(2);
  document.getElementById('micro-efficiency').textContent = `${efficiency} tokens/word`;
}

function startPacking() {
  const packWindow = document.getElementById('pack-window');
  const efficiency = document.getElementById('pack-efficiency');
  if (!packWindow || !efficiency) return;
  packWindow.innerHTML = '';
  
  let currentWidth = 0;
  const samples = [
    { w: 25, col: '#3B82F6', label: 'Sample A' },
    { w: 15, col: '#8B5CF6', label: 'Sample B' },
    { w: 35, col: '#EC4899', label: 'Sample C' },
    { w: 20, col: '#F59E0B', label: 'Sample D' }
  ];

  samples.forEach((s, i) => {
    setTimeout(() => {
      const block = document.createElement('div');
      block.className = 'h-12 rounded-lg border flex flex-col items-center justify-center animate-token relative group';
      block.style.width = `${s.w}%`;
      block.style.backgroundColor = `${s.col}20`;
      block.style.borderColor = `${s.col}50`;
      
      block.innerHTML = `
        <span class="text-[8px] font-bold text-white/80">${s.label}</span>
        <div class="absolute right-0 inset-y-0 w-px bg-white/40 group-hover:bg-white shadow-[0_0_10px_white]"></div>
      `;
      
      packWindow.appendChild(block);
      currentWidth += s.w;
      efficiency.textContent = `UTILIZATION: ${currentWidth}%`;
      if (currentWidth >= 95) efficiency.className = 'text-xs font-bold text-green-400 font-mono animate-pulse';
    }, i * 600);
  });
}

function initData() {
    tokenize();
}

document.addEventListener('configReady', initData);
