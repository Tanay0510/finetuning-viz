const words = ["Fine-tuning", "is", "the", "process", "of", "taking", "a", "pre-trained", "model", "and", "adapting", "it", "to", "a", "specific", "task", "or", "dataset.", "Sample", "packing", "makes", "this", "significantly", "more", "efficient", "by", "concatenating", "examples."];
const colors = ["#3B82F6", "#8B5CF6", "#EC4899", "#F59E0B", "#10B981", "#60A5FA"];

function tokenize() {
  const container = document.getElementById('token-stream');
  if (!container) return;
  container.innerHTML = '';
  
  words.forEach((word, i) => {
    setTimeout(() => {
      const chip = document.createElement('span');
      chip.className = 'token-chip animate-token';
      const color = colors[i % colors.length];
      chip.style.backgroundColor = `${color}15`;
      chip.style.color = color;
      chip.style.borderColor = `${color}30`;
      chip.textContent = word;
      container.appendChild(chip);
    }, i * 40);
  });
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
