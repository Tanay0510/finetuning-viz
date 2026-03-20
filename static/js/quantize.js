/**
 * Quantization Laboratory - llmviz.studio
 * Simulates bit-depth reduction on model weights.
 */

let canvas, ctx;
let bits = 16;
let animationId;

const THEORIES = {
    32: "32-bit (FP32) is the original 'Full Precision'. It uses 4 bytes per parameter. Perfect signal, but extremely memory intensive.",
    16: "16-bit (FP16/BF16) is the industry standard. It cuts memory in half with almost zero loss in model intelligence.",
    8: "8-bit (INT8) compression. 4x smaller than original. Noticeable 'stepping' in the signal, but LLMs handle this surprisingly well.",
    4: "4-bit (NF4) is the 'Magic' threshold. It allows 70B models to run on single GPUs. The signal is jagged, but the brain of the LLM remains intact.",
    2: "2-bit quantization is the 'Edge of Collapse'. The signal is heavily distorted. Only the most robust models can still function here."
};

function initQuantize() {
    canvas = document.getElementById('quant-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    
    // Set internal resolution
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    
    updateQuantization();
    animate();
}

function updateQuantization() {
    bits = parseInt(document.getElementById('bit-slider').value);
    document.getElementById('bit-val').textContent = `${bits}-bit`;
    
    // Update Method Label
    const methodEl = document.getElementById('quant-method');
    if (bits >= 16) methodEl.textContent = bits === 32 ? "FP32 (Full)" : "FP16 (Half)";
    else if (bits >= 8) methodEl.textContent = "INT8 (Integer)";
    else if (bits >= 4) methodEl.textContent = "NF4 (Normal)";
    else methodEl.textContent = "Binary/Extreme";

    // Update Compression
    const compression = (32 / bits).toFixed(1);
    document.getElementById('quant-comp').textContent = `${compression}x`;

    // Update Theory
    const theoryKey = [32, 16, 8, 4, 2].reduce((prev, curr) => Math.abs(curr - bits) < Math.abs(prev - bits) ? curr : prev);
    document.getElementById('quant-theory').textContent = THEORIES[theoryKey];

    // Update Bar
    const barWidth = (bits / 32) * 100;
    document.getElementById('quant-bar').style.width = `${barWidth}%`;
}

function drawSignal() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const centerY = canvas.height / 2;
    const width = canvas.width;
    const amplitude = canvas.height * 0.3;
    const frequency = 0.005;
    const time = Date.now() * 0.002;

    // Draw Original Signal (Faint)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(136, 146, 168, 0.1)';
    ctx.lineWidth = 2;
    for (let x = 0; x < width; x++) {
        const y = centerY + Math.sin(x * frequency + time) * amplitude;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Quantized Signal
    const levels = Math.pow(2, bits);
    ctx.beginPath();
    ctx.strokeStyle = '#3B82F6';
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';

    for (let x = 0; x < width; x++) {
        // Raw value (-1 to 1)
        const rawY = Math.sin(x * frequency + time);
        
        // Quantize
        // We simulate the 'stepping' effect
        let quantizedY;
        if (bits >= 16) {
            quantizedY = rawY; // Effectively smooth
        } else {
            // Map -1..1 to 0..levels
            const step = Math.round(((rawY + 1) / 2) * (levels - 1));
            // Map back to -1..1
            quantizedY = (step / (levels - 1)) * 2 - 1;
        }

        const screenY = centerY + quantizedY * amplitude;
        if (x === 0) ctx.moveTo(x, screenY);
        else ctx.lineTo(x, screenY);
    }
    ctx.stroke();

    // Draw Quantization Points (for low bits)
    if (bits <= 6) {
        ctx.fillStyle = '#60A5FA';
        for (let x = 0; x < width; x += 40) {
            const rawY = Math.sin(x * frequency + time);
            const step = Math.round(((rawY + 1) / 2) * (levels - 1));
            const quantizedY = (step / (levels - 1)) * 2 - 1;
            const screenY = centerY + quantizedY * amplitude;
            
            ctx.beginPath();
            ctx.arc(x, screenY, 4, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

function animate() {
    drawSignal();
    animationId = requestAnimationFrame(animate);
}

// Ensure init happens on configReady
document.addEventListener('configReady', initQuantize);
