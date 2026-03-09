"""
LLM Fine-Tuning Visual Studio — llmviz.studio
"""
from flask import Flask, render_template, jsonify

app = Flask(__name__)

METHODS = [
    {
        "id": "full", "name": "Full Fine-Tuning", "short": "Full FT", "emoji": "🔥",
        "color": "#FF4060", "accent": "#FF8095",
        "tagline": "Maximum power. Maximum cost.",
        "desc": "Updates every parameter via standard backpropagation. Adam maintains momentum+variance for all billions of parameters, requiring ~4x model size in memory.",
        "formula": "θ' = θ − α · ∇L(θ)",
        "insight": "For a 70B model in BF16: 140GB weights + 140GB gradients + 280GB optimizer states = ~560GB minimum VRAM.",
        "stats": {"params": "100%", "vram_7b": "~60GB", "vram_70b": "~560GB", "checkpoint": "Full model", "speed": "Slow"},
        "paper": {"title": "Standard backpropagation", "url": None, "year": None},
        "code": "from transformers import AutoModelForCausalLM, Trainer, TrainingArguments\n\nmodel = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\n\ntrainer = Trainer(\n    model=model,\n    args=TrainingArguments(\n        output_dir=\"./full-ft\",\n        per_device_train_batch_size=1,\n        gradient_accumulation_steps=16,\n        learning_rate=2e-5,\n        bf16=True,\n    ),\n    train_dataset=dataset,\n)\ntrainer.train()",
    },
    {
        "id": "lora", "name": "LoRA", "short": "LoRA", "emoji": "🔌",
        "color": "#3B82F6", "accent": "#93C5FD",
        "tagline": "Low-rank adapters. 1000x smaller checkpoints.",
        "desc": "Freezes all pretrained weights, injects trainable low-rank matrices A and B into attention layers. W' = W₀ + BA where rank r << d.",
        "formula": "W' = W₀ + BA, B∈ℝ^{d×r}, A∈ℝ^{r×d}",
        "insight": "With r=16 on d=4096, each adapter pair is 131K params vs 16.7M original — 0.8% of parameters, ~95% of full FT quality.",
        "stats": {"params": "~0.1-2%", "vram_7b": "~16GB", "vram_70b": "~80GB", "checkpoint": "10-50MB", "speed": "Fast"},
        "paper": {"title": "LoRA: Low-Rank Adaptation of LLMs", "url": "https://arxiv.org/abs/2106.09685", "year": 2021},
        "code": "from peft import LoraConfig, get_peft_model\nfrom transformers import AutoModelForCausalLM\n\nmodel = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\n\nconfig = LoraConfig(\n    r=16,\n    lora_alpha=32,\n    target_modules=[\"q_proj\",\"k_proj\",\"v_proj\",\"o_proj\"],\n    lora_dropout=0.05,\n    task_type=\"CAUSAL_LM\",\n)\nmodel = get_peft_model(model, config)\nmodel.print_trainable_parameters()\n# trainable: 4,194,304 || total: 6,742,609,920 || 0.062%",
    },
    {
        "id": "qlora", "name": "QLoRA", "short": "QLoRA", "emoji": "📦",
        "color": "#10B981", "accent": "#6EE7B7",
        "tagline": "4-bit base + full-precision adapters. 65B on one GPU.",
        "desc": "NF4 quantization of frozen weights + LoRA adapters in BF16. Paged optimizers swap states to CPU on memory pressure.",
        "formula": "W_NF4 + BA, paged AdamW",
        "insight": "65B model = ~33GB in NF4 + ~50MB adapters. Fine-tune on a single 48GB GPU. Quality gap vs LoRA: typically <1 point.",
        "stats": {"params": "~0.1-2%", "vram_7b": "~6GB", "vram_70b": "~36GB", "checkpoint": "10-50MB", "speed": "Moderate"},
        "paper": {"title": "QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/abs/2305.14314", "year": 2023},
        "code": "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\nfrom peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n\nbnb_config = BitsAndBytesConfig(\n    load_in_4bit=True,\n    bnb_4bit_quant_type=\"nf4\",\n    bnb_4bit_compute_dtype=\"bfloat16\",\n    bnb_4bit_use_double_quant=True,\n)\nmodel = AutoModelForCausalLM.from_pretrained(\n    \"meta-llama/Llama-2-70b-hf\",\n    quantization_config=bnb_config,\n    device_map=\"auto\",\n)\nmodel = prepare_model_for_kbit_training(model)\nmodel = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=\"all-linear\"))",
    },
    {
        "id": "prefix", "name": "Prefix Tuning", "short": "Prefix", "emoji": "📎",
        "color": "#F59E0B", "accent": "#FCD34D",
        "tagline": "Learned soft prompts steering frozen attention.",
        "desc": "Prepends trainable continuous vectors to K and V at every layer. All model weights frozen. Like a learned system prompt in continuous space.",
        "formula": "P = [p₁,...,pₘ] → K, V at each layer",
        "insight": "m=20 prefix tokens × 32 layers × 2(K,V) × 4096 dim = 5.2M params. That's 0.075% of a 7B model. Swap prefixes to swap tasks.",
        "stats": {"params": "~0.05-0.1%", "vram_7b": "~14GB", "vram_70b": "~140GB", "checkpoint": "~1MB", "speed": "Fast"},
        "paper": {"title": "Prefix-Tuning: Optimizing Continuous Prompts", "url": "https://arxiv.org/abs/2101.00190", "year": 2021},
        "code": "from peft import PrefixTuningConfig, get_peft_model\nfrom transformers import AutoModelForCausalLM\n\nmodel = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\nconfig = PrefixTuningConfig(\n    task_type=\"CAUSAL_LM\",\n    num_virtual_tokens=20,\n    prefix_projection=True,\n)\nmodel = get_peft_model(model, config)\nmodel.print_trainable_parameters()\n# trainable: 9,872,384 || 0.146%",
    },
    {
        "id": "adapters", "name": "Adapter Layers", "short": "Adapters", "emoji": "🧩",
        "color": "#8B5CF6", "accent": "#C4B5FD",
        "tagline": "Bottleneck modules between frozen layers.",
        "desc": "Inserts down-project → nonlinearity → up-project + residual modules after each sublayer. Starts as near-identity, gradually learns task-specific changes.",
        "formula": "h ← h + f(h · W_down) · W_up",
        "insight": "Bottleneck m=64 on d=4096: each adapter = 524K params. 2 per layer × 32 layers = 33.5M trainable. Composable via AdapterFusion.",
        "stats": {"params": "~0.5-3%", "vram_7b": "~18GB", "vram_70b": "~160GB", "checkpoint": "~30MB", "speed": "Moderate"},
        "paper": {"title": "Parameter-Efficient Transfer Learning for NLP", "url": "https://arxiv.org/abs/1902.00751", "year": 2019},
        "code": "from adapters import AutoAdapterModel\n\nmodel = AutoAdapterModel.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\nmodel.add_adapter(\"my_task\", config=\"seq_bn\")  # bottleneck adapter\nmodel.train_adapter(\"my_task\")  # freeze base, train adapter only\n\ntrainable = sum(p.numel() for p in model.parameters() if p.requires_grad)\nprint(f\"Trainable: {trainable:,}\")",
    },
    {
        "id": "rlhf", "name": "RLHF / DPO", "short": "RLHF/DPO", "emoji": "👍",
        "color": "#EC4899", "accent": "#F9A8D4",
        "tagline": "Human preferences sculpt model behavior.",
        "desc": "Aligns outputs with human preferences. RLHF: reward model + PPO. DPO: directly optimize on preference pairs, no reward model needed.",
        "formula": "L = −log σ(β·[log π/π_ref(y_w) − log π/π_ref(y_l)])",
        "insight": "DPO eliminates the reward model — the optimal reward is implicitly defined by the policy/reference log-ratio. Simpler, more stable, less memory.",
        "stats": {"params": "100%", "vram_7b": "~60GB (DPO)", "vram_70b": "~400GB+", "checkpoint": "Full model", "speed": "Very slow"},
        "paper": {"title": "DPO: Your LM is Secretly a Reward Model", "url": "https://arxiv.org/abs/2305.18290", "year": 2023},
        "code": "from trl import DPOTrainer, DPOConfig\nfrom transformers import AutoModelForCausalLM, AutoTokenizer\n\nmodel = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\nref_model = AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\ntokenizer = AutoTokenizer.from_pretrained(\"meta-llama/Llama-2-7b-hf\")\n\ntrainer = DPOTrainer(\n    model=model,\n    ref_model=ref_model,\n    args=DPOConfig(beta=0.1, learning_rate=5e-7, bf16=True),\n    train_dataset=preference_data,  # needs: prompt, chosen, rejected\n    tokenizer=tokenizer,\n)\ntrainer.train()",
    },
]

GPUS = [
    {"id": "rtx3090", "name": "RTX 3090", "vram": 24, "tflops": 35.6, "cost": 0, "tier": "Consumer"},
    {"id": "rtx4090", "name": "RTX 4090", "vram": 24, "tflops": 82.6, "cost": 0, "tier": "Consumer"},
    {"id": "a6000",   "name": "A6000",     "vram": 48, "tflops": 38.7, "cost": 0.80, "tier": "Pro"},
    {"id": "a100_40", "name": "A100 40GB", "vram": 40, "tflops": 312,  "cost": 1.10, "tier": "Cloud"},
    {"id": "a100_80", "name": "A100 80GB", "vram": 80, "tflops": 312,  "cost": 1.60, "tier": "Cloud"},
    {"id": "h100",    "name": "H100 80GB", "vram": 80, "tflops": 989,  "cost": 2.50, "tier": "Cloud"},
    {"id": "4xa100",  "name": "4×A100 80G","vram": 320,"tflops": 1248, "cost": 6.40, "tier": "Multi"},
    {"id": "8xh100",  "name": "8×H100",   "vram": 640,"tflops": 7912, "cost": 20.0, "tier": "Multi"},
]

MODEL_SIZES = [
    {"id": "1b",  "name": "1.3B", "params": 1.3, "d": 2048, "layers": 24},
    {"id": "3b",  "name": "3B",   "params": 3.0, "d": 3200, "layers": 26},
    {"id": "7b",  "name": "7B",   "params": 7.0, "d": 4096, "layers": 32},
    {"id": "13b", "name": "13B",  "params": 13.0,"d": 5120, "layers": 40},
    {"id": "33b", "name": "33B",  "params": 33.0,"d": 6656, "layers": 52},
    {"id": "65b", "name": "65B",  "params": 65.0,"d": 8192, "layers": 64},
    {"id": "70b", "name": "70B",  "params": 70.0,"d": 8192, "layers": 80},
]

# ═══════════════════════════════════════
# DEEP DIVE CONTENT — educational pages
# ═══════════════════════════════════════
DEEP_DIVES = {
    "full": {
        "sections": [
            {
                "title": "How It Works",
                "content": "Full fine-tuning is conceptually the simplest approach: take a pretrained model and continue training it on your task-specific dataset. Every single parameter in the model — from embedding layers through all attention heads and feed-forward networks to the output layer — receives gradient updates via backpropagation. The optimizer (typically AdamW) maintains two additional states per parameter: a running mean of gradients (momentum) and a running mean of squared gradients (variance). This means you need approximately 4× the model size in GPU memory just for training state: 1× for weights (BF16), 1× for gradients (BF16), and 2× for optimizer states (FP32)."
            },
            {
                "title": "Memory Breakdown",
                "content": "For a 7B parameter model in mixed precision: Weights = 7B × 2 bytes (BF16) = 14GB. Gradients = 7B × 2 bytes = 14GB. Adam optimizer states = 7B × 4 bytes × 2 (m + v) = 56GB. Activations vary with batch size and sequence length but typically add 5-20GB. Total: ~60-90GB. For a 70B model, multiply everything by 10: you need ~560GB+ minimum, requiring multi-GPU setups with FSDP or DeepSpeed ZeRO-3 to shard the training state across devices."
            },
            {
                "title": "When to Use It",
                "content": "Full fine-tuning makes sense when: (1) you have abundant compute and need maximum task performance, (2) your task domain is very different from the pretraining data, (3) you only need one task-specific model (no multi-task serving), and (4) you can afford to store and deploy a full model checkpoint per task. It's the gold standard for quality but rarely the practical choice given the cost."
            },
            {
                "title": "Key Hyperparameters",
                "content": "Learning rate is critical — typically 1e-5 to 5e-6 for large models (much lower than pretraining). Use cosine schedule with warmup. Batch size should be as large as memory allows, using gradient accumulation if needed. Train for 1-3 epochs to avoid overfitting. Consider mixing in a fraction of pretraining data to mitigate catastrophic forgetting."
            },
        ],
        "benchmarks": [
            {"task": "MMLU (5-shot)", "score": "68.9", "baseline": "63.5", "note": "LLaMA-2 7B full FT on instruction data"},
            {"task": "HumanEval", "score": "29.9", "baseline": "12.8", "note": "Code generation after full FT on code"},
            {"task": "MT-Bench", "score": "6.8", "baseline": "5.4", "note": "Multi-turn conversation quality"},
        ],
        "related": [
            {"title": "LoRA", "id": "lora", "why": "99% of the quality at 2% of the parameters"},
            {"title": "RLHF/DPO", "id": "rlhf", "why": "Often follows full FT as an alignment stage"},
        ],
        "tips": [
            "Use gradient checkpointing to trade compute for memory (saves ~40% VRAM)",
            "BF16 mixed precision is essential — never train large models in FP32",
            "DeepSpeed ZeRO-3 or FSDP is mandatory for models >13B on multi-GPU",
            "Watch for catastrophic forgetting — validate on pretrain-domain tasks too",
            "Save checkpoints frequently — full FT runs are expensive to restart",
        ],
    },
    "lora": {
        "sections": [
            {
                "title": "The Core Idea",
                "content": "LoRA's key insight is that the weight updates during fine-tuning have low intrinsic rank — you don't need to modify all d² parameters in a weight matrix to adapt it to a new task. Instead of updating a d×d weight matrix W directly, LoRA learns two small matrices: B (d×r) and A (r×d), where the rank r is much smaller than d (typically 8-64). The effective adapted weight becomes W' = W₀ + BA, where W₀ is the frozen pretrained weight. At initialization, A is drawn from a Gaussian and B is set to zero, so BA = 0 and the model starts with its pretrained behavior."
            },
            {
                "title": "Where Adapters Go",
                "content": "LoRA adapters are typically applied to the attention projection matrices: Q (query), K (key), V (value), and O (output). Some implementations also target the feed-forward network's gate and up/down projections. The 'target_modules' parameter in HuggingFace PEFT controls this. Targeting all linear layers ('all-linear') gives the best quality but uses more parameters. A common sweet spot is Q, K, V, O projections only."
            },
            {
                "title": "Rank Selection",
                "content": "The rank r is the most important hyperparameter. Higher rank = more parameters = more capacity but diminishing returns. Empirically: r=8 works for simple tasks, r=16-32 is the sweet spot for most cases, r=64+ only helps for very complex domain shifts. The lora_alpha parameter (typically 2×r) controls the scaling: the effective update is scaled by alpha/r. This means you can change r without dramatically changing the update magnitude."
            },
            {
                "title": "Inference: Merge or Swap",
                "content": "At inference time, you have two options: (1) Keep the adapter separate and add BA at each forward pass — allows hot-swapping between tasks. (2) Merge the adapter into the base weights: W' = W₀ + BA — zero latency overhead, but you lose the ability to swap. For multi-task serving, keep separate adapters and load them on demand. The merge operation is a simple matrix addition and takes seconds."
            },
        ],
        "benchmarks": [
            {"task": "MMLU (5-shot)", "score": "67.8", "baseline": "63.5", "note": "Within 1.1 points of full FT"},
            {"task": "HumanEval", "score": "27.6", "baseline": "12.8", "note": "r=16, slightly below full FT"},
            {"task": "MT-Bench", "score": "6.6", "baseline": "5.4", "note": "Comparable to full FT at 0.1% params"},
        ],
        "related": [
            {"title": "QLoRA", "id": "qlora", "why": "Same adapters but on a 4-bit quantized base"},
            {"title": "Full FT", "id": "full", "why": "The upper bound LoRA is trying to match"},
            {"title": "Adapters", "id": "adapters", "why": "Alternative PEFT approach with bottleneck modules"},
        ],
        "tips": [
            "Start with r=16 and lora_alpha=32 — this works well for most tasks",
            "Target at least Q, K, V, O projections — targeting only Q gives noticeably worse results",
            "Use lora_dropout=0.05 for regularization on small datasets",
            "Multiple LoRA adapters can share one base model in production — huge memory savings",
            "Merge adapters before benchmarking to measure true inference speed",
        ],
    },
    "qlora": {
        "sections": [
            {
                "title": "Three Innovations",
                "content": "QLoRA combines three techniques: (1) NormalFloat4 (NF4) — a 4-bit data type that is information-theoretically optimal for quantizing normally-distributed weights. Unlike uniform INT4, NF4 has unequal bin widths matched to the Gaussian distribution. (2) Double Quantization — the quantization constants themselves (one per block of 64 weights) are further quantized from FP32 to FP8, saving an additional ~0.3 bits per parameter. (3) Paged Optimizers — borrows from OS virtual memory concepts to automatically evict optimizer states to CPU RAM when GPU memory is exhausted, paging them back on demand."
            },
            {
                "title": "Why 4-bit Works",
                "content": "The insight is that the base model only needs to produce accurate forward pass activations — it doesn't need full-precision weights for this. 4-bit NF4 preserves forward pass quality surprisingly well because it's optimized for the actual weight distribution (which is approximately Gaussian). The LoRA adapters, however, need full BF16 precision for stable gradient computation. This asymmetry — low-precision base, high-precision adapters — is what makes QLoRA possible."
            },
            {
                "title": "Memory Math",
                "content": "For a 65B model: Base weights in NF4 = 65B × 0.5 bytes = ~32.5GB. Double quantization overhead = ~0.5GB. LoRA adapters in BF16 = ~50MB (negligible). Optimizer states for adapters = ~200MB. Activations = ~5-10GB. Total: ~38-43GB — fits on a single 48GB A100! Compare to BF16 inference-only: 65B × 2 bytes = 130GB. QLoRA gives you training for less memory than BF16 inference."
            },
            {
                "title": "Quality vs LoRA",
                "content": "Extensive benchmarks show QLoRA matches LoRA quality within 0.5-1 point on most tasks. The Guanaco models trained with QLoRA achieved 99.3% of ChatGPT performance on the Vicuna benchmark using only a single 48GB GPU for training. The quality gap comes from quantization noise in the forward pass, but the adapter gradients remain clean."
            },
        ],
        "benchmarks": [
            {"task": "Vicuna Benchmark", "score": "99.3% of ChatGPT", "baseline": "ChatGPT", "note": "Guanaco 65B, single 48GB GPU"},
            {"task": "MMLU (5-shot)", "score": "67.2", "baseline": "63.5", "note": "Within 0.6 points of LoRA"},
            {"task": "Memory (65B)", "score": "~41GB", "baseline": "~560GB (Full FT)", "note": "13.6× memory reduction"},
        ],
        "related": [
            {"title": "LoRA", "id": "lora", "why": "QLoRA is LoRA + quantization"},
            {"title": "Full FT", "id": "full", "why": "What QLoRA replaces for resource-constrained teams"},
        ],
        "tips": [
            "Use bitsandbytes library — it handles NF4 quantization and paged optimizers",
            "Always use bnb_4bit_use_double_quant=True for maximum memory savings",
            "Set bnb_4bit_compute_dtype to bfloat16, not float16 (better numerical stability)",
            "prepare_model_for_kbit_training() is required — it handles gradient checkpointing setup",
            "QLoRA is the go-to for anyone without datacenter GPUs — fine-tune 70B on consumer hardware",
        ],
    },
    "prefix": {
        "sections": [
            {
                "title": "Continuous Prompts",
                "content": "Prefix tuning is based on a simple idea: instead of manually crafting text prompts, learn continuous vectors that serve as 'virtual tokens' prepended to the model's input. But unlike prompt tuning (which only modifies the input embedding layer), prefix tuning prepends trainable vectors to the Key and Value matrices at every transformer layer. This gives the prefix much more influence — it can steer the attention patterns at every depth of the network."
            },
            {
                "title": "Reparameterization Trick",
                "content": "Directly optimizing the prefix vectors can be unstable (high variance gradients, since each vector is only d-dimensional). The solution: reparameterize the prefix through a small feedforward network (MLP). During training, the prefix is generated by: P = MLP(P_input), where P_input is a learnable embedding. This MLP is discarded after training — you only keep the final prefix vectors P. This trick significantly stabilizes training."
            },
            {
                "title": "Context Window Tradeoff",
                "content": "Prefix tokens consume part of the effective context window. With m=20 prefix tokens on a 2048-context model, you lose ~1% of your context. On longer-context models (8K+), this overhead becomes negligible. However, for very short prompts, the prefix-to-input ratio can be high, potentially hurting performance."
            },
            {
                "title": "Comparison to Prompt Tuning",
                "content": "Prompt tuning only adds trainable embeddings at the input layer — much simpler but less expressive. Prefix tuning adds to K,V at every layer — more parameters but significantly better on complex tasks. P-Tuning v2 extends prefix tuning to also prepend to intermediate representations, bridging the gap further. In practice, LoRA has largely replaced prefix methods due to better quality-to-parameter ratios."
            },
        ],
        "benchmarks": [
            {"task": "SuperGLUE", "score": "86.2", "baseline": "84.1 (full FT)", "note": "GPT-2 Large, surprisingly competitive"},
            {"task": "E2E (table-to-text)", "score": "BLEU 68.2", "baseline": "69.1 (full FT)", "note": "Within 1 point"},
            {"task": "Params (7B model)", "score": "5.2M", "baseline": "7B", "note": "0.075% of total — extremely lightweight"},
        ],
        "related": [
            {"title": "LoRA", "id": "lora", "why": "Generally preferred — better quality/param ratio"},
            {"title": "Adapters", "id": "adapters", "why": "Another PEFT approach with structural modifications"},
        ],
        "tips": [
            "num_virtual_tokens=20 is a good starting point — increase to 50+ for complex tasks",
            "Always use prefix_projection=True (the reparameterization MLP) for stable training",
            "Prefix tuning shines for multi-task serving — swap prefix vectors instantly, near-zero cost",
            "Consider LoRA instead if quality is the priority — prefix tuning has a lower ceiling",
            "Works well for classification and simple generation — less suited for complex instruction following",
        ],
    },
    "adapters": {
        "sections": [
            {
                "title": "Bottleneck Architecture",
                "content": "Adapter layers insert small neural network modules between the existing sublayers of each transformer block. Each adapter has a specific structure: a down-projection (d → m, where m << d), a nonlinear activation (GeLU or ReLU), an up-projection (m → d), and a residual connection from the input. The bottleneck dimension m forces the adapter to learn a compressed, task-specific representation. With m=64 on d=4096, the compression ratio is 64:1."
            },
            {
                "title": "The Residual Connection",
                "content": "The residual connection is the most important architectural detail. The adapter output is ADDED to the input: output = input + adapter(input). Since the adapter weights are initialized with small random values (near-zero output), the initial adapter is approximately an identity function. This means the model starts with its pretrained behavior intact, and the adapter gradually learns to add task-specific modifications on top. This is much more stable than directly modifying the pretrained weights."
            },
            {
                "title": "Placement Options",
                "content": "The original paper places two adapters per transformer block: one after the multi-head attention sublayer, and one after the feed-forward network. Later work (AdapterDrop) showed you can drop adapters from lower layers with minimal quality loss, reducing both parameters and inference latency. Pfeiffer-style adapters use only one adapter per block (after the FFN), halving the overhead."
            },
            {
                "title": "Composability with AdapterFusion",
                "content": "A unique advantage of adapters is composability. AdapterFusion trains a set of attention-like weights to combine pretrained adapters from different tasks. You can take an adapter trained on sentiment analysis and another trained on NER, then learn to combine them for a new task — without retraining either original adapter. The AdapterHub ecosystem provides hundreds of pretrained adapters you can download and compose."
            },
        ],
        "benchmarks": [
            {"task": "GLUE Average", "score": "85.7", "baseline": "86.4 (full FT)", "note": "BERT-base, within 0.7 points"},
            {"task": "SQuAD v2 F1", "score": "83.1", "baseline": "84.2 (full FT)", "note": "m=64 bottleneck"},
            {"task": "Inference overhead", "score": "+4-6%", "baseline": "0% (LoRA merged)", "note": "Sequential adapters add latency"},
        ],
        "related": [
            {"title": "LoRA", "id": "lora", "why": "No inference overhead (merge into weights)"},
            {"title": "Prefix Tuning", "id": "prefix", "why": "Another approach that doesn't modify weights"},
        ],
        "tips": [
            "Start with bottleneck dimension m=64 — increase to 256 for complex tasks",
            "Pfeiffer-style (one adapter per block, after FFN) is simpler and often sufficient",
            "Use AdapterDrop to remove lower-layer adapters for faster inference",
            "AdapterHub has pretrained adapters — check before training from scratch",
            "Consider LoRA for production — adapters add inference latency, LoRA doesn't after merge",
        ],
    },
    "rlhf": {
        "sections": [
            {
                "title": "The RLHF Pipeline",
                "content": "RLHF consists of three stages: (1) Supervised Fine-Tuning (SFT) — train on high-quality demonstration data to create a capable base model. (2) Reward Model Training — collect human preference data (pairs of responses where one is preferred) and train a model to predict which response humans prefer. (3) PPO Optimization — use the reward model as a scoring function and optimize the policy (language model) to maximize reward while staying close to the SFT model via a KL divergence penalty."
            },
            {
                "title": "DPO: The Simpler Alternative",
                "content": "DPO's breakthrough insight: the optimal policy under the RLHF objective has a closed-form relationship with the reward function. This means you can directly optimize on preference pairs WITHOUT ever training a reward model. The loss function compares the log-probability ratios of preferred vs rejected responses under the policy and reference models. If the policy assigns higher relative probability to the preferred response, the loss is low. DPO requires only 2 models (policy + frozen reference) vs PPO's 4 models (policy + reference + reward + value)."
            },
            {
                "title": "Preference Data",
                "content": "Both RLHF and DPO require preference data: pairs of (prompt, chosen_response, rejected_response). Quality matters enormously — noisy preferences lead to reward hacking. Sources include: human annotators (expensive but highest quality), AI feedback from a stronger model (Constitutional AI approach), or synthetic data from comparing outputs of different model checkpoints. Typical datasets range from 10K to 500K preference pairs."
            },
            {
                "title": "The KL Penalty and Beta",
                "content": "The β (beta) parameter controls how much the policy can diverge from the reference model. Higher β = more conservative updates (stay close to reference). Lower β = more aggressive optimization (risk of reward hacking). Typical values: β=0.1-0.5 for DPO, with the equivalent KL penalty coefficient for PPO. If your model starts generating degenerate outputs, increase β. If it's not learning from preferences, decrease β."
            },
        ],
        "benchmarks": [
            {"task": "MT-Bench", "score": "7.1 (DPO)", "baseline": "6.8 (SFT only)", "note": "Clear improvement from alignment"},
            {"task": "AlpacaEval 2", "score": "14.7% win rate", "baseline": "5.4% (SFT)", "note": "Zephyr-7B DPO vs GPT-4"},
            {"task": "Memory (DPO 7B)", "score": "~60GB", "baseline": "~120GB (PPO)", "note": "DPO needs 2 models, PPO needs 4"},
        ],
        "related": [
            {"title": "Full Fine-Tuning", "id": "full", "why": "SFT stage is essentially full FT"},
            {"title": "LoRA", "id": "lora", "why": "Can combine LoRA + DPO for memory-efficient alignment"},
        ],
        "tips": [
            "DPO is strongly preferred over PPO for most use cases — simpler, more stable, less memory",
            "Start with beta=0.1 and adjust based on output quality",
            "Quality of preference data matters more than quantity — 10K clean pairs > 100K noisy",
            "Always evaluate with human preference win-rates, not just automated metrics",
            "You can combine QLoRA + DPO to align large models on consumer hardware",
        ],
    },
}

@app.route("/")
def landing():
    return render_template("landing.html", methods=METHODS)

@app.route("/explore")
def explore():
    return render_template("explore.html", methods=METHODS)

@app.route("/playground")
def playground():
    return render_template("playground.html", methods=METHODS, gpus=GPUS, models=MODEL_SIZES)

@app.route("/learn/<method_id>")
def learn(method_id):
    method = next((m for m in METHODS if m["id"] == method_id), None)
    if method is None:
        return "Method not found", 404
    deep = DEEP_DIVES.get(method_id, {})
    return render_template("learn.html", method=method, deep=deep, methods=METHODS)

@app.route("/guide")
def guide():
    return render_template("guide.html", methods=METHODS)

@app.route("/api/methods")
def api_methods():
    return jsonify(METHODS)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
