# Installation Guide

This guide covers the installation steps for TAFC on different models.

## Prerequisites

- Python 3.8 or higher
- CUDA 11.8 or higher
- At least 16GB GPU memory (24GB+ recommended for video models)

## FLUX (Text-to-Image)

### 1. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate safetensors
pip install sentencepiece protobuf
```

### 2. Download Model Weights

```bash
# You need a Hugging Face token with FLUX access
huggingface-cli login

# Models will be downloaded automatically on first use
# Or manually download:
# flux-dev: black-forest-labs/FLUX.1-dev
# flux-schnell: black-forest-labs/FLUX.1-schnell
```

### 3. Run TAFC

```bash
cd FLUX
python tafc_generate.py \
    --prompt "a photo of an astronaut riding a horse" \
    --output_dir ./outputs \
    --tafc_thresh 0.2
```

## Wan2.1 (Text-to-Video & Image-to-Video)

### 1. Clone Wan2.1 Repository

```bash
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install imageio[ffmpeg] opencv-python pillow
```

### 3. Download Model Weights

```bash
# Using huggingface-cli
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B

# Or using modelscope (for users in China)
pip install modelscope
modelscope download Wan-AI/Wan2.1-T2V-14B --local_dir ./Wan2.1-T2V-14B
```

### 4. Copy TAFC Files

```bash
# Copy from TAFC repository
cp ../TAFC/Wan2.1/tafc_generate.py .
cp ../TAFC/Wan2.1/util_tafc.py .
```

### 5. Run TAFC

```bash
python tafc_generate.py \
    --prompt "A cat walks on the grass" \
    --save_path ./outputs \
    --video_length 81 \
    --tafc_thresh 0.2
```

## HunyuanVideo (Text-to-Video)

### 1. Clone HunyuanVideo Repository

```bash
git clone https://github.com/Tencent/HunyuanVideo.git
cd HunyuanVideo
```

### 2. Install Dependencies

Follow the official HunyuanVideo installation guide:
```bash
pip install -r requirements.txt
```

### 3. Download Model Weights

Follow the official HunyuanVideo guide to download model checkpoints.

### 4. Copy TAFC Files

```bash
# Copy from TAFC repository
cp ../TAFC/HunyuanVideo/HunyuanVideo/tafc_generate.py .
cp ../TAFC/HunyuanVideo/HunyuanVideo/util_tafc.py .
```

### 5. Run TAFC

```bash
python tafc_generate.py \
    --video-size 720 1280 \
    --video-length 33 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --tafc_thresh 0.3
```

## Evaluation Setup

### For Image Quality Metrics (FLUX)

```bash
pip install clean-fid torch-fidelity
pip install git+https://github.com/openai/CLIP.git
```

### For Video Quality Metrics (Wan2.1, HunyuanVideo)

```bash
pip install pyiqa
```

### For VBench (HunyuanVideo)

```bash
git clone https://github.com/Vchitect/VBench.git
cd VBench
pip install -e .
```

## Troubleshooting

### Out of Memory (OOM)

For FLUX:
```bash
python tafc_generate.py --offload  # Enable CPU offload
```

For Wan2.1:
```bash
python tafc_generate.py --offload_model True --t5_cpu
```

For HunyuanVideo:
```bash
python tafc_generate.py --use-cpu-offload
```

### Slow Download from Hugging Face

Use a mirror or modelscope (for users in China):
```bash
export HF_ENDPOINT=https://hf-mirror.com
# Or use modelscope as shown above
```

### CUDA Version Mismatch

Check your CUDA version:
```bash
nvcc --version
nvidia-smi
```

Install matching PyTorch:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Environment Separation

For evaluation with incompatible dependencies (e.g., VBench vs Wan2.1), use separate conda environments:

```bash
# Environment for generation
conda create -n tafc_gen python=3.9
conda activate tafc_gen
pip install -r requirements.txt

# Environment for evaluation
conda create -n tafc_eval python=3.9
conda activate tafc_eval
pip install pyiqa vbench
```

See memory files for more details on environment conflicts.
