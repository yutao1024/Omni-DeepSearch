# Omni DeepSearch Benchmark

A fully automated pipeline for generating, filtering, and evaluating **multimodal audio deep‑search benchmarks**.  
It acquires topic‑specific audio from YouTube, builds multi‑hop reasoning QA pairs with knowledge graphs and multimodal LLMs, and evaluates benchmark difficulty through dedicated agents and judge models.

## Requirements

- Python 3.8+
- FFmpeg (for audio extraction and slicing)
- Node.js (required by yt‑dlp's JavaScript runtime)

## Installation

```bash
# Clone the repository
git clone https://github.com/omni-benchmark/omni-bench.git
cd omni-bench

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

## Configuration

Edit config/config.yaml with your own API keys and paths:
```bash
serper_api_key: "your-serper-key"
jina_api_key: "your-jina-key"

llm:
  qwen:
    api_key: "your-qwen-key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: "qwen3.5-omni-plus"
  # Configure other providers (gemini, gpt, claude, etc.) as needed

# Path to a YouTube cookies file (required for downloading videos)
cookies_path: "config/cookies.txt"
```
Important: Downloading from YouTube requires a cookies.txt file exported from your browser. Place it under config/ and update the path above.

## Quick Start

All generated data will be placed under data_workspace/.
### 1. Acquire Audio
Downloads thematic audio clips from YouTube across SPEECH, MUSIC, BIO, and ENV domains.
```bash
python acquisition_pipeline.py
```

### 2. Generate Benchmarks
Choose the task type:
```bash
# Single‑audio and multi‑audio interaction questions
python audio_qa_pipeline.py

# Audio‑to‑image visual reasoning questions
python audio_to_image_pipeline.py

# Audio‑tracing (video source identification) questions
python audio_tracing_pipeline.py
```
Each script creates a timestamped folder inside data_workspace/benchmark_runs/ containing the raw benchmark, filtered versions, and necessity‑verified subsets.

### 3.Evaluate
After benchmarks are generated:
```bash
python inference_pipeline.py data_workspace/benchmark_runs/20250301_120000

python inference_pipeline.py data_workspace/benchmark_runs/20250301_120000 /path/to/previous/output
```


## License
This project is provided for research purposes only.

