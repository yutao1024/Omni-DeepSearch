<div align="center">
  <img width="800" alt="pipeline" src="assets/main.png" />
  <br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/) [![GitHub Repo](https://img.shields.io/badge/GitHub-Omni--Bench-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/yutao1024/Omni-DeepSearch) [![arXiv](https://img.shields.io/badge/arXiv-0000.00000-B31B1B?style=flat-square&logo=arxiv&logoColor=white)]() [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/Kirito-Lab/Omni-DeepSearch)

  ✨ **Focus on Multimodal Audio Deep-Search: Automated generation, filtering, and evaluation of multi-hop reasoning benchmarks** ✨  

  | 🧩 **Fully Automated Pipeline** | 🎵 **Rich Audio Domains** | 🧠 **Multi-Hop Reasoning QA** | 🤖 **Agentic Evaluation** |
</div>

## 📑 Table of Contents

<div align='center'>

[🔥 News](#-news) | [🌈 Overview](#-overview) | [🤗 Dataset](#-dataset) | [🚀 Quick Start](#-quick-start) | [📖 Citation](#-citation)

</div>

## 🔥 News

**[202X/0X/0X]** 🚀 Omni-DeepSearch dataset is now officially live on Hugging Face! Click here to explore our [Dataset](https://huggingface.co/datasets/Kirito-Lab/Omni-DeepSearch).

**[202X/0X/0X]** 📄 Omni-DeepSearch is now available on [arXiv]().

## 🌈 Overview    

<div align="center">
  <img width="800" alt="overview" src="assets/benchmark2_compressed.png" />
</div>
<br/>

## 🤗 Dataset

Access the Omni-DeepSearch dataset, featuring complex multi-hop reasoning audio QA pairs, now available on Hugging Face: [Omni-DeepSearch Dataset](https://huggingface.co/datasets/Kirito-Lab/Omni-DeepSearch).

## 🚀 Quick Start

### Prerequisites
Before installation, ensure you have the following system dependencies installed:
- **Python** 3.8+
- **FFmpeg** (for audio extraction and slicing)
- **Node.js** (required by `yt‑dlp`'s JavaScript runtime)

### Installation

#### 1. Clone this repository and navigate to the folder:
```bash
git clone https://github.com/yutao1024/Omni-DeepSearch.git
cd omni-bench
```

#### 2. Create a dedicated Python environment
We strongly recommend using a virtual environment to avoid dependency conflicts.
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using Conda
conda create -n omni-bench python=3.10
conda activate omni-bench
```

#### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configuration
Edit `config/config.yaml` with your own API keys, LLM configurations, and paths.  
A minimal example:
```yaml
serper_api_key: "your-serper-key"     # used for web search (Serper.dev)
jina_api_key: "your-jina-key"         # used for full‑text webpage extraction (Jina.ai)

llm:
  default_provider: ""               # main model for generation 
  filter_provider: ""                # lighter model for filtering/auditing tasks
  gemini:
    api_key: "your-gemini-key"
    base_url: "your-gemini-url"
    model_name: "your-model-version"
  # add other providers (qwen, gpt, claude, mimo) similarly

# Where downloaded videos and generated benchmarks will be stored
acquisition:
  workspace: data_workspace

# Path to a YouTube cookies file (required for age‑restricted or private videos)
cookies_path: "config/cookies.txt"
```

**How to obtain the required API keys:**

- **Serper API Key (serper_api_key):**  
  Go to [https://serper.dev](https://serper.dev), sign up, and copy your API key. This key is used for Google search lookups when generating questions or evaluating models.  

- **Jina API Key (jina_api_key):**  
  Visit [https://jina.ai](https://jina.ai) and obtain a free API key. It allows the pipeline to fetch and summarize full webpage content.  

- **Cookies file (cookies_path):**  
  The pipeline downloads videos from YouTube. To bypass age‑restrictions and playback limits you must export a `cookies.txt` file from your browser while logged into a YouTube account.  
  - Use a browser extension like “Get cookies.txt LOCALLY” (Chrome, Edge) or “cookies.txt” (Firefox) while on youtube.com.  
  - Save the exported file as `config/cookies.txt`.  

- **LLM Provider Keys:**  
  For each LLM provider you intend to use (Qwen, Gemini, GPT, Claude, etc.) you need to fill in the `api_key` and `base_url` fields under the respective provider block in `config.yaml`. The `default_provider` is used for question generation; `filter_provider` is used for quick quality checks and audits.

⚠️ **Important:**  
- Make sure all API keys are valid and have sufficient quota.  
- Adjust the `workspace` path if you want your downloads and generated data to reside elsewhere (e.g., an external drive).

#### 5. Tuning acquisition volume
The amount of data downloaded and the diversity of the final benchmark are controlled by several settings in `config.yaml`. Add or modify them under the `acquisition`, `max_threads`, and `sampling` sections according to your needs.

| Parameter | Section | Description |
|-----------|---------|-------------|
| `themes_per_domain` | `acquisition` | Number of distinct topics/themes the LLM generates for each audio domain (SPEECH, MUSIC, BIO, ENV). Each theme leads to one or more YouTube searches. |
| `queries_per_theme` | `acquisition` | How many different YouTube search queries the LLM creates per theme. More queries increase the chance of finding suitable, high‑quality videos. |
| `videos_per_query` | `acquisition` | Maximum number of videos the pipeline attempts to download for each search query. Higher values provide fallback options. |
| `max_threads` | root | Maximum number of concurrent download/processing tasks. Raise this for faster downloads; lower it if you encounter rate‑limit errors. |
| `interval` | `sampling` | Time in seconds between consecutive slices taken from a downloaded video. |
| `audio_duration` | `sampling` | Duration in seconds of each extracted audio clip. |
| `max_samples_per_video` | `sampling` | Maximum number of consecutive slices used from a single video when building a clip. |

Example for increasing collection scale:
```yaml
acquisition:
  workspace: large_workspace
  themes_per_domain: 10
  queries_per_theme: 3
  videos_per_query: 2
max_threads: 20
sampling:
  interval: 8.0
  audio_duration: 12.0
  max_samples_per_video: 4
```

These values directly influence the total number of downloaded videos, the size of the generated benchmark, and the load on your system and API quotas. Adjust them based on your machine’s capacity and your target dataset size.

### 🧪 Running the Pipelines

All tasks are managed through a single entry point `run_pipeline.py`.  
You can also run individual scripts directly if you already have the necessary downloaded videos.

#### Available task types
The pipeline supports both **full‑cycle** tasks (download + generation) and **generation‑only** tasks (skip download).  
The category labels used in the output benchmarks are `single`, `image`, `video` (audio tracing), and `multi`.

| Command | Description | Category |
|---------|-------------|----------|
| `acquire-audio` | Download thematic audio clips from YouTube (SPEECH, MUSIC, BIO, ENV) | – |
| `single-qa` | Download clips + generate single‑audio multi‑hop questions | single |
| `single-qa-only` | Generate single‑audio questions **without downloading** (videos must already be present) | single |
| `image-qa` | Download clips + generate audio‑to‑image visual reasoning questions | image |
| `image-qa-only` | Generate audio‑to‑image questions **without downloading** | image |
| `video-qa` | Full audio tracing pipeline (download + generation) | video |
| `video-qa-only` | Generate tracing questions **without downloading** (videos must exist in trace workspace) | video |
| `multi-qa` | Download interaction clips + generate multi‑audio interaction questions | multi |
| `multi-qa-only` | Generate interaction questions **without downloading** (interaction clips must exist) | multi |
| `evaluate` | Run model inference on a previously generated benchmark JSON | – |

#### Basic usage
```bash
# Full‑cycle tasks (download + generate)
python run_pipeline.py single-qa
python run_pipeline.py image-qa
python run_pipeline.py video-qa
python run_pipeline.py multi-qa

# Generation‑only tasks (skip downloading, assume videos are already in the workspace)
python run_pipeline.py single-qa-only
python run_pipeline.py image-qa-only
python run_pipeline.py video-qa-only
python run_pipeline.py multi-qa-only

# Download only (no generation)
python run_pipeline.py acquire-audio

# Evaluate a generated benchmark
python run_pipeline.py evaluate /path/to/merged.json
```

#### Specifying a custom workspace
By default, all data is placed under `data_workspace/`. You can override this at runtime with the `--workspace` flag:
```bash
python run_pipeline.py acquire-audio --workspace ./my_custom_data
python run_pipeline.py single-qa --workspace /experiments/run1
```
Downloads will be stored in `<workspace>/downloads/`, and generated benchmarks in `<workspace>/benchmark_runs/`.

#### Using individual scripts (when videos are already downloaded)
If you have already successfully run `acquire-audio` (or `tracing` acquisition) and only want to generate benchmarks, you can call the generation scripts directly. This is useful when you want to tweak generation parameters without re‑downloading videos.

- **Single‑audio QA:**  
  `python audio_qa_pipeline.py` – runs `run_full_pipeline()` internally (generation + filtering).  
- **Audio‑to‑image QA:**  
  `python audio_to_image_pipeline.py` – runs `run_pipeline()`.  
- **Audio tracing:**  
  `python audio_tracing_pipeline.py` – runs `run_full_pipeline()` (includes its own acquisition step unless you call `run_generation_stage()`).  
- **Multi‑audio interaction QA:**  
  First ensure interaction downloads are available by running `python acquisition_pipeline.py` (in interaction mode) or `python run_pipeline.py multi-qa-only` if you have already acquired the interaction clips.  
  Then run `python audio_qa_pipeline.py` with `is_interaction=True` (or use the wrapper).

#### Resuming and incremental runs
- The acquisition stage remembers already downloaded video IDs and processed themes using JSON logs inside the workspace. You can safely interrupt and resume `acquire-audio`, `single-qa`, `image-qa`, or `multi-qa` without re‑downloading already processed items.  
- For evaluation, the pipeline automatically detects existing `final_report.json` and skips samples that have already been processed.

### 📊 Inference & Evaluation

After generating benchmarks, you can evaluate model performance using the evaluation command.

**Basic evaluation:**
```bash
python run_pipeline.py evaluate /path/to/merged.json
```
The pipeline automatically detects task categories, runs inference, and saves a detailed report (`final_report.json`) in a timestamped folder next to the input file.

**Specifying an output directory or file (recommended for clarity and resuming):**
```bash
# Save results into a specific directory
python run_pipeline.py evaluate /path/to/merged.json ./my_eval

# Save results into a specific JSON file
python run_pipeline.py evaluate /path/to/merged.json ./my_eval/results.json
```
- If the second argument is an existing directory, the pipeline will resume from any previous `final_report.json` inside that directory (skipping already‑evaluated samples).  
- If it is a new path ending with `.json`, the report will be written directly to that file (the parent folder is created automatically if needed).  
- If no second argument is given, a timestamped subfolder is created next to the input data.

**Resuming an interrupted evaluation:**
Simply run the exact same command again (with the same output directory or file). Already‑processed samples will be skipped, and only the remaining ones will be evaluated.

**Output report structure:**
- `overall_accuracy` – total correct rate across all categories.
- `macro_category_stats` – accuracy per macro‑category: `SINGLE`, `IMAGE`, `VIDEO`, `MULTI`.
- `sub_category_stats` – accuracy per detailed sub‑category (e.g., `single/SPEECH`, `image/BIO`, `video/ENV`, `multi/2WAY`).
- `per_question_status` – verdict (`Correct` / `Wrong`) and number of retries for each sample.
- `detailed_data` – full records including model answer, evaluator votes, thought process, etc.

### 📂 Output structure
After a successful run, you will find:
```
data_workspace/
├── downloads/           # Raw video files (organised by domain)
├── interactions/        # Multi‑audio interaction source data
├── trace/               # Tracing‑specific downloads
└── benchmark_runs/      # Generated benchmarks, filtered versions, evaluation reports
    └── <timestamp>_<TASK>/
        ├── benchmark.json
        ├── benchmark_FINAL_HARD.json
        ├── benchmark_AUDIO_NECESSARY.json
        ├── benchmark_VISUAL_VERIFIED.json   (for visual tasks)
        ├── audio_clips/                     # Final curated audio clips
        ├── target_images_for_llm/           # (audio‑to‑image task)
        └── ground_truth_images/             # (tracing task)
```

## 📄 License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This project and dataset are licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**. 

## 📖 Citation
Please kindly cite our paper if you find this project or dataset helpful.
```bibtex
@misc{omnideepsearch202X,
      title={}, 
      author={},
      year={},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={},
      url={}, 
}
```