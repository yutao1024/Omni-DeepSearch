import os
import sys
import asyncio
import argparse
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from acquisition_pipeline import AcquisitionPipeline
from audio_qa_pipeline import AudioDeepSearchPipeline
from audio_to_image_pipeline import AudioToImagePipeline
from audio_tracing_pipeline import AudioTracingPipeline

sys.path.insert(0, os.path.join(BASE_DIR, "inference"))
from inference_pipeline import InferencePipeline


def set_workspace(workspace_path):
    config_path = os.path.join(BASE_DIR, "config", "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if 'acquisition' not in config:
        config['acquisition'] = {}
    config['acquisition']['workspace'] = os.path.abspath(workspace_path)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "acquire-audio",
            "single-qa",
            "single-qa-only",
            "image-qa",
            "image-qa-only",
            "video-qa",
            "video-qa-only",
            "multi-qa",
            "multi-qa-only",
            "evaluate",
        ],
    )
    parser.add_argument("input_path", nargs="*")
    parser.add_argument("--workspace", default=None)
    args = parser.parse_args()

    if args.workspace:
        set_workspace(args.workspace)

    if args.task == "acquire-audio":
        acq = AcquisitionPipeline()
        await acq.run()

    elif args.task == "single-qa":
        acq = AcquisitionPipeline()
        await acq.run()
        pipeline = AudioDeepSearchPipeline()
        await pipeline.run_full_pipeline()

    elif args.task == "single-qa-only":
        pipeline = AudioDeepSearchPipeline()
        await pipeline.run_full_pipeline()

    elif args.task == "image-qa":
        acq = AcquisitionPipeline()
        await acq.run()
        pipeline = AudioToImagePipeline()
        await pipeline.run_pipeline()

    elif args.task == "image-qa-only":
        pipeline = AudioToImagePipeline()
        await pipeline.run_pipeline()

    elif args.task == "video-qa":
        pipeline = AudioTracingPipeline()
        await pipeline.run_full_pipeline()

    elif args.task == "video-qa-only":
        pipeline = AudioTracingPipeline()
        await pipeline.run_generation_stage()

    elif args.task == "multi-qa":
        acq = AcquisitionPipeline()
        await acq.run_interaction_pipeline()
        pipeline = AudioDeepSearchPipeline(is_interaction=True)
        await pipeline.run_interaction_full_pipeline()

    elif args.task == "multi-qa-only":
        pipeline = AudioDeepSearchPipeline(is_interaction=True)
        await pipeline.run_interaction_full_pipeline()

    elif args.task == "evaluate":
        if len(args.input_path) < 1:
            print("Error: evaluate requires at least one argument (merged.json file).")
            return
        json_path = args.input_path[0]
        output_spec = args.input_path[1] if len(args.input_path) > 1 else None
        eval_pipeline = InferencePipeline(workspace_root=json_path, output_dir=output_spec)
        await eval_pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())