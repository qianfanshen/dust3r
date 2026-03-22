import argparse
import os
import sys
import yaml
import turibolt as bolt

def load_config(yaml_file_path):
    with open(yaml_file_path) as f:
        config = yaml.safe_load(f)
    return config

def bolt_submit(config, exclude):
    return bolt.submit(config, tar='.', exclude=exclude, max_retries=3, interactive=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/bolt_8gpu.yaml", type=str)
    args = parser.parse_args()

    sys.setrecursionlimit(10000)
    os.environ['S3_ENDPOINT_URL'] = 'https://conductor.data.apple.com'

    config = load_config(args.config)
    exclude = [
        "venv", ".venv", ".git", ".idea", ".cache", ".pytest_cache", ".pytype",
        "__pycache__", ".ruff_cache", ".DS_Store", "data_conversion", "misc", ".github", ".claude",
        "checkpoints"
    ]
    bolt_submit(config, exclude)
