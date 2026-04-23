import modal
import os
import subprocess
from pathlib import Path

app = modal.App("rl-ad-bidding-training")

# 1. Define the environment image and add local files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        local_path=".",
        remote_path="/root/rl-ad-bidding",
        ignore=["**/.git", "**/venv", "**/__pycache__", "**/saved_models"]
    )
)

# 2. Define a persistent volume for saved_models
volume = modal.Volume.from_name("rl-ad-bidding-models", create_if_missing=True)

# 3. Define the training function
@app.function(
    image=image,
    volumes={"/root/rl-ad-bidding/saved_models": volume},
    cpu=32.0, # Removed GPU to save costs, maximizing CPU power
    timeout=86400, # 24 hours max
    secrets=[modal.Secret.from_name("wandb-secret")],
    schedule=modal.Cron("0 0 * * *") # Runs automatically every day at midnight!
)
def train_on_modal(config_path: str = "configs/default.yaml"):
    # Change working directory to the mounted project
    os.chdir("/root/rl-ad-bidding")
    
    print("Starting training on Modal...")
    
    # Execute the existing train.py script
    result = subprocess.run(
        ["python", "scripts/train.py", "--config", config_path],
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Training failed with exit code {result.returncode}")
        
    print("Training finished. Checkpoints should be saved to the persistent volume.")

# 5. Local entrypoint to trigger the function
@app.local_entrypoint()
def main(config: str = "configs/default.yaml"):
    print(f"Submitting job to Modal with config: {config}")
    train_on_modal.remote(config)
