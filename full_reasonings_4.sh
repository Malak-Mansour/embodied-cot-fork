#!/bin/bash
#SBATCH -p cscc-gpu-p                  # Partition
#SBATCH -q cscc-gpu-qos               # QoS
#SBATCH --gres=gpu:1                  # GPU request
#SBATCH --mem=64G                     # Memory
#SBATCH --cpus-per-task=8            # CPU cores
#SBATCH --job-name=full_reasoning    # Job name
#SBATCH --output=full_reasoning_%j.out  # Stdout
#SBATCH --error=full_reasoning_%j.err   # Stderr

# Load environment
source ~/.bashrc
conda activate openvla

# Run the script
python scripts/generate_embodied_data/full_reasonings_4.py
