#!/bin/bash
#SBATCH -p cscc-gpu-p                 # Partition
#SBATCH -q cscc-gpu-qos              # QoS
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mem=64G                    # Memory
#SBATCH --cpus-per-task=8           # Number of CPUs
#SBATCH --job-name=desc_gen         # Job name
#SBATCH --output=desc_gen_%j.out    # Output file
#SBATCH --error=desc_gen_%j.err     # Error file

# Load your environment if needed
source ~/.bashrc
conda activate openvla

# Run your command
python scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py --id 1 --gpu 0
