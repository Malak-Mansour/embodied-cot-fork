#!/bin/bash
#SBATCH -p cscc-gpu-p                 # Partition
#SBATCH -q cscc-gpu-qos              # QoS
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --mem=64G                    # Memory
#SBATCH --cpus-per-task=8           # Number of CPUs
#SBATCH --job-name=gen_bboxes       # Job name
#SBATCH --output=gen_bboxes_%j.out  # Output log
#SBATCH --error=gen_bboxes_%j.err   # Error log

# Load your environment if needed
source ~/.bashrc
conda activate openvla

# Run the job
python scripts/generate_embodied_data/bounding_boxes/generate_bboxes.py \
  --id 1 \
  --gpu 0 \
  --splits 4 \
  --data-path /l/users/malak.mansour/Datasets/do_manual/hdf5_rgb
