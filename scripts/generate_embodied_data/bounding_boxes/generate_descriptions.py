# python scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py --id 1 --gpu 0

import argparse
import json
import os
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from utils import NumpyFloatValuesEncoder

from prismatic import load

parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--splits", default=4, type=int)
parser.add_argument("--results-path", default="./")

args = parser.parse_args()

device = f"cuda:{args.gpu}"
# hf_token = "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"
hf_token = "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"
vlm_model_id = "prism-dinosiglip+7b"

warnings.filterwarnings("ignore")


split_percents = 100 // args.splits
start = args.id * split_percents
end = (args.id + 1) * split_percents

# Load Bridge V2
# ds = tfds.load(
#     "bridge_orig",
#     data_dir="<TODO: Enter path to BridgeV2>",
#     split=f"train[{start}%:{end}%]",
# )
import h5py
import glob

dataset_paths = sorted(glob.glob("/l/users/malak.mansour/Datasets/do_manual/hdf5_rgb/*.h5"))
# selected_paths = dataset_paths[start * len(dataset_paths) // 100 : end * len(dataset_paths) // 100]
selected_paths = dataset_paths
print("Selected paths:", selected_paths)

# Load Prismatic VLM
print(f"Loading Prismatic VLM ({vlm_model_id})...")
vlm = load(vlm_model_id, hf_token=hf_token)
vlm = vlm.to(device, dtype=torch.bfloat16)
# vlm = vlm.to(device, dtype=torch.float16)
# vlm = vlm.to(device)
# if device == "cuda":
#     vlm = vlm.to(device, dtype=torch.bfloat16)
# else:
#     vlm = vlm.to(device)  # default float32 on CPU


results_json_path = os.path.join(args.results_path, f"results_{args.id}.json")


def create_user_prompt(lang_instruction):
    user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    # user_prompt = "Briefly describe the objects in this scene."]
    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    return user_prompt


results_json = {}
# for episode in tqdm(ds):
#     episode_id = episode["episode_metadata"]["episode_id"].numpy()
#     file_path = episode["episode_metadata"]["file_path"].numpy().decode()
#     for step in episode["steps"]:
#         lang_instruction = step["language_instruction"].numpy().decode()
#         image = Image.fromarray(step["observation"]["image_0"].numpy())
for file_path in tqdm(selected_paths):
    print(f"Processing file: {file_path}")
    with h5py.File(file_path, "r") as f:
        for ep_name in f:
            print(f"  Episode: {ep_name}")  # Add this
            ep_group = f[ep_name]
            if "obs" not in ep_group: 
                print(f"    ⚠️ Skipping {ep_name} — 'obs' group not found.")
                continue
            obs = ep_group["obs"]

            if "agentview_rgb" not in obs:
                print(f"    ⚠️ Skipping {ep_name} — 'agentview_rgb' not in obs.")
                continue
        
            image = obs["agentview_rgb"][0]  # or "eye_in_hand_rgb"
            lang_instruction = ep_name.replace("_", " ")  # crude fallback for now

            image = Image.fromarray(image)

            # user_prompt = "Describe the objects in this scene. Be specific."
            user_prompt = create_user_prompt(lang_instruction)
            prompt_builder = vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=user_prompt)
            prompt_text = prompt_builder.get_prompt()

            torch.manual_seed(0)
            caption = vlm.generate(
                image,
                prompt_text,
                do_sample=True,
                temperature=0.4,
                max_new_tokens=64,
                min_length=1,
            )
            # break
            print(f"    ✅ Caption: {caption[:60]}...")

            # episode_json = {
            #     "episode_id": int(episode_id),
            #     "file_path": file_path,
            #     "caption": caption,
            # }

            # if file_path not in results_json.keys():
            #     results_json[file_path] = {}

            # results_json[file_path][int(episode_id)] = episode_json

            if  file_path not in results_json:
                results_json[file_path] = {}

            results_json[file_path][ep_name] = {
                "caption": caption,
                "prompt": prompt_text,
            }


            with open(results_json_path, "w") as out_f:
                json.dump(results_json, out_f, cls=NumpyFloatValuesEncoder)

