'''
python scripts/generate_embodied_data/bounding_boxes/generate_bboxes.py --id 0 --gpu 0 --splits 4 --data-path /l/users/malak.mansour/Datasets/do_manual/hdf5 
'''

import argparse
import json
import os
import time
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils import NumpyFloatValuesEncoder, post_process_caption

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()

parser.add_argument("--id", type=int)
parser.add_argument("--gpu", type=int)
parser.add_argument("--splits", default=24)
parser.add_argument("--data-path", type=str)
parser.add_argument("--result-path", default="./bboxes")

args = parser.parse_args()
bbox_json_path = os.path.join(args.result_path, f"results_{args.id}_bboxes.json")
viz_dir = os.path.join(args.result_path, "visualizations")

print("Loading data...")
# split_percents = 100 // args.splits
# start = args.id * split_percents
# end = (args.id + 1) * split_percents

# ds = tfds.load("bridge_orig", data_dir=args.data_path, split=f"train[{start}%:{end}%]")
import glob
import h5py
dataset_paths = sorted(glob.glob(os.path.join(args.data_path, "*.h5")))
# selected_paths = dataset_paths[start * len(dataset_paths) // 100 : end * len(dataset_paths) // 100]
selected_paths = dataset_paths
print("Done.")

print("Loading Prismatic descriptions...")
results_json_path = "results_0.json"
with open(results_json_path, "r") as results_f:
    results_json = json.load(results_f)
print("Done.")

print(f"Loading gDINO to device {args.gpu}...")
model_id = "IDEA-Research/grounding-dino-base"
device = f"cuda:{args.gpu}"

processor = AutoProcessor.from_pretrained(model_id, size={"shortest_edge": 256, "longest_edge": 256})
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
print("Done.")

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.2
KEEP_TOP1_ONLY = False  # set True to keep only first match

def draw_bboxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        box = [int(v) for v in box]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1] - 10), label, fill="red")
    return image

bbox_results_json = {}
# for ep_idx, episode in enumerate(ds):

#     episode_id = episode["episode_metadata"]["episode_id"].numpy()
#     file_path = episode["episode_metadata"]["file_path"].numpy().decode()
#     print(f"ID {args.id} starting ep: {episode_id}, {file_path}")

#     if file_path not in bbox_results_json.keys():
#         bbox_results_json[file_path] = {}

#     episode_json = results_json[file_path][str(episode_id)]
#     description = episode_json["caption"]

#     start = time.time()
#     bboxes_list = []
#     for step_idx, step in enumerate(episode["steps"]):
#         if step_idx == 0:
#             lang_instruction = step["language_instruction"].numpy().decode()
#         image = Image.fromarray(step["observation"]["image_0"].numpy())
for file_path in selected_paths:
    file_name = os.path.basename(file_path).split(".")[0]

    with h5py.File(file_path, "r") as f:
        for ep_name in f:
            ep_group = f[ep_name]
            if "obs" not in ep_group:
                continue
            obs = ep_group["obs"]
            lang_instruction = ep_name.replace("_", " ")

            if file_path not in bbox_results_json:
                bbox_results_json[file_path] = {}

            if file_path not in results_json or ep_name not in results_json[file_path]:
                print(f"⚠️ Missing caption for: {file_path} / {ep_name}")
                continue

            description = results_json[file_path][ep_name]["caption"]
            bboxes_list = []
            start_time = time.time()

            for i in range(len(obs["agentview_rgb"])):
                image = Image.fromarray(obs["agentview_rgb"][i])




                inputs = processor(
                    images=image,
                    text=post_process_caption(description, lang_instruction),
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD,
                    target_sizes=[image.size[::-1]],
                )[0]

                logits, phrases, boxes = (
                    results["scores"].cpu().numpy(),
                    results["labels"],
                    results["boxes"].cpu().numpy(),
                )

                bboxes = []
                for lg, p, b in zip(logits, phrases, boxes):
                    b = list(b.astype(int))
                    lg = round(lg, 5)
                    bboxes.append((lg, p, b))
                    if KEEP_TOP1_ONLY:
                        break

                bboxes_list.append(bboxes)

                # Save visualization
                viz_path = os.path.join(viz_dir, file_name, ep_name)
                os.makedirs(viz_path, exist_ok=True)
                img_copy = image.copy()
                img_viz = draw_bboxes(img_copy, boxes[:1] if KEEP_TOP1_ONLY else boxes, phrases[:1] if KEEP_TOP1_ONLY else phrases)
                img_viz.save(os.path.join(viz_path, f"{i:04}.jpg"))

                # break
            # end = time.time()
            # bbox_results_json[file_path][str(ep_idx)] = {
            #     "episode_id": int(episode_id),

            bbox_results_json[file_path][ep_name] = {
                "episode_id": ep_name,
                "file_path": file_path,
                "bboxes": bboxes_list,
            }

            with open(bbox_json_path, "w") as out_f:
                json.dump(bbox_results_json, out_f, cls=NumpyFloatValuesEncoder)
            # print(f"ID {args.id} finished ep ({ep_idx} / {len(ds)}). Elapsed time: {round(end - start, 2)}")
            print(f"✅ {file_path} / {ep_name} done. Time: {round(time.time() - start_time, 2)}s")