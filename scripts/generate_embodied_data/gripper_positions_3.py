import cv2
import mediapy
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline
from concurrent.futures import ThreadPoolExecutor

# === Initialization ===
checkpoint = "google/owlvit-base-patch16"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=0)
sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to("cuda").eval()
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
image_dims = (224, 224)

# === Utilities ===
def get_bounding_boxes(img, prompt="the black robotic gripper"):
    return detector(img, candidate_labels=[prompt], threshold=0.05)

def get_gripper_mask(img, prediction):
    box = prediction.get("box")
    if not box:
        raise ValueError("Prediction does not contain a bounding box.")

    # Handle both dict and list formats
    if isinstance(box, dict):
        x0 = float(box["xmin"])
        y0 = float(box["ymin"])
        x1 = float(box["xmax"])
        y1 = float(box["ymax"])
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        x0, y0, x1, y1 = map(float, box)
    else:
        raise ValueError(f"Invalid box format: {box}")

    input_box = [[round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]]
    inputs = sam_processor(img, input_boxes=[input_box], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = sam_model(**inputs)

    mask = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0][0][0].cpu().numpy()

    return mask

def sq(w, h):
    coords = np.indices((h, w)).transpose(1, 2, 0)
    return coords[..., ::-1]  # (h, w, 2)

def mask_to_pos_naive(mask):
    pos = sq(*image_dims)
    weight = pos[..., 0] + pos[..., 1]
    flat_idx = np.argmax((weight * mask).flatten())
    return flat_idx % image_dims[0], flat_idx // image_dims[0]

def get_gripper_pos_raw_np(img_np):
    img_pil = Image.fromarray(img_np)
    predictions = get_bounding_boxes(img_pil)
    if not predictions or predictions[0]["score"] < 0.05:
        return (-1, -1), np.zeros(image_dims), None
    mask = get_gripper_mask(img_pil, predictions[0])
    pos = mask_to_pos_naive(mask)
    return (int(pos[0]), int(pos[1])), mask, predictions[0]

def process_trajectory_hdf5(images, states):
    # Use ThreadPoolExecutor for I/O + SAM latency
    with ThreadPoolExecutor(max_workers=8) as executor:
        raw_trajectory = list(executor.map(lambda args: get_gripper_pos_raw_np(*args), zip(images)))

    prev_found = list(range(len(raw_trajectory)))
    next_found = list(range(len(raw_trajectory)))

    prev_found[0] = -1e6
    next_found[-1] = 1e6
    for i in range(1, len(raw_trajectory)):
        if raw_trajectory[i][2] is None:
            prev_found[i] = prev_found[i - 1]
    for i in reversed(range(len(raw_trajectory) - 1)):
        if raw_trajectory[i][2] is None:
            next_found[i] = next_found[i + 1]
    if next_found[0] == next_found[-1]:
        return None
    for i in range(len(raw_trajectory)):
        if raw_trajectory[i][2] is None:
            near_idx = prev_found[i] if i - prev_found[i] < next_found[i] - i else next_found[i]
            raw_trajectory[i] = raw_trajectory[near_idx]
    return [(tr[0], tr[1], state) for tr, state in zip(raw_trajectory, states)]

def get_corrected_positions(episode_id, episode, plot=False):
    images = episode["obs"]["agentview_rgb"]
    states = episode["obs"]["ee_states"]
    raw_traj = process_trajectory_hdf5(images, states)

    if raw_traj is None:
        raise RuntimeError(f"No gripper found in episode {episode_id}")

    pos_2d = np.array([tr[0] for tr in raw_traj], dtype=np.float32)
    pos_3d = np.array([tr[2][:3] for tr in raw_traj], dtype=np.float32)

    from sklearn.linear_model import RANSACRegressor
    points_3d_pr = np.concatenate([pos_3d, np.ones_like(pos_3d[:, :1])], axis=-1)
    points_2d_pr = np.concatenate([pos_2d, np.ones_like(pos_2d[:, :1])], axis=-1)

    reg = RANSACRegressor(random_state=0).fit(points_3d_pr, points_2d_pr)
    pr_pos = reg.predict(points_3d_pr)[:, :-1].astype(int)

    if plot:
        rendered = [
            cv2.circle(img.copy(), (int(p[0]), int(p[1])), radius=5, color=(255, 0, 0), thickness=-1)
            for img, p in zip(images, pr_pos)
        ]
        mediapy.show_video(rendered, fps=10)

    return pr_pos
