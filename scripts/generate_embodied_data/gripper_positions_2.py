import cv2
import matplotlib
import mediapy
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from transformers import SamModel, SamProcessor, pipeline

checkpoint = "google/owlvit-base-patch16"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

image_dims = (224, 224)
image_label = "image_0"


def get_bounding_boxes(img, prompt="the black robotic gripper"):
    predictions = detector(img, candidate_labels=[prompt], threshold=0.01)

    return predictions


def show_box(box, ax, meta, color):
    x0, y0 = box["xmin"], box["ymin"]
    w, h = box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
    ax.add_patch(
        matplotlib.patches.FancyBboxPatch((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2, label="hehe")
    )
    ax.text(x0, y0 + 10, "{:.3f}".format(meta["score"]), color="white")


def get_median(mask, p):
    row_sum = np.sum(mask, axis=1)
    cumulative_sum = np.cumsum(row_sum)

    if p >= 1.0:
        p = 1

    total_sum = np.sum(row_sum)
    threshold = p * total_sum

    return np.argmax(cumulative_sum >= threshold)


def get_gripper_mask(img, pred):
    box = [
        round(pred["box"]["xmin"], 2),
        round(pred["box"]["ymin"], 2),
        round(pred["box"]["xmax"], 2),
        round(pred["box"]["ymax"], 2),
    ]

    inputs = sam_processor(img, input_boxes=[[[box]]], return_tensors="pt")

    with torch.no_grad():
        outputs = sam_model(**inputs)

    mask = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
    )[0][0][0].numpy()

    return mask


def sq(w, h):
    return np.concatenate(
        [(np.arange(w * h).reshape(h, w) % w)[:, :, None], (np.arange(w * h).reshape(h, w) // w)[:, :, None]], axis=-1
    )


def mask_to_pos_weighted(mask):
    pos = sq(*image_dims)

    weight = pos[:, :, 0] + pos[:, :, 1]
    weight = weight * weight

    x = np.sum(mask * pos[:, :, 0] * weight) / np.sum(mask * weight)
    y = get_median(mask * weight, 0.95)

    return x, y


def mask_to_pos_naive(mask):
    pos = sq(*image_dims)
    weight = pos[:, :, 0] + pos[:, :, 1]
    min_pos = np.argmax((weight * mask).flatten())

    return min_pos % image_dims[0] - (image_dims[0] / 16), min_pos // image_dims[0] - (image_dims[0] / 24)


def get_gripper_pos(episode_id, frame, builder, plot=True):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))
    images = [step["observation"][image_label] for step in episode["steps"]]

    img = Image.fromarray(images[frame].numpy())
    predictions = get_bounding_boxes(img)

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(img)

        for prediction in predictions:
            if prediction["score"] < 0.05:
                continue
            box = prediction["box"]
            show_box(box, ax, prediction, "red")

    if len(predictions) > 0:
        mask = get_gripper_mask(img, predictions[0])
        pos = mask_to_pos_naive(mask)

        if plot:
            plt.imshow(mask, alpha=0.5)
            plt.scatter([pos[0]], [pos[1]])
    else:
        print("No valid bounding box")

    if plot:
        plt.show()


def get_gripper_pos_raw(img):
    img = Image.fromarray(img.numpy())
    predictions = get_bounding_boxes(img)

    if len(predictions) > 0:
        mask = get_gripper_mask(img, predictions[0])
        pos = mask_to_pos_naive(mask)
    else:
        mask = np.zeros(image_dims)
        pos = (-1, -1)
        predictions = [None]

    return (int(pos[0]), int(pos[1])), mask, predictions[0]


def process_trajectory_hdf5(images, states):
    """
    Args:
        images: list of (H, W, 3) NumPy images
        states: list of (6,) end-effector states
    Returns:
        List of tuples: (pos_2d, mask, state)
    """
    raw_trajectory = [(*get_gripper_pos_raw(torch.tensor(img)), state) for img, state in zip(images, states)]

    prev_found = list(range(len(raw_trajectory)))
    next_found = list(range(len(raw_trajectory)))

    prev_found[0] = -1e6
    next_found[-1] = 1e6

    for i in range(1, len(raw_trajectory)):
        if raw_trajectory[i][2] is None:
            prev_found[i] = prev_found[i - 1]

    for i in reversed(range(0, len(raw_trajectory) - 1)):
        if raw_trajectory[i][2] is None:
            next_found[i] = next_found[i + 1]

    if next_found[0] == next_found[-1]:
        return None  # gripper was never found

    for i in range(len(raw_trajectory)):
        raw_trajectory[i] = raw_trajectory[
            prev_found[i] if i - prev_found[i] < next_found[i] - i else next_found[i]
        ]

    return raw_trajectory


def get_corrected_positions(episode_id, episode, plot=False):
    """
    Args:
        episode_id: just for logging
        episode: dict with keys: obs/agentview_rgb, obs/ee_states
        plot: bool
    Returns:
        np.ndarray of shape (T, 2): estimated 2D gripper positions
    """

    images = episode["obs"]["agentview_rgb"]  # shape (T, H, W, 3)
    states = episode["obs"]["ee_states"]      # shape (T, 6)

    raw_traj = process_trajectory_hdf5(images, states)

    if raw_traj is None:
        raise RuntimeError(f"No gripper found in episode {episode_id}")

    pos_2d = np.array([tr[0] for tr in raw_traj], dtype=np.float32)
    pos_3d = np.array([tr[-1][:3] for tr in raw_traj], dtype=np.float32)

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
