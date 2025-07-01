import numpy as np

# Describes a movement vector with discrete direction names
def describe_move(move_vec):
    names = [
        {-1: "backward", 0: None, 1: "forward"},                  # Y
        {-1: "right", 0: None, 1: "left"},                        # X
        {-1: "down", 0: None, 1: "up"},                           # Z
        {-1: "tilt down", 0: None, 1: "tilt up"},                 # roll
        {},                                                       # pitch (unused directly)
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},  # yaw
        {-1: "close gripper", 0: None, 1: "open gripper"}         # gripper
    ]

    # Basic xyz movement
    xyz_move = [names[i][move_vec[i]] for i in range(0, 3)]
    xyz_move = [m for m in xyz_move if m is not None]

    description = "move " + " ".join(xyz_move) if xyz_move else ""

    # Handle rolling and tilting
    if move_vec[3] == 0:
        move_vec[3] = move_vec[4]  # use pitch if roll is 0

    if move_vec[3] != 0:
        if description:
            description += ", "
        description += names[3][move_vec[3]]

    # Add yaw rotation
    if move_vec[5] != 0:
        if description:
            description += ", "
        description += names[5][move_vec[5]]

    # Add gripper state
    if move_vec[6] != 0:
        if description:
            description += ", "
        description += names[6][move_vec[6]]

    if not description:
        description = "stop"

    return description


# Classifies a small movement into a discrete direction
def classify_movement(move, threshold=0.03):
    diff = move[-1] - move[0]  # from state[0] to state[-1]

    # Normalize translation (xyz)
    if np.sum(np.abs(diff[:3])) > 3 * threshold:
        diff[:3] *= 3 * threshold / np.sum(np.abs(diff[:3]))

    # Normalize rotation
    diff[3:6] /= 10

    # Convert to discrete movement vector (-1, 0, 1)
    move_vec = 1 * (diff > threshold) - 1 * (diff < -threshold)

    return describe_move(move_vec), move_vec


# Global dictionary to log movement->action mappings
move_actions = dict()


# New: Compatible with HDF5 episode format
def get_move_primitives_episode(episode, threshold=0.03):
    """
    Given a dict-style HDF5 episode with keys:
        episode["obs"]["ee_states"]: (N, 6)
        episode["obs"]["gripper_states"]: (N, 1)
    This function returns the list of (label, vector) for each primitive movement.
    """

    ee_states = np.array(episode["obs"]["ee_states"])               # (N, 6)
    gripper_states = np.array(episode["obs"]["gripper_states"]).squeeze(-1)  # (N,)
    states = np.concatenate([ee_states, gripper_states[:, None]], axis=1)   # (N, 7)

    # Sliding window of 4 states to classify movements
    move_trajs = [states[i : i + 4] for i in range(len(states) - 4)]

    primitives = [classify_movement(move, threshold) for move in move_trajs]
    primitives.append(primitives[-1])  # repeat last for alignment

    return primitives


# Deprecated for your case — TFDS version kept for reference
def get_move_primitives(episode_id, builder):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))
    return get_move_primitives_episode(episode)
