import json
import os
import re
import h5py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


from primitive_movements_2 import get_move_primitives_episode
from gripper_positions import get_corrected_positions


class LocalLLM:
    def __init__(self, model_name="teknium/OpenHermes-2.5-Mistral-7B", device="cuda"):
        print("🔄 Loading model:", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=torch.bfloat16)
        self.device = device

    def generate(self, prompt, max_new_tokens=1024):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def build_prompt(features, language_instruction, caption=None, list_only_moves=False):
    structured_features = "{\n"
    keys = list(features.keys())
    for i in range(len(features[keys[0]])):
        if list_only_moves:
            structured_features += f'    {i}: "{features["move_primitive"][i]}"\n'
        else:
            structured_features += f'    {i}: {{\n'
            for key in keys:
                val = features[key][i]
                val = f'"{val}"' if isinstance(val, str) else val
                structured_features += f'        "{key}": {val},\n'
            structured_features += "    },\n"
    structured_features += "}"

    features_desc = (
        "Each entry in that dictionary corresponds to a single step on the trajectory and describes the move that is "
        "about to be executed." if list_only_moves else
        "Each entry corresponds to a trajectory step. Features include:\n"
        '- "state_3d": 3D coordinates of the end effector\n'
        '- "move_primitive": movement type\n'
        '- "gripper_position": position in the 256x256 frame'
    )

    caption = f"\n\n## Scene description\n{caption}" if caption else ""
    break_line = ""

    return f"""# Annotate the training trajectory with reasoning

## Instruction
"{language_instruction}"

## Trajectory

```python
trajectory_features = {structured_features}
```

{features_desc}

{caption}## Your objective

I want you to annotate the given trajectory with reasoning. That is, for each step, I need to know not only {
break_line}which action should be chosen, but importantly what reasoning justifies that action choice. I want you to {
break_line}be descriptive and include all the relevant information available. The reasoning should include the task {
break_line}to complete, the remaining high-level steps, the high-level movements that should be executed and why they {
break_line}are required, the premises that allow inferring the direction of each move, including the locations of {
break_line}relevant objects, possible obstacles or difficulties to avoid, and any other relevant justification.

### Begin by describing the task

Start by giving an overview of the task. Make it more comprehensive than the simple instruction. Include the activity, {
break_line}the objects the robotic arm interacts with, and their relative locations in the environment. Then, describe {
break_line}the high-level movements that were most likely executed, based on the task that was completed and the {
break_line}primitive movements that were executed. Then, for each high-level movement write the interval of steps that {
break_line}movement consists of. Also, for each high-level movement write a justification for why it should be {
break_line}executed. Write an answer for this part using markdown and natural language. Be descriptive and highlight {
break_line}all the relevant details, but ensure that your description is consistent with the trajectory that was {
break_line}executed, specified by the features listed above in the `trajectory_features` dictionary.

### List the reasonings for each step

Finally, for each step describe the reasoning that allows to determine the correct action. For each step describe the {
break_line}remaining part of the objective, the current progress, the objects that are still relevant for determining {
break_line}the plan, and the plan for the next steps, based on the available features. Start the reasoning from a high {
break_line}level and gradually add finer features. I need you to be descriptive and very precise. Ensure that the {
break_line}reasoning is consistent with the task and the executed trajectory. Write the answer for this part as a {
break_line}Python-executable dictionary. For every step in the initial trajectory there should be exactly one separate {
break_line}item of the form <step id>:<reasoning>. Do not group the answers. The final dictionary should have exactly {
break_line}the same set of integer keys as the dictionary of features provided in the `trajectory_features` dictionary {
break_line}above. The reasoning should be a single string that describes the reasoning in natural language and {
break_line}includes all the required features.

Each reasoning string should have the following form:
- Describe the full task that remains to be completed (but only describe what remains), and place it inside a {
break_line}tag <task>.
- Describe the complete high-level plan for completing the remaining task (the list of remaining high-level steps), {
break_line}and place it inside a tag <plan>.
- Describe the high-level step that should be executed now (chosen from the list of high-level steps), and place it {
break_line}inside a tag <subtask>.
- Describe why the chosen high-level step should be executed now, which features of the current environment influence {
break_line}that decision, and how it should be done. Place it within a tag <subtask_reason>.
- Describe the current primitive movement of the arm that needs to be executed, and place it inside a tag <move>.
- Describe why the chosen movement should be executed now and which features of the current environment influence that {
break_line}decision. Place it inside a tag <move_reason>.

## Task summary

Here is a breakdown of what needs to be done:

- Describe the task.
- Describe the high-level movements that were executed, based on the completed task and the listed features.
- Describe the plan for the solution that allowed the robot to complete the task successfully.
- For each step on the trajectory, describe the reasoning that leads to determining the correct action. The reasoning {
break_line}should be descriptive and precise. You should provide exactly one reasoning string for each step on the {
break_line}trajectory specified by `trajectory_features`.
- At the very end of the response, write a single label FINISHED to indicate that the answer is complete."""

def find_task_occurrences(input_string, tags):
    pattern = r"(\d+):"
    for tag in tags:
        pattern += r"\s*<" + tag + r">([^<]*)</" + tag + ">"
    return re.findall(pattern, input_string)


def extract_reasoning_dict(reasoning_output, tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")):
    if reasoning_output is None:
        return dict()
    return {
        int(match[0]): dict(zip(tags, match[1:]))
        for match in find_task_occurrences(reasoning_output, tags)
    }


def get_reasoning_dict(features, metadata, lm):
    prompt = build_prompt(features, metadata["language_instruction"], caption=metadata.get("caption", ""), list_only_moves=True)
    print("🧠 Prompt built. Generating reasoning...")
    reasoning_output = lm.generate(prompt)
    print("📥 Output received.")
    return extract_reasoning_dict(reasoning_output)


def load_episode_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        return {
            "obs": {
                "ee_states": f["obs/ee_states"][:],
                "gripper_states": f["obs/gripper_states"][:],
                "agentview_rgb": f["obs/agentview_rgb"][:]
            },
            "actions": f["actions"][:],
            "teleop_actions": f["teleop_actions"][:],
            "metadata": {
                "language_instruction": f.attrs["language_instruction"],
                "episode_id": str(f.attrs.get("episode_id", 0)),
                "file_path": str(h5_path)
            }
        }


def build_single_reasoning_h5(h5_path, lm, captions):
    episode = load_episode_h5(h5_path)
    obs = episode["obs"]

    ft = {
        "state_3d": obs["ee_states"][:, :3].tolist(),
        "move_primitive": [move[0] for move in get_move_primitives_episode(episode)],
        "gripper_positions": get_corrected_positions(
            episode["metadata"]["episode_id"],
            episode,
            plot=False
        ),
    }

    mt = episode["metadata"]
    mt["caption"] = captions[mt["file_path"]][mt["episode_id"]]["caption"]

    reasoning = get_reasoning_dict(ft, mt, lm)
    return {"reasoning": reasoning, "features": ft, "metadata": mt}


def generate_reasonings(h5_file_paths, save_path="reasonings.json"):
    reasonings = {}
    lm = LocalLLM()

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            reasonings = json.load(f)

    with open("results_0.json", "r") as captions_file:
        captions_dict = json.load(captions_file)

    for h5_path in h5_file_paths:
        entry = build_single_reasoning_h5(h5_path, lm, captions_dict)
        file_key, ep_id = entry["metadata"]["file_path"], entry["metadata"]["episode_id"]
        reasonings.setdefault(file_key, {})[ep_id] = entry
        print(f"✅ Reasoning saved for {file_key} episode {ep_id}")

    with open(save_path, "w") as out_f:
        json.dump(reasonings, out_f, indent=2)


if __name__ == "__main__":
    h5_dir = "/l/users/malak.mansour/Datasets/do_manual/hdf5"
    h5_files = [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith(".h5")]
    generate_reasonings(h5_files)