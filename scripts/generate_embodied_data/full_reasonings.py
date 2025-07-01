# python scripts/generate_embodied_data/full_reasonings.py


import json
import os
import re
import torch
import h5py
import numpy as np

from primitive_movements_2 import get_move_primitives_episode
from gripper_positions import get_corrected_positions

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tensorflow as tf  # Add at the top if not already

class LocalLLM:
    def __init__(self, model_name="teknium/OpenHermes-2.5-Mistral-7B", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=torch.float16
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt):
        outputs = self.pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7)
        return outputs[0]["generated_text"]


def build_prompt(features, language_instruction, caption=None, list_only_moves=False):
    structured_features = "{\n"

    keys = list(features.keys())

    for i in range(len(features[keys[0]])):
        if list_only_moves:
            structured_features = structured_features + f'    {i}: "{features["move_primitive"][i]}"\n'
        else:
            structured_features = structured_features + f'    {i}: {"{"}\n'

            for key in keys:
                feature_value = features[key][i]
                if isinstance(feature_value, str):
                    feature_value = f'"{feature_value}"'

                structured_features = structured_features + f'        "{key}": {feature_value},\n'

            structured_features = structured_features + "    },\n"

    structured_features = structured_features + "}"

    if list_only_moves:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on the "
            "trajectory and describes the move that is about to be executed."
        )
    else:
        features_desc = (
            "Each entry in that dictionary corresponds to a single step on "
            "the trajectory. The provided features are the following:\n"
            "\n"
            '- "state_3d" are the current 3d coordinates of the robotic arm end effector; '
            "moving forward increases the first coordinate; moving left increases the second "
            "coordinate; moving up increases the third coordinate,\n"
            '- "move_primitive" describes the move that is about to be executed,\n'
            '- "gripper_position" denotes the location of the gripper in the 256x256 image observation'
        )

    if caption is None:
        caption = ""
    else:
        caption = f"""## Scene description

The robot is operating in the following environment. {caption}

"""

    break_line = ""  # for line formatting

    return f"""# Annotate the training trajectory with reasoning

## Specification of the experimental setup

You're an expert reinforcement learning researcher. You've trained an optimal policy for controlling a robotic arm. The
robot successfully completed a task specified by the instruction: "{language_instruction}". For that purpose, the
robotic arm executed a sequence of actions. Consecutive moves that were executed are the following:


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
        pattern = pattern + r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"

    matches = re.findall(pattern, input_string)
    return matches


def extract_reasoning_dict(reasoning_output, tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason")):
    if reasoning_output is None:
        return dict()

    trajectory = dict()

    matches = find_task_occurrences(reasoning_output, tags)

    for match in matches:
        trajectory[int(match[0])] = dict(zip(tags, match[1:]))

    return trajectory


def get_reasoning_dict(features, metadata, lm):
    language_instruction = metadata["language_instruction"]
    caption = metadata["caption"] if "caption" in metadata.keys() else None

    prompt = build_prompt(features, language_instruction, caption=caption, list_only_moves=True)
    print("metadata:", metadata, "\nprompt:", prompt)

    reasoning_output = lm.generate(prompt)

    print("reasoning:", reasoning_output)

    return extract_reasoning_dict(reasoning_output)


def build_single_reasoning(h5_path, lm, captions):
    entries = {}
    with h5py.File(h5_path, "r") as f:
        ep_keys = list(f.keys())
        for ep_key in ep_keys:
            episode = f[ep_key]
            obs = episode["obs"]
            ee_states = obs["ee_states"][:, :3]
            gripper = obs["gripper_states"][:, 0]
            actions = episode["teleop_actions"][:, :3]  # relative movement

            lang_instr = ep_key.replace("_", " ")

            # Construct BridgeV2-style episode
            ep_struct = {
                "steps": [],
                "episode_metadata": {
                    "episode_id": ep_key,
                    "file_path": os.path.basename(h5_path),
                }
            }

            # for i in range(len(ee_states)):
            #     step = {
            #         "observation": {
            #             "state": ee_states[i],
            #         },
            #         "action": actions[i],  # ✅ KEY FIX
            #         "language_instruction": lang_instr.encode(),
            #     }
            #     ep_struct["steps"].append(step)
            for i in range(len(ee_states)):
                step = {
                    "observation": {
                        "state": tf.convert_to_tensor(ee_states[i]),  # ✅ Make it compatible
                    },
                    "action": tf.convert_to_tensor(actions[i]),       # ✅ Fix this line
                    "language_instruction": lang_instr.encode(),
                }
                ep_struct["steps"].append(step)


            # Now compatible with get_move_primitives_episode()
            move_primitives = get_move_primitives_episode(ep_struct)
            ft = {
                "state_3d": ee_states.tolist(),
                "move_primitive": [m[0] for m in move_primitives],
                "gripper_positions": get_corrected_positions(ep_key, h5_path, plot=False),
            }

            mt = {
                "episode_id": ep_key,
                "file_path": os.path.basename(h5_path),
                "n_steps": len(ee_states),
                "language_instruction": lang_instr,
                "caption": captions.get(h5_path, {}).get(ep_key, {}).get("caption", "")
            }

            reasoning = get_reasoning_dict(ft, mt, lm)
            entries[ep_key] = {"reasoning": reasoning, "features": ft, "metadata": mt}

    return entries

def generate_reasonings(dataset_paths, lm, save_path="reasonings.json"):
    reasonings = dict()
    # lm = Gemini()

    if os.path.exists(save_path):
        print(save_path, "existing, loading contents")
        with open(save_path, "r") as f:
            reasonings = json.load(f)

        print("loaded reasonings:", sum([len(v) for v in reasonings.values()]), "entries")

    with open("results_0.json", "r") as captions_file:
        captions_dict = json.load(captions_file)

    for h5_path in dataset_paths:
        episode_entries = build_single_reasoning(h5_path, lm, captions_dict)
        for ep_id, entry in episode_entries.items():
            file_key = entry["metadata"]["file_path"]
            if file_key not in reasonings:
                reasonings[file_key] = {}
            reasonings[file_key][ep_id] = entry

    with open(save_path, "w") as out_f:
        json.dump(reasonings, out_f)


if __name__ == "__main__":
    # builder = tfds.builder_from_directory(builder_dir='<TODO: Enter path to BridgeV2>')
    # episode_ids = range(53192)  # All training episodes

    import glob
    dataset_paths = sorted(glob.glob("/l/users/malak.mansour/Datasets/do_manual/hdf5/*.h5"))

    # NOTE the generator expects the captions.json file to be present in the working directory
    # The captions should be generated using the script in
    # scripts/generate_embodied_data/bounding_boxes/generate_descriptions.py
    # generate_reasonings(builder, episode_ids)
    reasonings = {}
    lm = LocalLLM()

    with open("results_0.json", "r") as captions_file:
        captions_dict = json.load(captions_file)

    generate_reasonings(dataset_paths, lm)
