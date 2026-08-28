import argparse
import os

import pandas as pd
from tqdm import tqdm

from utilsLiteLLM import *
from utilsOutput import *


def main():
    # litellm._turn_on_debug()

    parser = argparse.ArgumentParser(description="test")

    parser.add_argument("--model", type=str, required=True, help="Name of the LLM")
    parser.add_argument("--task", type=str, required=True, help="Name of the task (svo, risk, cpr)")
    parser.add_argument("--version", type=str, required=True, help="Version of the prompt")
    parser.add_argument("--nsample", type=int, default=1, help="number of samples. For models without logprobs access, this is the number of independent draws used to build an empirical distribution (default 1)")
    parser.add_argument("--framing", action="store_true", help="framing flag (SVO framing on system prompt; only used with --task svo)")
    parser.add_argument("--blinding", action="store_true", help="blinding flag")
    parser.add_argument("--default", type=int, help="Default value")
    parser.add_argument("--group_ext", type=int, help="Group extraction value (only used with --task cpr)")
    parser.add_argument("--payoff", type=int, help="Payoff value (only used with --task cpr, together with --group_ext)")
    parser.add_argument("--output_subdir", type=str, default="", help="Subfolder of output/ to save the JSON and images into (used by experiment_maker.py)")

    args = parser.parse_args()

    model_id = args.model
    task = args.task
    version = args.version
    nsample = args.nsample
    framing = args.framing
    blinding = args.blinding
    default_value = args.default
    group_ext_value = args.group_ext
    payoff_value = args.payoff
    output_subdir = args.output_subdir

    if blinding:
        system_type = "blinded"
    else:
        system_type = "unblinded"
        if framing:
            system_type += "-framed"

    print(f"Model: {model_id}")
    print(f"Task: {task}")
    print(f"Prompt version: {version}")
    print(f"nb sample: {nsample}")

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")

    models_path = os.path.join(data_dir, "models.json")
    with open(models_path, 'r') as f:
        model_list = json.load(f)

    model_info = model_list[model_id]

    model = model_info['provider'] + '/' + model_info['model']
    kwargs = model_info['kwargs']

    print(litellm.get_llm_provider(model=model))
    print(litellm.get_supported_openai_params(model=model))

    llm = CachedLLM(
        model=model,
        db_path=os.path.join(project_dir, "llm_cache.db"),
        memory_cache=True,
    )

    prompts_path = os.path.join(project_dir, "prompts")

    system_prompt_path = os.path.join(prompts_path, "system", f"{system_type}-system-prompt.txt")
    with open(system_prompt_path, encoding='utf-8') as f:
        system = f.read()

    instructions_path = os.path.join(prompts_path, task, f"{task}-instructions-v{version}.txt")
    with open(instructions_path, encoding='utf-8') as f:
        instructions = f.read()

    output = {}

    if task == 'svo':
        for item in tqdm(range(1, 7)):
            print(f"item {item}\n")
            item_path = os.path.join(prompts_path, task, f"{task}-item{item}.txt")
            with open(item_path, encoding='utf-8') as f:
                list_choices = f.read()

            messages = construct_messages(system, instructions + list_choices)
            print(messages)

            results = generate_responses(llm, messages, nsample, task,
                                         print_time=False, **kwargs)

            output[f"item{item}"] = results

    elif task == 'risk':
        gambles_path = os.path.join(prompts_path, task, f"{task}-gambles.txt")
        with open(gambles_path, encoding='utf-8') as f:
            list_gambles = f.read()

        messages = construct_messages(system, instructions + list_gambles)
        print(messages)

        results = generate_responses(llm, messages, nsample, task, print_time=True, **kwargs)
        output['gambles'] = results
    
    elif task == 'cpr':
        cpr_path = os.path.join(prompts_path, task, f"{task}-table.txt")
        default_text = ""
        group_ext_text = ""
        with open(cpr_path, encoding='utf-8') as f:
            cpr_table = f.read()
        if default_value is not None:
            default_path = os.path.join(prompts_path, task, f"default-v0.txt")
            with open(default_path, encoding="utf-8") as f:
                default_text = f.read()
            default_text = default_text.replace("%i", str(default_value))
        if group_ext_value is not None and payoff_value is not None:
            group_ext_path = os.path.join(prompts_path, task, f"group_extraction-v0.txt")
            with open(group_ext_path, encoding="utf-8") as f:
                group_ext_text = f.read()
            group_ext_text = group_ext_text.replace("%x", str(group_ext_value)).replace("%y", str(payoff_value))
        parts = [instructions, cpr_table]
        if default_text:
            parts.append(default_text)
        if group_ext_text:
            parts.append(group_ext_text)
        messages = construct_messages(system, "\n\n".join(part.strip() for part in parts))
        print(messages)

        results = generate_responses(llm, messages, nsample, task, print_time=True, **kwargs)
        output['cpr'] = results
        if len(default_text) > 0:
            task = task + "-default=" + str(default_value)
        if len(group_ext_text) > 0:
            task = task + f"-group_ext={group_ext_value}-payoff={payoff_value}"
    

    output_file_name = f"{model_id}-{task}-v{version}-n{nsample}-{system_type}.json"

    output_dir = os.path.join(project_dir, "output", output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    print(f"Saved JSON to: {output_path}")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    image_path = plot_output_averages(output, output_path, images_dir, default_value=default_value)
    print(f"Saved plot to: {image_path}")


if __name__ == "__main__":
    main()
