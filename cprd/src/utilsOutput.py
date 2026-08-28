import glob
import json
import re
import time

import litellm
import math
import os

import matplotlib.pyplot as plt

from utilsLiteLLM import CachedLLM

# Fixed categorical order (colorblind-validated), reused so a given model
# always maps to the same color across plots regardless of which subset of
# models ran for a given condition.
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]


# def format_logprobs(response):
#     results = {}
#     for idx, choice in enumerate(response.choices):
#         content = choice.logprobs.content
#         rows = []
#         for pos, token in enumerate(content):
#             row = {
#                 "token": repr(token.token),
#                 "prob": f"{math.exp(token.logprob):.4%}",
#                 "top": [
#                     f"{repr(t.token)}: {math.exp(t.logprob):.4%}"
#                     for t in token.top_logprobs
#                 ],
#             }
#             rows.append(row)
#         results[f'sample-{idx}'] = rows
#     return results
#
#
# def print_logprobs(response):
#     logprobs = format_logprobs(response)
#     for sample, rows in logprobs.items():
#         print(f'\n{sample}')
#         for pos, row in enumerate(rows):
#             print(f"\nToken {pos}: {row['token']}  (p={row['prob']})")
#             print("  Alternatives:")
#             for alt in row["top"]:
#                 print(f"    {alt}")


def _reconstruct_number_probs(content_text: str,
                              positions: list,
                              choices: list,
                              ):
    """Turn per-position top-k logprobs into a probability per `choice`,
    handling both tokenizer behaviors seen across providers:

    - atomic: the generated answer is spelled out in a single token at
      position 0 (e.g. GPT emits "20" as one token). Detected by the chosen
      token at position 0 matching the full answer text exactly. In this
      case every top-k alternative at position 0 already IS a complete
      candidate answer, so its probability is used as-is (this reproduces
      the original position-0-only behavior exactly for these models).
    - split-digit: the answer is spelled out digit-by-digit across several
      positions (e.g. Qwen/Apertus emit "2" then "0" for "20"). Detected by
      the chosen token at position 0 NOT matching the full answer text. Here
      only the actually-sampled first-digit branch has an observed
      position-1 distribution, so only that branch's two-digit (or
      terminated single-digit) probabilities can be reconstructed as a
      sequence product; the other top-k alternatives at position 0 have no
      known continuation and are left unresolved (not fabricated).

    `positions` is a list of (chosen_token, chosen_logprob, [(alt_token,
    alt_logprob), ...]) tuples, one per generated position, oldest first.
    """
    d = dict.fromkeys(choices)
    if not positions:
        return d

    pos0_tok, _, pos0_alts = positions[0]
    atomic = content_text.strip() == pos0_tok.strip()

    if atomic:
        for tok, lp in pos0_alts:
            tok_s = tok.strip()
            if tok_s in choices:
                d[tok_s] = (d[tok_s] or 0.0) + math.exp(lp)
        return d

    # split-digit tokenizer: reconstruct only the sampled first-digit branch
    for tok, lp in pos0_alts:
        tok_s = tok.strip()
        if tok != pos0_tok or not tok_s.isdigit() or len(tok_s) != 1 or len(positions) <= 1:
            continue  # not the sampled branch, or no observed continuation -> unresolved
        prob = math.exp(lp)
        _, _, pos1_alts = positions[1]
        for tok2, lp2 in pos1_alts:
            tok2_s = tok2.strip()
            prob2 = prob * math.exp(lp2)
            if tok2_s.isdigit() and len(tok2_s) == 1:
                two = tok_s + tok2_s
                if two in choices:
                    d[two] = (d[two] or 0.0) + prob2
            else:
                # anything non-digit after a single digit means the digit
                # was itself the complete (terminated) answer
                d[tok_s] = (d[tok_s] or 0.0) + prob2

    return d


def get_probs_task(response,
                   task: str,
                   output_format: int = 1,
                   ):
    assert task in ['svo', 'cpr', 'risk']

    choices = []
    if task == 'svo':
        choices = [str(i) for i in range(1, 10)]
    elif task == 'risk':
        choices = [str(i) for i in range(1, 7)]
    elif task == 'cpr':
        choices = [str(i) for i in range(0, 31)]
    elif task == 'cg':
        if output_format == 1:
            choices = ['X', 'Y']
        elif output_format == 2:
            choices = [str(i) for i in range(4)]


    probs = {}
    for idx, c in enumerate(response.choices):
        # print(c)
        if c.message.content in choices:
            # print(f'Valid response. Response: {c.message.content}')
            lp = getattr(c, "logprobs", None)
            if lp is None:  # provider returned no logprobs at all for this response
                raise RuntimeError(f"No logprobs returned for sample-{idx}; provider/model does not support logprobs")
            if lp.content is None:  # this case is used for huggingface/together/meta-llama/Llama-3.3-70B-Instruct-Turbo
                positions = [
                    (str(tok), None, [(str(t), v) for t, v in lp.top_logprobs[i].items()])
                    for i, tok in enumerate(lp.tokens)
                ]
            else:
                positions = [
                    (str(t.token), t.logprob, [(str(a.token), a.logprob) for a in t.top_logprobs])
                    for t in lp.content
                ]

            first_token = positions[0][0] if positions else None
            if first_token in choices:
                d = _reconstruct_number_probs(c.message.content, positions, choices)
            else:
                d = dict.fromkeys(choices)

            probs[f'sample-{idx}'] = d
        else:
            print(f'Invalid response. Response: {c.message.content}')

    return probs


def get_probs_task_sampled(response,
                           task: str,
                           output_format: int = 1,
                           ):
    """Same output shape as get_probs_task, but for models with no logprobs
    access: each returned choice becomes a one-hot indicator (1.0 for the
    picked answer, 0.0 for every other) instead of a partial, possibly-None
    probability vector. Averaging one-hot draws across many independent
    samples (see generate_responses) yields the empirical selection
    frequency per choice, the sampling-based analog of the logprob average.
    """
    assert task in ['svo', 'cpr', 'risk']

    choices = []
    if task == 'svo':
        choices = [str(i) for i in range(1, 10)]
    elif task == 'risk':
        choices = [str(i) for i in range(1, 7)]
    elif task == 'cpr':
        choices = [str(i) for i in range(0, 31)]
    elif task == 'cg':
        if output_format == 1:
            choices = ['X', 'Y']
        elif output_format == 2:
            choices = [str(i) for i in range(4)]

    probs = {}
    for idx, c in enumerate(response.choices):
        if c.message.content in choices:
            d = {choice: 0.0 for choice in choices}
            d[c.message.content] = 1.0
            probs[f'sample-{idx}'] = d
        else:
            print(f'Invalid response. Response: {c.message.content}')

    return probs


def generate_responses(llm: CachedLLM,
                       messages: list[dict],
                       nsample: int,
                       task: str,
                       output_format: int = 1,
                       print_time: bool = False,
                       **kwargs,
                       ) -> litellm.ModelResponse:
    """Draw `nsample` independent responses and turn each into a per-choice
    probability dict, in the shape plot_output_averages/plot_combined_models
    expect.

    If `kwargs` requests logprobs (the model supports them), one call
    already yields the full per-choice probability distribution via
    get_probs_task, so `nsample` is forced to 1 regardless of what's
    passed in — extra outer calls would just be redundant, wasted requests.

    If the model has no logprobs access (kwargs has no truthy "logprobs"),
    each call returns only a single chosen answer, so `nsample` (as passed
    in) independent calls are made and merged into one dict of one-hot draws
    (sample-0..sample-{nsample-1}) rather than one dict per call — matching
    the {sample-i: {choice: prob}} shape that plot_output_averages /
    plot_combined_models read from output[key][0], and letting every draw
    contribute to the average. Set --nsample explicitly (e.g. --nsample=100)
    to get a meaningful read on the distribution's shape — with the default
    of 1 you get a single one-hot draw, not a distribution.
    """
    results = []
    t0 = time.perf_counter()

    use_logprobs = bool(kwargs.get("logprobs"))
    call_kwargs = kwargs if use_logprobs else {
        k: v for k, v in kwargs.items() if k not in ("logprobs", "top_logprobs")
    }
    effective_nsample = 1 if use_logprobs else nsample

    if use_logprobs:
        for i in range(effective_nsample):
            response = llm.complete(
                messages=messages,
                sample_idx=i,  # whatever integer, just needs to be different from the previous call to miss the cache
                **call_kwargs,
            )
            results.append(get_probs_task(response, task, output_format))
    else:
        merged = {}
        for i in range(effective_nsample):
            response = llm.complete(
                messages=messages,
                sample_idx=i,
                **call_kwargs,
            )
            draw = get_probs_task_sampled(response, task, output_format)
            for d in draw.values():
                merged[f'sample-{len(merged)}'] = d
        results.append(merged)

    if print_time:
        t1 = time.perf_counter()
        mode = "logprobs (n forced to 1)" if use_logprobs else f"sampled (n={effective_nsample})"
        print(f"time {(t1 - t0) * 1000:.0f} ms [{mode}]")

    return results


def construct_messages(system_prompt: str,
                       user_prompt: str,
                       ) -> list[dict]:
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def get_cg_output(output_format_text: str,
                  system_prompt: str,
                  instructions_prompt: str,
                  llm: CachedLLM,
                  nsample: int,
                  print_time: bool = False,
                  **kwargs,
                  ):
    role_and_nodes = {'A': ['1', '3', '5'], 'B': ['2', '4', '6']}
    cg_output = {}
    for role, nodes in role_and_nodes.items():
        tmp = {}
        for node in nodes:
            out_format = output_format_text.format(role=role, node=node)
            # print(out_format)

            messages = construct_messages(system_prompt, instructions_prompt + out_format)  # !!

            results = generate_responses(llm, messages, nsample,
                                         task='cg', output_format=1,
                                         print_time=print_time,
                                         **kwargs)
            tmp[node] = results

        cg_output[f"role{role}"] = tmp
    return cg_output


def plot_output_averages(output: dict,
                         output_path: str,
                         images_dir: str,
                         default_value: int = None,
                         ) -> str:
    """Plot the per-choice average probability (across samples) as a barplot
    for each key in `output`, one subplot per key, and save next to
    `images_dir` using the same base filename as `output_path` (with .png).
    If `default_value` is given, it is shown in the title and marked with a
    red dashed vertical line at the corresponding choice.
    """
    keys = list(output.keys())

    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4), squeeze=False)
    axes = axes[0]

    for ax, key in zip(axes, keys):
        samples = output[key][0]
        choices = list(next(iter(samples.values())).keys())
        averages = []
        for choice in choices:
            values = [sample[choice] for sample in samples.values() if sample[choice] is not None]
            averages.append(sum(values) / len(values) if values else 0.0)

        positions = range(len(choices))
        ax.bar(positions, averages)
        tick_step = 5 if len(choices) > 10 else 1
        ax.set_xticks(positions[::tick_step], choices[::tick_step])

        title = key
        if default_value is not None and str(default_value) in choices:
            default_pos = choices.index(str(default_value))
            ax.axvline(default_pos, color="red", linestyle="--")
            title = f"{key} (default={default_value})"
        ax.set_title(title)
        ax.set_xlabel("choice")
        ax.set_ylabel("average probability")

    fig.tight_layout()

    base_name = os.path.splitext(os.path.basename(output_path))[0]
    image_path = os.path.join(images_dir, f"{base_name}.png")
    fig.savefig(image_path)
    plt.close(fig)

    return image_path


# Cycled in after the 8 validated hues are exhausted, so a 9th+ series reuses
# a hue with a different line style instead of a generated (unvalidated) color.
LINESTYLES = ["-", "--", ":", "-."]


def model_color_map(model_ids: list[str]) -> dict[str, tuple[str, str]]:
    """Assign each model a fixed (color, linestyle) pair, in the given order,
    so a model keeps the same styling across every combined plot regardless
    of which subset of models actually ran for a given condition. Only the
    8 validated categorical hues are ever used; past 8 models, hues repeat
    with a different linestyle rather than a generated, unvalidated color.
    """
    n_colors = len(CATEGORICAL_COLORS)
    return {
        model_id: (CATEGORICAL_COLORS[i % n_colors], LINESTYLES[(i // n_colors) % len(LINESTYLES)])
        for i, model_id in enumerate(model_ids)
    }


def plot_combined_models(condition_dir: str,
                         task_suffix: str,
                         version: str,
                         nsample: int,
                         system_type: str,
                         images_dir: str,
                         all_model_ids: list[str],
                         default_value: int = None,
                         group_ext_value: int = None,
                         payoff_value: int = None,
                         ) -> str | None:
    """Overlay every model's per-choice average probability curve for one
    experimental condition (task_suffix encodes default/group_ext/payoff, if
    any) on a single reference plot, colored by model. Returns the saved
    image path, or None if no matching output files were found.
    """
    suffix = f"-{task_suffix}-v{version}-n{nsample}-{system_type}.json"
    paths = sorted(glob.glob(os.path.join(condition_dir, f"*{suffix}")))
    if not paths:
        return None

    colors = model_color_map(all_model_ids)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    choices = None
    for path in paths:
        model_id = os.path.basename(path)[:-len(suffix)]
        with open(path, encoding="utf-8") as f:
            output = json.load(f)
        samples = output["cpr"][0]
        if choices is None:
            choices = list(next(iter(samples.values())).keys())
        averages = []
        for choice in choices:
            values = [sample[choice] for sample in samples.values() if sample[choice] is not None]
            averages.append(sum(values) / len(values) if values else 0.0)

        color, linestyle = colors.get(model_id, ("#898781", "-"))
        ax.plot(range(len(choices)), averages, marker="o", markersize=8,
               markeredgecolor="#fcfcfb", markeredgewidth=1.5, linestyle=linestyle,
               linewidth=2, label=model_id, color=color, zorder=3)

    positions = list(range(len(choices)))
    tick_step = 5 if len(choices) > 10 else 1
    ax.set_xticks(positions[::tick_step], choices[::tick_step])

    if default_value is not None and str(default_value) in choices:
        ax.axvline(choices.index(str(default_value)), color="#898781",
                  linestyle="--", linewidth=1.5, zorder=2,
                  label=f"default={default_value}")

    condition_parts = []
    if default_value is not None:
        condition_parts.append(f"default={default_value}")
    if group_ext_value is not None and payoff_value is not None:
        condition_parts.append(f"group_ext={group_ext_value}, payoff={payoff_value}")
    condition_label = ", ".join(condition_parts) if condition_parts else "no default, no group extraction"

    ax.set_title(f"Average extraction probability by model ({condition_label})", color="#0b0b0b")
    ax.set_xlabel("choice", color="#52514e")
    ax.set_ylabel("average probability", color="#52514e")
    ax.tick_params(colors="#898781")
    ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor="#0b0b0b")

    fig.tight_layout()

    image_name = f"combined-{task_suffix}-v{version}-n{nsample}-{system_type}.png"
    image_path = os.path.join(images_dir, image_name)
    fig.savefig(image_path)
    plt.close(fig)

    return image_path


RUN_NAME_RE = re.compile(
    r"^(?P<model_id>.+?)-(?P<task_suffix>(?P<task>svo|risk|cpr)(?:-\S+?)?)"
    r"-v(?P<version>[^-]+)-n(?P<nsample>\d+)-(?P<system_type>blinded|unblinded|unblinded-framed)\.json$"
)


def parse_run_filename(filename: str) -> dict | None:
    """Parse an output JSON's basename into its run metadata (model, task,
    condition, version, nsample, system_type), or None if it doesn't match
    the naming convention written by main.py.
    """
    m = RUN_NAME_RE.match(filename)
    if m is None:
        return None
    info = m.groupdict()
    info["condition"] = condition_label(info["task_suffix"], info["task"])
    return info


def load_run_row(path: str, task: str) -> tuple[list[str], list[float], str]:
    """Load one run's per-choice average probabilities, plus a `notes` string
    flagging anything worth a human's attention: an unreadable file, no
    valid samples at all (every response was invalid/unparsed), or fewer
    valid samples than the filename's nsample promises (some draws failed).
    Returns (choices, averages, notes); choices/averages are [] when there is
    nothing to plot.
    """
    try:
        with open(path, encoding="utf-8") as f:
            output = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return [], [], f"unreadable file: {e}"

    task_output = output.get(task)
    if not task_output:
        return [], [], f"no '{task}' key in output"

    samples = task_output[0]
    if not samples:
        return [], [], "no valid samples (all responses invalid/unparsed)"

    choices = list(next(iter(samples.values())).keys())
    averages = []
    for choice in choices:
        values = [s[choice] for s in samples.values() if s[choice] is not None]
        averages.append(sum(values) / len(values) if values else 0.0)

    notes = ""
    m = re.search(r"-n(\d+)-", os.path.basename(path))
    if m:
        expected_n = int(m.group(1))
        if len(samples) < expected_n:
            notes = f"only {len(samples)}/{expected_n} valid samples"

    return choices, averages, notes


def condition_label(task_suffix: str, task: str) -> str:
    """Turn a filename task_suffix (e.g. 'cpr-default=5-group_ext=0-payoff=20')
    into a human-readable condition label ('default=5, group_ext=0, payoff=20').
    """
    rest = task_suffix[len(task):].lstrip("-")
    return ", ".join(rest.split("-")) if rest else "no default, no group extraction"


def plot_model_conditions(model_id: str,
                          task: str,
                          output_root: str,
                          images_dir: str,
                          version: str = None,
                          system_type: str = None,
                          ) -> str | None:
    """Overlay every experimental condition run so far for a single model
    (plain / default=.../group_ext=.../payoff=...) on one plot, colored by
    condition. Searches `output_root` recursively for any JSON file whose
    name starts with '{model_id}-{task}', so it picks up runs regardless of
    which output_subdir they were saved into and independent of whether
    run_all_models.sh / experiment_maker.py produced them. If `version` or
    `system_type` are given, only files matching those are included; runs
    with different --nsample are all included (nsample only affects sampling
    resolution, not the shape of the plotted curve).
    """
    name_re = re.compile(
        rf"^{re.escape(model_id)}-(?P<task_suffix>{re.escape(task)}.*)"
        rf"-v(?P<version>[^-]+)-n(?P<nsample>\d+)-(?P<system_type>blinded|unblinded|unblinded-framed)\.json$"
    )

    pattern = os.path.join(output_root, "**", f"{model_id}-{task}*.json")
    matches = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        m = name_re.match(os.path.basename(path))
        if m is None:
            continue
        if version is not None and m.group("version") != version:
            continue
        if system_type is not None and m.group("system_type") != system_type:
            continue
        matches.append((path, m.group("task_suffix"), m.group("nsample"), m.group("system_type")))

    if not matches:
        return None

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    choices = None
    n_colors = len(CATEGORICAL_COLORS)
    for i, (path, task_suffix, nsample, sys_type) in enumerate(matches):
        with open(path, encoding="utf-8") as f:
            output = json.load(f)
        samples = output[task][0]
        if choices is None:
            choices = list(next(iter(samples.values())).keys())
        averages = []
        for choice in choices:
            values = [sample[choice] for sample in samples.values() if sample[choice] is not None]
            averages.append(sum(values) / len(values) if values else 0.0)

        color = CATEGORICAL_COLORS[i % n_colors]
        linestyle = LINESTYLES[(i // n_colors) % len(LINESTYLES)]
        label = f"{condition_label(task_suffix, task)} (n={nsample}, {sys_type})"
        ax.plot(range(len(choices)), averages, marker="o", markersize=8,
               markeredgecolor="#fcfcfb", markeredgewidth=1.5, linestyle=linestyle,
               linewidth=2, label=label, color=color, zorder=3)

    positions = list(range(len(choices)))
    tick_step = 5 if len(choices) > 10 else 1
    ax.set_xticks(positions[::tick_step], choices[::tick_step])

    ax.set_title(f"Average extraction probability by condition ({model_id})", color="#0b0b0b")
    ax.set_xlabel("choice", color="#52514e")
    ax.set_ylabel("average probability", color="#52514e")
    ax.tick_params(colors="#898781")
    ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor="#0b0b0b")

    fig.tight_layout()

    image_name = f"conditions-{model_id}-{task}.png"
    image_path = os.path.join(images_dir, image_name)
    fig.savefig(image_path)
    plt.close(fig)

    return image_path
