import json
from openai import OpenAI

client = OpenAI(api_key="your api", base_url="your url")


def generate_prompts(prompts, num_prompts):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"{prompts}"}],
        max_tokens=400 * num_prompts,
    )

    response = response.choices[0].message.content.strip().split("\n")
    augmented_prompts = []

    for line in response:
        try:
            # 去除可能的序号和多余符号
            clean_line = line.split(":", 1)[-1].strip().strip('"')
            prompt_dict = json.loads(clean_line)
            augmented_prompts.append(prompt_dict)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            try:
                json_str = line[line.find("{") : line.rfind("}") + 1]
                prompt_dict = json.loads(json_str)
                augmented_prompts.append(prompt_dict)
            except:
                pass

    return augmented_prompts


def augment_prompts_from_samples(prompts, num_prompts):
    combined_prompts = " ".join([json.dumps(prompt) for prompt in prompts])
    input_prompt = f"""You are a prompt generation expert. Please generate {num_prompts} different prompts based on the following existing prompts:
{combined_prompts}

    Requirements:
    1. The generated scenes must be based on the overall context of these scenes
    2. Each object's description should only contain its own visual attributes, without interactions with other objects or the environment
    3. Each scene must have distinct detail variations, and each object's prompt must be detailed
    4. Each non-background object must retain the same base color as in the original prompt, ensuring a consistent color theme.
    5. The background color must not contain colors mentioned in the descriptions of other objects
    6. Maintain the original JSON format and output JSON objects directly without additional explanations. Each prompt should be output on a single line."""

    return input_prompt


def generate_from_input(fore_objs, background, negative, scene, num_prompts):
    format = fore_objs.copy()
    format["background"] = background
    format["negative"] = negative
    output_format = {k: "" for k in format.keys()}
    input_prompt = f"""You are a prompt generation expert. Please generate {num_prompts} different prompts based on the following elements and their base descriptions:
    {json.dumps(format)};

    Requirements:
    1. The basic environment description is {scene}, and the overall scene must not deviate from this basic description
    2. Each object's description should only contain its own visual attributes, without interactions with other objects or the environment
    3. Each scene must have distinct detail variations, and each object's prompt must be detailed
    4. The color of background must not contain colors mentioned in the descriptions of {fore_objs.keys()}
    5. Output directly in the same format as {json.dumps(output_format)}. Each prompt should be output on a single line."""

    return input_prompt


def read_prompts(file_path):
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def write_prompts(file_path, prompts):
    with open(file_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + "\n")


def main(mode, **kwargs):
    if mode == "example":
        original_prompts = read_prompts(input_path)
        input_prompt = augment_prompts_from_samples(original_prompts, num_prompts)
    elif mode == "input":
        input_prompt = generate_from_input(
            fore_objs=kwargs["fore_objs"],
            background=kwargs["background"],
            negative=kwargs["negative"],
            scene=kwargs["scene"],
            num_prompts=kwargs["num_prompts"],
        )
    results = generate_prompts(input_prompt, num_prompts)
    write_prompts(kwargs["output_path"], results)


if __name__ == "__main__":

    import os
    from discoverse import DISCOVERSE_ROOT_DIR

    mode = "example"  # or 'example'
    task_name = "block_place"
    input_path = os.path.join(
        DISCOVERSE_ROOT_DIR, f"discoverse/randomain/prompts/{task_name}/example.jsonl"
    )
    output_path = os.path.join(
        DISCOVERSE_ROOT_DIR, f"discoverse/randomain/prompts/{task_name}/prompts.jsonl"
    )
    num_prompts = 50

    fore_objs = {
        "block_green": "A green block",
        "bowl_pink": "A pink bowl",
        "robot": "A black robot arm",
    }
    background = "A table"
    scene = (
        "In one room, a robotic arm is doing the task of clipping a block into a bowl"
    )
    negative = "No extra objects in the scene"

    main(
        mode=mode,
        fore_objs=fore_objs,
        background=background,
        negative=negative,
        scene=scene,
        input_path=input_path,
        output_path=output_path,
        num_prompts=num_prompts,
    )
