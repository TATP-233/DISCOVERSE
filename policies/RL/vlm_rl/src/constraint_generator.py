"""
ConstraintGenerator: Use VLM to generate task constraints from images and instructions.

Calls GPT-4o to analyze the scene and generate constraint functions for RL training.
"""

import os
import json
import base64
import re
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class GenerationResult:
    """Result of constraint generation."""
    num_stages: int
    grasp_keypoints: List[int]      # keypoint index to grasp at each stage (-1 = none)
    release_keypoints: List[int]    # keypoint index to release at each stage (-1 = none)
    subgoal_constraints: Dict[int, List[str]]  # {stage: [constraint_code, ...]}
    path_constraints: Dict[int, List[str]]     # {stage: [constraint_code, ...]}
    raw_output: str
    output_dir: str


class ConstraintGenerator:
    """
    Generate task constraints using VLM (GPT-4o).

    Given an annotated scene image and task instruction, generates:
    - Task decomposition into stages
    - Subgoal constraints for each stage end
    - Path constraints for each stage
    - Grasp/release keypoint assignments
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        prompt_template_path: Optional[str] = None,
        cache_results: bool = True,
    ):
        """
        Args:
            model: OpenAI model name (default: gpt-4o)
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
            prompt_template_path: Path to prompt template file
            cache_results: Whether to cache VLM results
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.cache_results = cache_results

        # Load prompt template
        if prompt_template_path is None:
            # Use default prompt in prompts directory
            module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            prompt_template_path = os.path.join(module_dir, "prompts", "constraint_prompt.txt")

        self.prompt_template = self._load_prompt_template(prompt_template_path)

    def _load_prompt_template(self, path: str) -> str:
        """Load prompt template from file."""
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
        else:
            # Return default prompt if file doesn't exist
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Return default prompt template."""
        return '''You are a robot task planner. Given an image of a scene with numbered keypoints and a task instruction, you need to:

1. Analyze the scene and identify relevant objects
2. Decompose the task into sequential stages (grasping must be a separate stage)
3. Generate constraint functions for each stage

## Constraint Function Format

```python
def stage{N}_{type}_constraint{M}(end_effector, keypoints):
    """
    Brief description of what this constraint enforces.

    Args:
        end_effector: np.ndarray [3] - current end effector position
        keypoints: np.ndarray [K, 3] - all keypoint positions

    Returns:
        cost: float - constraint cost (<=0 means satisfied, >0 means violated)
    """
    # Your implementation
    return cost
```

## Constraint Types
- **subgoal**: Must be satisfied at the END of the stage
- **path**: Must be satisfied THROUGHOUT the stage

## Important Rules
1. Use only numpy operations (np.linalg.norm, np.dot, np.cross, np.mean, etc.)
2. Reference keypoints by index: keypoints[0], keypoints[3], etc.
3. Return a scalar cost value where negative/zero means satisfied
4. Grasping action must be its own stage (stage 1 typically)
5. Common patterns:
   - Distance: `np.linalg.norm(a - b) - threshold`
   - Height: `target_height - point[2]`
   - Alignment: `np.linalg.norm(np.cross(vec1, vec2))`

## Output Format

First provide your analysis, then output the following metadata and functions:

```
NUM_STAGES: {number}
GRASP_KEYPOINTS: [{list of keypoint indices, -1 for no grasp}]
RELEASE_KEYPOINTS: [{list of keypoint indices, -1 for no release}]
```

Then provide all constraint functions.

Task instruction: {instruction}
Keypoint description: {keypoint_description}
'''

    def generate(
        self,
        image: np.ndarray,
        instruction: str,
        keypoint_description: str,
        output_dir: str,
    ) -> GenerationResult:
        """
        Generate constraints from image and instruction.

        Args:
            image: RGB image with keypoint annotations (H, W, 3) uint8
            instruction: Natural language task instruction
            keypoint_description: Text description of keypoints
            output_dir: Directory to save results

        Returns:
            GenerationResult with parsed constraints
        """
        os.makedirs(output_dir, exist_ok=True)

        # Check for cached result
        cache_path = os.path.join(output_dir, "generation_result.json")
        if self.cache_results and os.path.exists(cache_path):
            print(f"Loading cached result from {cache_path}")
            return self._load_cached_result(cache_path, output_dir)

        # Prepare prompt
        prompt = self.prompt_template.format(
            instruction=instruction,
            keypoint_description=keypoint_description,
        )

        # Save prompt for debugging
        with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        # Save input image
        import cv2
        cv2.imwrite(
            os.path.join(output_dir, "input_image.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

        # Call VLM
        raw_output = self._call_vlm(image, prompt)

        # Save raw output
        with open(os.path.join(output_dir, "raw_output.txt"), "w") as f:
            f.write(raw_output)

        # Parse output
        result = self._parse_output(raw_output, output_dir)

        # Cache result
        if self.cache_results:
            self._save_cached_result(result, cache_path)

        return result

    def _call_vlm(self, image: np.ndarray, prompt: str) -> str:
        """Call OpenAI VLM API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        client = OpenAI(api_key=self.api_key)

        # Encode image to base64
        import cv2
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Call API
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )

        return response.choices[0].message.content

    def _parse_output(self, raw_output: str, output_dir: str) -> GenerationResult:
        """Parse VLM output into structured result."""
        # Extract metadata
        num_stages = self._extract_int(raw_output, r"NUM_STAGES:\s*(\d+)")
        grasp_keypoints = self._extract_int_list(raw_output, r"GRASP_KEYPOINTS:\s*\[([\d,\s-]+)\]")
        release_keypoints = self._extract_int_list(raw_output, r"RELEASE_KEYPOINTS:\s*\[([\d,\s-]+)\]")

        # Default values if not found
        if num_stages is None:
            num_stages = 1
        if grasp_keypoints is None:
            grasp_keypoints = [-1] * num_stages
        if release_keypoints is None:
            release_keypoints = [-1] * num_stages

        # Pad lists to match num_stages
        while len(grasp_keypoints) < num_stages:
            grasp_keypoints.append(-1)
        while len(release_keypoints) < num_stages:
            release_keypoints.append(-1)

        # Extract constraint functions
        subgoal_constraints: Dict[int, List[str]] = {}
        path_constraints: Dict[int, List[str]] = {}

        # Find all function definitions
        func_pattern = r"(def\s+stage(\d+)_(subgoal|path)_constraint\d+\s*\([^)]*\):.*?return\s+[^\n]+)"
        matches = re.findall(func_pattern, raw_output, re.DOTALL)

        for match in matches:
            func_code = match[0]
            stage = int(match[1])
            constraint_type = match[2]

            # Clean up the function code
            func_code = self._clean_function_code(func_code)

            if constraint_type == "subgoal":
                if stage not in subgoal_constraints:
                    subgoal_constraints[stage] = []
                subgoal_constraints[stage].append(func_code)
            else:
                if stage not in path_constraints:
                    path_constraints[stage] = []
                path_constraints[stage].append(func_code)

        # Save constraint files
        for stage in range(1, num_stages + 1):
            if stage in subgoal_constraints:
                self._save_constraints(
                    output_dir,
                    f"stage{stage}_subgoal_constraints.py",
                    subgoal_constraints[stage]
                )
            if stage in path_constraints:
                self._save_constraints(
                    output_dir,
                    f"stage{stage}_path_constraints.py",
                    path_constraints[stage]
                )

        return GenerationResult(
            num_stages=num_stages,
            grasp_keypoints=grasp_keypoints,
            release_keypoints=release_keypoints,
            subgoal_constraints=subgoal_constraints,
            path_constraints=path_constraints,
            raw_output=raw_output,
            output_dir=output_dir,
        )

    def _extract_int(self, text: str, pattern: str) -> Optional[int]:
        """Extract integer from text using regex pattern."""
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
        return None

    def _extract_int_list(self, text: str, pattern: str) -> Optional[List[int]]:
        """Extract list of integers from text."""
        match = re.search(pattern, text)
        if match:
            values = match.group(1).split(",")
            return [int(v.strip()) for v in values]
        return None

    def _clean_function_code(self, code: str) -> str:
        """Clean up extracted function code."""
        # Remove markdown code blocks if present
        code = re.sub(r"```python\s*", "", code)
        code = re.sub(r"```\s*", "", code)

        # Ensure proper indentation
        lines = code.split("\n")
        cleaned_lines = []
        for line in lines:
            # Remove leading spaces beyond function indentation
            stripped = line.rstrip()
            if stripped:
                cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines)

    def _save_constraints(self, output_dir: str, filename: str, constraints: List[str]) -> None:
        """Save constraints to file."""
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write("import numpy as np\n\n")
            for constraint in constraints:
                f.write(constraint)
                f.write("\n\n")

    def _save_cached_result(self, result: GenerationResult, cache_path: str) -> None:
        """Save result to cache."""
        cache_data = {
            "num_stages": result.num_stages,
            "grasp_keypoints": result.grasp_keypoints,
            "release_keypoints": result.release_keypoints,
            "subgoal_constraints": result.subgoal_constraints,
            "path_constraints": result.path_constraints,
            "raw_output": result.raw_output,
            "output_dir": result.output_dir,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

    def _load_cached_result(self, cache_path: str, output_dir: str) -> GenerationResult:
        """Load result from cache."""
        with open(cache_path, "r") as f:
            data = json.load(f)

        return GenerationResult(
            num_stages=data["num_stages"],
            grasp_keypoints=data["grasp_keypoints"],
            release_keypoints=data["release_keypoints"],
            subgoal_constraints={int(k): v for k, v in data["subgoal_constraints"].items()},
            path_constraints={int(k): v for k, v in data["path_constraints"].items()},
            raw_output=data["raw_output"],
            output_dir=output_dir,
        )


class ConstraintLoader:
    """
    Load and execute constraint functions from generated files.

    Provides safe execution of VLM-generated constraint code.
    """

    # Blacklist of dangerous operations
    BLACKLIST = [
        "import ", "exec", "eval", "__", "open", "file",
        "os.", "sys.", "subprocess", "compile", "globals", "locals"
    ]

    def __init__(self):
        """Initialize constraint loader."""
        self.loaded_functions: Dict[str, Callable] = {}

    def load_from_result(
        self,
        result: GenerationResult
    ) -> Dict[int, Dict[str, List[Callable]]]:
        """
        Load constraint functions from generation result.

        Args:
            result: GenerationResult from ConstraintGenerator

        Returns:
            {stage: {"subgoal": [fn1, fn2, ...], "path": [fn1, ...]}}
        """
        constraints = {}

        for stage in range(1, result.num_stages + 1):
            constraints[stage] = {
                "subgoal": [],
                "path": [],
            }

            # Load subgoal constraints
            if stage in result.subgoal_constraints:
                for code in result.subgoal_constraints[stage]:
                    fn = self._compile_function(code)
                    if fn is not None:
                        constraints[stage]["subgoal"].append(fn)

            # Load path constraints
            if stage in result.path_constraints:
                for code in result.path_constraints[stage]:
                    fn = self._compile_function(code)
                    if fn is not None:
                        constraints[stage]["path"].append(fn)

        return constraints

    def load_from_directory(self, directory: str) -> Dict[int, Dict[str, List[Callable]]]:
        """
        Load constraint functions from directory.

        Args:
            directory: Directory containing constraint .py files

        Returns:
            {stage: {"subgoal": [fn1, ...], "path": [fn1, ...]}}
        """
        constraints = {}

        # Find all constraint files
        import glob
        pattern = os.path.join(directory, "stage*_*_constraints.py")
        files = glob.glob(pattern)

        for filepath in files:
            filename = os.path.basename(filepath)
            match = re.match(r"stage(\d+)_(subgoal|path)_constraints\.py", filename)
            if not match:
                continue

            stage = int(match.group(1))
            constraint_type = match.group(2)

            if stage not in constraints:
                constraints[stage] = {"subgoal": [], "path": []}

            # Load functions from file
            with open(filepath, "r") as f:
                content = f.read()

            # Extract individual functions
            func_pattern = r"(def\s+stage\d+_\w+_constraint\d+\s*\([^)]*\):.*?return\s+[^\n]+)"
            matches = re.findall(func_pattern, content, re.DOTALL)

            for func_code in matches:
                fn = self._compile_function(func_code)
                if fn is not None:
                    constraints[stage][constraint_type].append(fn)

        return constraints

    def _compile_function(self, code: str) -> Optional[Callable]:
        """
        Safely compile a constraint function.

        Args:
            code: Python function code string

        Returns:
            Callable function or None if compilation fails
        """
        # Security check
        for banned in self.BLACKLIST:
            if banned in code:
                print(f"Warning: Banned operation '{banned}' found in constraint code")
                return None

        try:
            # Create safe namespace
            namespace = {"np": np, "numpy": np}

            # Compile and execute to get function
            exec(code, namespace)

            # Find the function in namespace
            for name, obj in namespace.items():
                if callable(obj) and name.startswith("stage"):
                    return obj

            return None

        except Exception as e:
            print(f"Error compiling constraint: {e}")
            return None
