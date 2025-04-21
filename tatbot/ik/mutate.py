import argparse
from dataclasses import dataclass, field
import os
import random
import re
import subprocess
import sys
import time
import uuid
import yaml
from typing import List, Dict
from enum import Enum, auto
import asyncio
import logging
import aiofiles

from tatbot.ai import async_inference, AI_MODEL_MAP, AIConfig
from tatbot.ik.morph import ActiveMorph, MorphState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class MutateConfig:
    seed: int = 33
    morph_name: str = "template" # Name of the protomorph being mutated
    root_dir: str = os.environ.get("TATBOT_ROOT") # root directory of the warp-ik project
    morph_dir: str = f"{root_dir}/warp_ik/morphs" # directory for the morphs
    prompts_dir: str = f"{root_dir}/warp_ik/prompts" # directory for the prompts
    output_dir: str = field(init=False) # directory for mutation outputs
    enabled_models: List[str] = field(default_factory=lambda: AIConfig().enabled_models) # Get defaults from AIConfig

    def __post_init__(self):
        if not self.root_dir:
            raise ValueError("TATBOT_ROOT environment variable must be set.")
        # Create a dedicated output dir for this mutation process if needed
        self.output_dir = os.path.join(self.root_dir, "output", "mutations", f"{self.morph_name}-{str(uuid.uuid4())[:6]}")
        os.makedirs(self.output_dir, exist_ok=True)

async def load_prompt_async(prompt_filepath: str) -> str:
    """Asynchronously loads a prompt file."""
    try:
        async with aiofiles.open(prompt_filepath, "r", encoding="utf-8") as f:
            return await f.read()
    except FileNotFoundError:
        log.error(f"Prompt file not found: {prompt_filepath}")
        raise
    except Exception as e:
        log.error(f"Error loading prompt {prompt_filepath}: {e}")
        raise

async def morph_to_prompt_async(morph_filepath: str) -> str:
    """Asynchronously reads the content of a morph file."""
    try:
        async with aiofiles.open(morph_filepath, "r", encoding="utf-8") as f:
            return await f.read()
    except FileNotFoundError:
        log.error(f"Morph file not found: {morph_filepath}")
        return "" # Return empty string if morph file doesn't exist
    except Exception as e:
        log.error(f"Error reading morph file {morph_filepath}: {e}")
        return ""

async def reply_to_morph_async(reply: str, name: str, output_morph_dir: str, debug_output_dir: str) -> ActiveMorph | None:
    """Asynchronously saves the AI's reply (code) to a new morph file."""
    # Basic check for API error messages or empty replies
    if not reply or "API error" in reply or "timed out" in reply:
        log.warning(f"Skipping saving morph {name} due to invalid reply: {reply[:100]}...")
        return None

    # Save raw reply to debug directory
    try:
        os.makedirs(debug_output_dir, exist_ok=True)
        raw_reply_path = os.path.join(debug_output_dir, f"{name}-reply_raw.txt")
        async with aiofiles.open(raw_reply_path, "w", encoding="utf-8") as f:
            await f.write(reply)
        log.info(f"💾 Saved raw reply to {raw_reply_path}")
    except Exception as e:
        log.error(f"Failed to save raw reply for {name}: {e}")

    # Process the reply
    processed_reply = reply
    # Remove <think> sections (used by replicate model)
    processed_reply = re.sub(r'<think>.*?</think>', '', processed_reply, flags=re.DOTALL)
    processed_reply = re.sub(r'^```python\s*', '', processed_reply, flags=re.MULTILINE)
    processed_reply = re.sub(r'\n```$', '', processed_reply, flags=re.MULTILINE)
    processed_reply = re.sub(r'^```$', '', processed_reply, flags=re.MULTILINE)
    # Remove any leading/trailing whitespace after processing
    processed_reply = processed_reply.strip()

    # Save processed reply to debug directory
    try:
        processed_reply_path = os.path.join(debug_output_dir, f"{name}-reply_processed.txt")
        async with aiofiles.open(processed_reply_path, "w", encoding="utf-8") as f:
            await f.write(processed_reply)
        log.info(f"💾 Saved processed reply to {processed_reply_path}")
    except Exception as e:
        log.error(f"Failed to save processed reply for {name}: {e}")

    # Basic sanity check: does it contain 'class Morph(BaseMorph):'?
    if 'class Morph(BaseMorph):' not in processed_reply:
        log.warning(f"Skipping saving morph {name} - does not appear to contain valid Morph class structure.")
        return None

    morph = ActiveMorph(score=0, name=name) # Initialize with score 0
    morph_filepath = os.path.join(output_morph_dir, f"{name}.py")
    try:
        async with aiofiles.open(morph_filepath, "w", encoding="utf-8") as f:
            await f.write(processed_reply)
        log.info(f"💾 Saved neomorph to {morph_filepath}")
        return morph
    except Exception as e:
        log.error(f"Failed to write morph file {morph_filepath}: {e}")
        return None

async def mutate_async(config: MutateConfig, protomorph: ActiveMorph) -> List[ActiveMorph]:
    """
    Mutates a protomorph by querying multiple AI models to generate new algorithms.

    Args:
        config: The mutation configuration.
        protomorph: The active morph to use as a basis for mutation prompts.

    Returns:
        A list of new ActiveMorph objects, one for each successfully generated neomorph.
    """
    log.info(f"🧫 Mutating protomorph: {protomorph.name} using models: {config.enabled_models}")

    # --- Load Prompts ---
    # Get all .txt files from the prompts directory
    prompt_files = [f for f in os.listdir(config.prompts_dir) if f.endswith('.txt')]
    if not prompt_files:
        log.error(f"No .txt prompt files found in {config.prompts_dir}")
        return []
    
    # Randomly select one prompt file
    selected_prompt_file = random.choice(prompt_files)
    prompt_filepath = os.path.join(config.prompts_dir, selected_prompt_file)
    protomorph_filepath = os.path.join(config.morph_dir, f"{protomorph.name}.py")

    try:
        prompt_task = load_prompt_async(prompt_filepath)
        morph_code_task = morph_to_prompt_async(protomorph_filepath)
        prompt, morph_code = await asyncio.gather(prompt_task, morph_code_task)
    except Exception as e:
        log.error(f"Failed to load necessary files for mutation: {e}")
        return []

    if not morph_code:
        log.error(f"Protomorph code for {protomorph.name} is empty or could not be read. Aborting mutation.")
        return []

    # --- Construct Prompts for Each Model ---
    random.seed(config.seed) # Seed for reproducible randomization if needed later
    model_prompts = {}
    for model in config.enabled_models:
        final_prompt = prompt.replace("{{MORPH_CODE_PLACEHOLDER}}", morph_code)
        model_prompts[model] = final_prompt
        # Optionally, save the prompt used for each model
        final_prompt_filepath = os.path.join(config.output_dir, f"mutation_prompt_{protomorph.name}_{model}.txt")
        try:
            async with aiofiles.open(final_prompt_filepath, "w", encoding="utf-8") as f:
                await f.write(final_prompt)
            log.info(f"Saved final mutation prompt for {model} to {final_prompt_filepath}")
        except Exception as e:
            log.error(f"Failed to save final mutation prompt for {model}: {e}")

    # --- Call AI Models ---
    log.info(f"🧬 Calling AI models ({', '.join(config.enabled_models)}) for novel or improved algorithms...")
    # Call each model with its own prompt
    inference_tasks = [async_inference(
        prompt=model_prompts[model],
        image_path=None,
        enabled_models=[model],
    ) for model in config.enabled_models]
    replies_list = await asyncio.gather(*inference_tasks)
    # replies_list is a list of dicts with one key each
    replies = {model: reply_dict.get(model, "") for model, reply_dict in zip(config.enabled_models, replies_list)}

    # --- Process Replies and Create Neomorphs ---
    neomorphs: List[ActiveMorph] = []
    save_tasks = []
    for model_name, reply in replies.items():
        if isinstance(reply, str) and "API error" not in reply and "timed out" not in reply and reply.strip():
            log.info(f"✅ Received potential neomorph from {model_name}")
            # Create a unique name, e.g., "gpt-a1b2c3"
            neomorph_name = f"{model_name}-{str(uuid.uuid4())[:6]}"
            # Create task to save the morph asynchronously
            save_tasks.append(reply_to_morph_async(reply, neomorph_name, config.morph_dir, config.output_dir))
        else:
            log.warning(f"❌ Failed or empty reply from {model_name}: {str(reply)[:100]}...")

    # Wait for all save tasks to complete
    results = await asyncio.gather(*save_tasks)

    # Filter out None results (failed saves)
    neomorphs = [morph for morph in results if morph is not None]

    log.info(f"🥚 Generated {len(neomorphs)} new morph(s): {[m.name for m in neomorphs]}")
    return neomorphs


# Updated main block for testing mutate.py directly
async def main():
    parser = argparse.ArgumentParser(description="Mutate a protomorph using AI models.")
    parser.add_argument("--seed", type=int, default=MutateConfig.seed, help="Random seed.")
    parser.add_argument("--morph", type=str, required=True, help="Name of the protomorph python file (without .py) in the morphs directory.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models to use (e.g., 'gpt,claude').")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    if args.debug:
        # Get the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # Set root level

        # Remove existing handlers (if any) added by basicConfig to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create a stream handler to output to stderr (or stdout)
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)  # Ensure handler also processes debug messages

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            '%(asctime)s|%(name)s|%(levelname)s|%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add the handler to the root logger
        root_logger.addHandler(handler)

        # Set levels for specific loggers if needed (optional now, root covers it)
        log.setLevel(logging.DEBUG)  # 'mutate' logger
        logging.getLogger('ai').setLevel(logging.DEBUG)  # 'ai' logger

        # Add test messages
        log.debug("DEBUG logging test message from 'mutate' logger.")
        logging.getLogger('ai').debug("DEBUG logging test message from 'ai' logger.")
        logging.debug("DEBUG logging test message from root logger.")  # Test root directly

    else:
        # Ensure default INFO level if not debugging
        log.setLevel(logging.INFO)
        logging.getLogger('ai').setLevel(logging.INFO)
        # Consider if root logger needs setting back to INFO or if basicConfig handled it
        logging.getLogger().setLevel(logging.INFO)

    enabled_models_list = args.models.split(",") if args.models else AIConfig().enabled_models

    config = MutateConfig(
        seed=args.seed,
        morph_name=args.morph,
        enabled_models=enabled_models_list,
    )

    # Create a dummy protomorph object for the function call
    protomorph_obj = ActiveMorph(score=0, name=args.morph)

    # Check if protomorph file exists before attempting mutation
    protomorph_filepath = os.path.join(config.morph_dir, f"{config.morph_name}.py")
    if not os.path.exists(protomorph_filepath):
        log.error(f"Protomorph file not found: {protomorph_filepath}")
        return

    log.info(f"Starting mutation for protomorph: {config.morph_name}")
    generated_morphs = await mutate_async(config, protomorph_obj)

    if generated_morphs:
        log.info("Mutation process completed. Generated morphs:")
        for morph in generated_morphs:
            log.info(f"  - {morph.name}")
    else:
        log.warning("Mutation process completed, but no valid morphs were generated.")

if __name__ == "__main__":
    asyncio.run(main())