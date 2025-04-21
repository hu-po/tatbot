import argparse
from dataclasses import dataclass, field
import os
import random
import re
import subprocess
import time
import uuid
import yaml
import json
from typing import List, Dict
import logging
import asyncio

from tatbot.ai import AIConfig
from tatbot.ik.mutate import mutate_async, MutateConfig
from tatbot.ik.morph import ActiveMorph, MorphState, MorphConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s|%(name)s|%(levelname)s|%(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

@dataclass
class EvolveConfig:
    seed: int = 42 # random seed
    num_rounds: int = 3 # number of rounds of evolution (mutation + elimination)
    num_morphs: int = 20 # desired size of the population of morphs (may overshoot slightly)
    topk_morphs: int = 10 # number of top morphs to keep each round
    mutate_on_start: bool = False # whether to mutate protomorphs at the start
    root_dir: str = os.environ.get("TATBOT_ROOT")
    protomorphs_str: str = os.environ.get("PROTOMORPHS", "ik_geom_6d,ik_fd_6d") # comma separated list
    output_dir: str = field(init=False)
    morph_dir: str = field(init=False)
    enabled_models: List[str] = field(default_factory=lambda: AIConfig().enabled_models) # Models for mutation
    num_envs: int = int(os.environ.get("NUM_ENVS", 4)) # Env var for morph run
    track_morphs: bool = True # Whether to track individual morph runs with WandB
    headless_morphs: bool = True # Whether to run morphs headlessly

    def __post_init__(self):
        if not self.root_dir:
            raise ValueError("TATBOT_ROOT environment variable must be set.")
        self.output_dir = os.path.join(self.root_dir, "output")
        self.morph_dir = os.path.join(self.root_dir, "warp_ik", "morphs")
        # Ensure base output dir exists
        os.makedirs(self.output_dir, exist_ok=True)

def get_available_morphs(morph_dir: str) -> List[str]:
    """Returns a list of available morph names from the morphs directory."""
    if not os.path.exists(morph_dir):
        return []
    return [os.path.splitext(f)[0] for f in os.listdir(morph_dir) 
            if f.endswith('.py') and os.path.isfile(os.path.join(morph_dir, f))]

def run_morph_subprocess(config: EvolveConfig, morph: ActiveMorph) -> float:
    """Runs a single morph simulation using subprocess and returns its score."""
    log.info(f"\t⏯️\tRunning {morph.name} with {config.num_envs} envs...")
    morph_script_path = os.path.join(config.root_dir, "warp_ik", "src", "morph.py")
    morph_output_dir = os.path.join(config.output_dir, morph.name)
    results_filepath = os.path.join(morph_output_dir, "results.json")

    # Ensure morph-specific output directory exists
    os.makedirs(morph_output_dir, exist_ok=True)

    command = [
        "uv", "run", "python", morph_script_path,
        "--morph", morph.name,
        "--backend", config.backend,
        "--num_envs", str(config.num_envs),
        "--seed", str(config.seed),
    ]
    if config.track_morphs:
        command.append("--track")
    if config.headless_morphs:
        command.append("--headless")

    log.debug(f"Running command: {' '.join(command)}")
    score = 0.0 # Default score if run fails

    try:
        # Using subprocess.run for simpler error checking and output capture
        # Timeout could be added here if needed: timeout=SOME_SECONDS
        proc = subprocess.run(command, capture_output=True, text=True, check=False, cwd=config.root_dir)

        if proc.returncode != 0:
            log.error(f"\t❌\tError when running {morph.name} (Code: {proc.returncode})")
            log.error(f"\t\tStderr: {proc.stderr.strip()}")
            log.error(f"\t\tStdout: {proc.stdout.strip()}")
            morph.state = MorphState.ERRORED_OUT
            return score # Return default score 0.0

        # Process finished successfully, try to read results
        if not os.path.exists(results_filepath):
             log.error(f"\t❌\tResults file not found for {morph.name} at {results_filepath}")
             log.error(f"\t\tStdout: {proc.stdout.strip()}") # Log stdout to help debug file creation issues
             morph.state = MorphState.ERRORED_OUT
             return score

        try:
            with open(results_filepath, "r") as f:
                morph_output = json.load(f) # results.json should be JSON
            # Use 'score' which is 1 / (1 + pos_err + ori_err) - higher is better
            score = morph_output.get("score", 0.0)
            if not isinstance(score, (float, int)):
                 log.warning(f"\t⚠️ Invalid score type ({type(score)}) found for {morph.name}, using 0.0.")
                 score = 0.0
            morph.score = score # Update the morph object directly
            morph.state = MorphState.ALREADY_RAN
            log.info(f"\t🏁\t{morph.name} scored {score:.6f}")
        except json.JSONDecodeError:
            log.error(f"\t❌\tFailed to decode results JSON for {morph.name} at {results_filepath}")
            morph.state = MorphState.ERRORED_OUT
        except KeyError:
            log.error(f"\t❌\t'score' key not found in results for {morph.name}")
            morph.state = MorphState.ERRORED_OUT
        except Exception as e:
             log.error(f"\t❌\tError reading results file for {morph.name}: {e}")
             morph.state = MorphState.ERRORED_OUT

    except FileNotFoundError:
        log.error(f"\t❌\tCommand 'uv' not found. Make sure uv is installed and in PATH.")
        morph.state = MorphState.ERRORED_OUT
    except Exception as e:
        log.error(f"\t❌\tUnexpected error running subprocess for {morph.name}: {e}")
        morph.state = MorphState.ERRORED_OUT

    return score


async def evolve_async(config: EvolveConfig):
    """Runs the evolutionary process asynchronously."""
    log.info("🌱 Starting Evolution 🌱")
    log.info(f"Seed: {config.seed}")
    random.seed(config.seed)
    np_rng = random.Random(config.seed) # Separate RNG for numpy if needed later

    # --- Initialize Population ---
    morphs: List[ActiveMorph] = []
    
    # Handle "all" case for protomorphs
    if config.protomorphs_str.strip().lower() == "all":
        available_morphs = get_available_morphs(config.morph_dir)
        if not available_morphs:
            log.error("No morphs found in morphs directory. Aborting evolution.")
            return
        # Randomly select num_morphs from available morphs
        selected_morphs = random.sample(available_morphs, min(config.num_morphs, len(available_morphs)))
        protomorph_names = selected_morphs
        log.info(f"Selected {len(selected_morphs)} random morphs from {len(available_morphs)} available morphs")
    else:
        protomorph_names = [p.strip() for p in config.protomorphs_str.split(",") if p.strip()]
    
    log.info("Loading protomorphs:")
    for name in protomorph_names:
        morph_path = os.path.join(config.morph_dir, f"{name}.py")
        if os.path.exists(morph_path):
            morphs.append(ActiveMorph(score=0, name=name, state=MorphState.NOT_RUN_YET)) # Initial score 0
            log.info(f"\t✅\tLoaded ~{name}~")
        else:
            log.warning(f"\t❌\tProtomorph file not found: {morph_path}, skipping.")

    if not morphs:
        log.error("No valid protomorphs found. Aborting evolution.")
        return

    # --- Setup Session ---
    session_id = str(uuid.uuid4())[:6]
    session_dir = os.path.join(config.output_dir, f"evolve_session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    log.info(f"Evolution session: {session_id} | Output Dir: {session_dir}")

    # --- Evolution Loop ---
    for round_num in range(config.num_rounds):
        log.info(f"\n--- 🥊 Round {round_num+1}/{config.num_rounds} ---")
        current_population_names = {m.name for m in morphs}
        log.info(f"Population size at start of round: {len(morphs)}")

        # --- Mutation ---
        log.info("🧬 Performing Mutation...")
        # Keep track of successful survivors from previous round
        survivors = morphs.copy() # These are potential parents
        if not survivors:
             log.warning("No survivors from previous round to mutate from. Using initial protomorphs.")
             # Reload initial protomorphs if survivors list is empty
             survivors = []
             for name in protomorph_names:
                 morph_path = os.path.join(config.morph_dir, f"{name}.py")
                 if os.path.exists(morph_path):
                     survivors.append(ActiveMorph(score=0, name=name, state=MorphState.NOT_RUN_YET))


        num_to_mutate = config.num_morphs - len(morphs)
        if num_to_mutate <= 0 and not config.mutate_on_start and round_num > 0:
            log.info("Population full, skipping mutation for this round.")
        else:
            if config.mutate_on_start and round_num == 0:
                 log.info("Mutating initial protomorphs...")
                 parents_to_mutate = morphs # Mutate all initial ones
                 morphs = [] # Start fresh population
            else:
                 # Select parents randomly from survivors to fill the gap
                 parents_to_mutate = random.choices(survivors, k=max(1, num_to_mutate // len(config.enabled_models) + 1)) # Ensure at least one mutation attempt

            mutation_tasks = []
            mutate_config = MutateConfig(
                seed=config.seed + round_num, # Vary seed per round
                root_dir=config.root_dir,
                enabled_models=config.enabled_models
                # morph_name will be set per parent inside the loop
            )

            log.info(f"Selected {len(parents_to_mutate)} parents for mutation: {[p.name for p in parents_to_mutate]}")

            for parent in parents_to_mutate:
                # Pass the specific parent's name to MutateConfig for output dir naming
                mutate_config.morph_name = parent.name
                mutation_tasks.append(mutate_async(mutate_config, parent))

            # Run mutation tasks concurrently
            mutation_results: List[List[ActiveMorph]] = await asyncio.gather(*mutation_tasks)

            # Add newly generated morphs to the population
            newly_added_count = 0
            for morph_list in mutation_results:
                for neomorph in morph_list:
                    # Avoid adding duplicates if somehow generated twice
                    if neomorph.name not in current_population_names:
                        morphs.append(neomorph)
                        current_population_names.add(neomorph.name)
                        newly_added_count += 1

            log.info(f"Added {newly_added_count} new unique morphs from mutation.")
            log.info(f"Population size after mutation: {len(morphs)}")


        # --- Evaluation ---
        log.info("📊 Evaluating Population...")
        leaderboard: Dict[str, float] = {}
        # Can run evaluation in parallel later if needed, for now sequential
        for morph in morphs:
            if morph.state == MorphState.ALREADY_RAN:
                log.info(f"\t⏩\tSkipping {morph.name} (already ran with score {morph.score:.6f})")
                leaderboard[morph.name] = morph.score # Use existing score
            elif morph.state == MorphState.ERRORED_OUT:
                log.info(f"\t⏩\tSkipping {morph.name} (previously errored)")
                leaderboard[morph.name] = 0.0 # Assign score 0 for errored morphs
            else:
                # Check if the morph file actually exists before trying to run
                morph_file_path = os.path.join(config.morph_dir, f"{morph.name}.py")
                if not os.path.exists(morph_file_path):
                     log.error(f"\t❌\tMorph file missing for {morph.name} at {morph_file_path}. Setting state to ERRORED_OUT.")
                     morph.state = MorphState.ERRORED_OUT
                     leaderboard[morph.name] = 0.0
                     continue # Skip to next morph

                # Run the morph simulation
                score = run_morph_subprocess(config, morph) # This updates morph.state and morph.score internally
                leaderboard[morph.name] = score

        # Sort leaderboard (higher score is better)
        sorted_leaderboard = {k: v for k, v in sorted(leaderboard.items(), key=lambda item: item[1], reverse=True)}

        # Save leaderboard for the round
        leaderboard_filepath = os.path.join(session_dir, f"leaderboard.r{round_num+1}.yaml")
        try:
            with open(leaderboard_filepath, "w") as f:
                yaml.safe_dump(sorted_leaderboard, f, default_flow_style=False, sort_keys=False)
            log.info(f"Saved round leaderboard to {leaderboard_filepath}")
        except Exception as e:
            log.error(f"Failed to save leaderboard: {e}")

        # --- Selection (Elimination) ---
        log.info("🔪 Performing Selection...")
        # Sort morph objects by score (descending)
        morphs.sort(key=lambda m: m.score, reverse=True)

        log.info("Leaderboard for Round:")
        for i, morph in enumerate(morphs):
             rank_char = "🏆" if i < config.topk_morphs else "💔"
             log.info(f"\t{rank_char} {i+1: >2}. {morph.name:<20} Score: {morph.score:.6f} ({morph.state.name})")


        # Keep only the top K morphs
        survivors = morphs[:config.topk_morphs]
        doomed_count = len(morphs) - len(survivors)

        if doomed_count > 0:
             log.info(f"Eliminating {doomed_count} morph(s)...")
             # Optional: Clean up files of doomed morphs? Be careful here.
             # for doomed_morph in morphs[config.topk_morphs:]:
             #     log.info(f"\t🗑️ Removing {doomed_morph.name}")
             #     # Example cleanup (use with caution):
             #     # doomed_file = os.path.join(config.morph_dir, f"{doomed_morph.name}.py")
             #     # doomed_output = os.path.join(config.output_dir, doomed_morph.name)
             #     # if os.path.exists(doomed_file): os.remove(doomed_file)
             #     # if os.path.exists(doomed_output): shutil.rmtree(doomed_output) # Need import shutil
             pass # No cleanup for now

        morphs = survivors # Update population for the next round

        if not morphs:
            log.error("Population extinct! No survivors after selection. Stopping evolution.")
            break # Exit evolution loop

    log.info("\n🏁 Evolution Finished! 🏁")
    if morphs:
        log.info("Final Population:")
        for i, morph in enumerate(morphs):
            log.info(f"\t🏅 {i+1}. {morph.name:<20} Score: {morph.score:.6f}")
    else:
        log.info("Population did not survive.")


async def main():
    parser = argparse.ArgumentParser(description="Evolve IK morphs.")
    parser.add_argument("--seed", type=int, default=EvolveConfig.seed)
    parser.add_argument("--protomorphs", type=str, default=EvolveConfig.protomorphs_str, help="Comma-separated list of protomorphs.")
    parser.add_argument("--num_rounds", type=int, default=EvolveConfig.num_rounds)
    parser.add_argument("--num_morphs", type=int, default=EvolveConfig.num_morphs, help="Target population size per round.")
    parser.add_argument("--topk_morphs", type=int, default=EvolveConfig.topk_morphs, help="Number of top morphs to keep.")
    parser.add_argument("--mutate_on_start", action="store_true", default=EvolveConfig.mutate_on_start, help="Mutate protomorphs at start.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated list of models for mutation (overrides default).")
    parser.add_argument("--num_envs", type=int, default=EvolveConfig.num_envs, help="Number of envs for morph simulation runs.")
    parser.add_argument("--no-track", action="store_true", help="Disable WandB tracking for morph runs.")
    parser.add_argument("--render", action="store_true", help="Enable rendering for morph runs (run headful).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    if args.debug:
        log.setLevel(logging.DEBUG)
        logging.getLogger('mutate').setLevel(logging.DEBUG)
        logging.getLogger('ai').setLevel(logging.DEBUG)
        logging.getLogger('morph').setLevel(logging.DEBUG)


    enabled_models_list = args.models.split(",") if args.models else AIConfig().enabled_models

    config = EvolveConfig(
        seed=args.seed,
        protomorphs_str=args.protomorphs,
        num_rounds=args.num_rounds,
        num_morphs=args.num_morphs,
        topk_morphs=args.topk_morphs,
        mutate_on_start=args.mutate_on_start,
        enabled_models=enabled_models_list,
        num_envs=args.num_envs,
        track_morphs=not args.no_track,
        headless_morphs=not args.render,
    )

    await evolve_async(config)


if __name__ == "__main__":
     asyncio.run(main())