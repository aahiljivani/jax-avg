import os
from pathlib import Path
import pickle
import wandb
from flax import serialization
def save_avg_checkpoint(path, agent) -> Path:
    ''' save checkpoint for avg '''
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    blob = {
        "actor": agent.actor.params,
        "critic": agent.critic.params,
        "normalizer": serialization.to_state_dict(agent.normalizer),
    }
    with open(path / "avg_state.pkl", "wb") as f:
        pickle.dump(blob, f)
    return path
def wandb_run(args):
    ''' initialize wandb run '''
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "AVG"),
        name=f"AVG-training-seed-{args.seed}",
        config=args,
        save_code=True)
    
    return run

def log_fn(metrics, run):
    ''' log metrics to wandb '''
    run.log({
        "Timesteps": metrics["timesteps"],
        "Episode_Returns": metrics["episode_return"],
        "Policy_Loss": metrics["ploss"],
        "Critic_Loss": metrics["qloss"],
    })