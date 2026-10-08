import torch, time, pickle
import argparse, os, traceback
import jax
import numpy as np
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F

from gymnasium.wrappers import NormalizeObservation, ClipAction
from datetime import datetime
from sim2sim.avg.experiment_tracker import record_video
from sim2sim.avg.jax.td_error_scalar import TDErrorScaler
from sim2sim.avg.jax.model_utils import init_actor, init_Q
from sim2sim.avg.jax.td_error_scalar import init_td_error_scaler


def human_format_numbers(num, use_float=False):
    # Make human readable short-forms for large numbers
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if use_float:
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])
    return '%d%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def set_one_thread():
    '''
    N.B: Pytorch over-allocates resources and hogs CPU, which makes experiments very slow!
    Set number of threads for pytorch to 1 to avoid this issue. This is a temporary workaround.
    '''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)

class AVG:
    def __init__(self, cfg):
        self.cfg = cfg
        self.steps = 0  
        # model initializations and parameter initialization
        self.actor = init_actor(net_params=cfg.net_params, action_dim=cfg.action_dim, layer_norm=cfg.layer_norm, dtype=cfg.dtype)
        self.Q = init_Q(net_params=cfg.net_params, obs_dim=cfg.obs_dim, action_dim=cfg.action_dim, layer_norm=cfg.layer_norm, dtype=cfg.dtype)
        self.td_error_scaler = init_td_error_scaler()
        self.alpha, self.gamma = cfg.alpha_lr, cfg.gamma
        self.G = 0
        self.rng = jax.random.PRNGKey(cfg.seed)

    def compute_action(self, obs):
        rng, key = jax.random.split(self.rng)
        action, action_info = self.actor.apply_fn(self.actor.params, key, obs)
        return action, action_info  

    def update(self, obs, action, next_obs, reward, done, **kwargs):
        # TODO: Implement the update step with warm start critic and actor updates.
        if self.cfg.warm_start_critic is not None and self.steps <= self.cfg.warm_start_critic:
            ploss = torch.zeros((), device=self.device)
        else:
            ploss = self.alpha * lprob - self.Q(obs, action) # N.B: USE reparametrized action
            self.popt.zero_grad()
            ploss.backward()
            self.popt.step()

        self.qopt.zero_grad()
        qloss.backward()
        self.qopt.step()

        return ploss.item(), qloss.item()   


    def save(self, model_dir, unique_str):
        model = {
            "actor": self.actor.state_dict(),
            "critic": self.Q.state_dict(),
            "policy_opt": self.popt.state_dict(),
            "critic_opt": self.qopt.state_dict(),
        }
        torch.save(model, '%s/%s.pt' % (model_dir, unique_str))
    
class SqueezeVecEnv(gym.Env):
    def __init__(self, vec, seed=0):
        self.vec = vec
        self.observation_space = vec.single_observation_space
        self.action_space = vec.single_action_space

    def reset(self, *, seed=None, options=None):
        # Fixed start pose: vec.reset(seed=None) reuses the key from the last seeded reset.
        obs, info = self.vec.reset(seed=seed)
        return obs[0], info

    def step(self, action):
        obs, r, term, trunc, info = self.vec.step(np.asarray(action, np.float32)[None])
        if (term[0] or trunc[0]) and info.get("_final_observation", [False])[0]:
            obs = info["final_observation"]                             
        return obs[0], float(r[0]), bool(term[0]), bool(trunc[0]), info





def train(args, env, restore_path:str|None=None, log_fn=None):
    """log_fn(steps_done, episode_return) is called at the end of every episode."""
    tic = time.time()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"-{args.algo}-{args.env}_seed-{args.seed}"
    
    # Env
    env = SqueezeVecEnv(env)
    env = gym.wrappers.FlattenObservation(env)
    env = NormalizeObservation(env)
    env = ClipAction(env)

    #### Reproducibility
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    ####

    # Learner
    args.obs_dim =  env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    agent = AVG(args)

    if restore_path is not None:
        from sim2sim.playground.param_transfer import convert
        actor_sd, critic_sd, obs_rms = convert(
            restore_path,
            q_head=0)
        if args.transfer_actor:
            agent.actor.load_state_dict(actor_sd, strict=True)
        if args.transfer_critic:
            agent.Q.load_state_dict(critic_sd, strict=True)
        if args.transfer_obs_rms:
        # load the Sim1 normalizer stats from sim1 into sim2 env's NormalizeObservation.
            rms = env.get_wrapper_attr("obs_rms")
            assert obs_rms["mean"].shape == rms.mean.shape, (obs_rms["mean"].shape, rms.mean.shape)
            rms.mean, rms.var, rms.count = obs_rms["mean"].copy(), obs_rms["var"].copy(), obs_rms["count"]

    # Interaction     
    rets, ep_steps = [], []
    # wandb logging
    ret, step, ploss, qloss = 0, 0, 0, 0
    terminated, truncated = False, False
    obs, _ = env.reset()
    ep_tic = time.time()    
    try:
        for t in range(args.N):
            # N.B: Action is a torch.Tensor
            action, action_info = agent.compute_action(obs)                
            sim_action = action.detach().cpu().view(-1).numpy()

            # Receive reward and next state
            next_obs, reward, terminated, truncated, _ = env.step(sim_action)
            step += 1
            p_loss, q_loss = agent.update(obs, action, next_obs, reward, terminated, **action_info, step=step)            
            ret += reward
            ploss += p_loss
            qloss+= q_loss


            obs = next_obs

            if t % args.checkpoint == 0 and args.save_model:
                agent.save(model_dir=args.results_dir, unique_str=f"{run_id}_model_{human_format_numbers(t)}")

            # Termination
            if terminated or truncated:
                rets.append(ret)
                ep_steps.append(step)
                print("E: {}| D: {:.3f}| S: {}| R: {:.2f}| T: {}".format(len(rets), time.time() - ep_tic, step, ret, t))
                if log_fn is not None:
                    log_fn(t + 1, ret, ploss/step, qloss/step)

                ep_tic = time.time()
                obs, _ = env.reset()
                ret, step, ploss, qloss = 0, 0, 0, 0
    except Exception as e:
        print(e)
        print("Exiting this run, storing partial logs in the database for future debugging...")
        traceback.print_exc()
        raise  # N.B: otherwise a crashed AVG run looks finished and exper moves on to the next stage

    if not (terminated or truncated):
        # N.B: We're adding a partial episode just to make plotting easier. But this data point shouldn't be used
        print("Appending partial episode #{}, length: {}, Total Steps: {}".format(len(rets), step, t+1))
        rets.append(ret)
        ep_steps.append(step)
    
    # Save returns and args before exiting run
    if args.save_model:
        agent.save(model_dir=args.results_dir, unique_str=f"{run_id}_model")

    print("Run with id: {} took {:.3f}s!".format(run_id, time.time()-tic))
    
    # Eval
    if args.n_eval:
        record_video(env, agent, num_episodes=args.n_eval, video_filename=f"{args.results_dir}/{run_id}.avi")

    return ep_steps, rets, ploss, qloss


