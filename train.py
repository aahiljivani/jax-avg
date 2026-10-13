import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.struct
from flax.training.train_state import TrainState
from brax.training.acme import running_statistics, specs
from brax import envs
from brax.envs.wrappers.training import wrap
from td_error_scalar import TDErrorScaler, init_td_error_scaler
from model_utils import init_actor, init_Q, Transition, critic_update, actor_update
from utils import wandb_run, log_fn, save_avg_checkpoint

@flax.struct.dataclass
class AVGState:
    actor: TrainState
    critic: TrainState
    td_error_scaler: TDErrorScaler
    normalizer: running_statistics.RunningStatisticsState
    rng: jnp.ndarray
    steps: jnp.ndarray



def init_obs_normalizer(obs_dim):
    ''' observation normalizer'''
    return running_statistics.init_state(specs.Array((obs_dim,), jnp.float32))


def AVG(cfg, env):
    rng = jax.random.PRNGKey(cfg.seed)
    rng, actor_key, critic_key = jax.random.split(rng, 3)
    obs_dim = env.observation_size
    action_dim = env.action_size
    num_envs = cfg.num_envs
    net_params = {"mlp": [cfg.nhid_actor, cfg.nhid_actor]}
    layer_norm = False
    dtype = jnp.float32
    scaler = init_td_error_scaler()
    scaler = jax.tree.map(lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), scaler)
    return AVGState(
        actor=init_actor(actor_key, cfg.actor_lr, obs_dim, net_params, action_dim, layer_norm, dtype),
        critic=init_Q(critic_key, cfg.critic_lr, obs_dim, net_params, action_dim, layer_norm, dtype),
        td_error_scaler=scaler,
        normalizer=init_obs_normalizer(obs_dim),
        rng=rng,
        steps=jnp.zeros((), jnp.int32),
    )


def sample_action(actor, params, key, obs_i):
    ''' sample action from actor, return action and log probability'''
    action, info = actor.apply_fn({"params": params}, key, obs_i, temperature=1.0)
    return action, info["lprob"]


def make_step(env, alpha, gamma, warm_start):
    ''' 
    step consists of normalizing obs, sampling action, 
    env step, and updating both critic and actor.
    '''
    def step(agent, env_state, ep_ret):
        rng, act_key, upd_key = jax.random.split(agent.rng, 3)
        num_envs = env_state.obs.shape[0]
        keys = jax.random.split(act_key, num_envs)
        # obs normalization happens here, update statistics, normalize obs
        normalizer = running_statistics.update(agent.normalizer, env_state.obs)
        obs_n = running_statistics.normalize(env_state.obs, normalizer)
        # sample action
        action, lprob = jax.vmap(sample_action, in_axes=(None, None, 0, 0))(
            agent.actor, agent.actor.params, keys, obs_n)
        env_state = env.step(env_state, action)
        next_n = running_statistics.normalize(env_state.obs, normalizer)
        batch = Transition(
            obs=obs_n,
            action=action,
            reward=env_state.reward,
            done=env_state.done,
            next_obs=next_n,
            lprob=lprob,
            sample_key=keys,
        )

        scaler, qloss_fn = critic_update(
            upd_key, agent.actor, agent.critic, 1.0, batch, gamma, agent.td_error_scaler, alpha)
        qloss, qgrads = jax.value_and_grad(qloss_fn)(agent.critic.params)
        rng, agrads, ploss = actor_update(rng, agent.critic, batch, agent.actor, alpha)
        critic = agent.critic.apply_gradients(grads=qgrads)
        actor = jax.lax.cond(
            agent.steps >= warm_start,
            lambda a: a.apply_gradients(grads=agrads),
            lambda a: a,
            agent.actor,
        )
        ploss = jnp.where(agent.steps >= warm_start, ploss, 0.0)

        ep_ret = ep_ret + env_state.reward
        finished_ret = ep_ret
        ep_ret = jnp.where(env_state.done, jnp.zeros_like(ep_ret), ep_ret)

        agent = agent.replace(
            actor=actor,
            critic=critic,
            td_error_scaler=scaler,
            normalizer=normalizer,
            rng=rng,
            steps=agent.steps + num_envs,
        )
        metrics = {"ploss": ploss, "qloss": qloss, "timesteps": agent.steps}
        return agent, env_state, ep_ret, finished_ret, metrics

    return jax.jit(step)


def train(cfg, agent, env):
    num_envs = cfg.num_envs
    run = wandb_run(cfg) if cfg.wandb else None
    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), num_envs)
    env_state = env.reset(keys)
    ep_ret = jnp.zeros((num_envs,))
    step = make_step(env, cfg.alpha_lr, cfg.gamma, cfg.warm_start_critic)
    # training loop
    for _ in range(cfg.N // num_envs):
        agent, env_state, ep_ret, finished_ret, metrics = step(agent, env_state, ep_ret)
        if run is not None and bool(jnp.any(env_state.done)):
            done = jnp.asarray(env_state.done)
            for i in range(num_envs):
                if bool(done[i]):
                    log_fn(
                        {
                            "timesteps": float(metrics["timesteps"]),
                            "episode_return": float(finished_ret[i]),
                            "ploss": float(metrics["ploss"]),
                            "qloss": float(metrics["qloss"]),
                        },
                        run,
                    )
    return agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default="walker2d", type=str, help="e.g., 'walker2d' check brax documentation")
    parser.add_argument('--seed', default=42, type=int, help="Seed for random number generator")       
    parser.add_argument('--N', default=10001000, type=int, help="# timesteps for the run")
    # SAVG params
    parser.add_argument('--actor_lr', default=0.0063, type=float, help="Actor step size")
    parser.add_argument('--critic_lr', default=0.0087, type=float, help="Critic step size")
    parser.add_argument('--beta1', default=0., type=float, help="Beta1 parameter of Adam optimizer")
    parser.add_argument('--gamma', default=0.99, type=float, help="Discount factor")
    parser.add_argument('--alpha_lr', default=0.07, type=float, help="Entropy Coefficient for AVG")
    parser.add_argument('--l2_actor', default=0, type=float, help="L2 Regularization")
    parser.add_argument('--l2_critic', default=0, type=float, help="L2 Regularization")    
    parser.add_argument('--nhid_actor', default=256, type=int)
    parser.add_argument('--nhid_critic', default=256, type=int)
    # Miscellaneous
    parser.add_argument('--checkpoint', default=50000, type=int, help="Save plots and rets every checkpoint")
    parser.add_argument('--results_dir', default="./results", type=str, help="Location to store results")
    parser.add_argument('--device', default="cpu", type=str)    
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--n_eval', default=0, type=int, help="Number of eval episodes")
    parser.add_argument('--num_envs', default=5, type=int, help="Number of environments")
    parser.add_argument('--episode_length', default=1000, type=int, help="Episode horizon")
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--warm_start_critic', default=5000, type=int, help="Env steps before the actor updates")
    parser.add_argument('--model_dir', default="./models", type=str, help="Location to store models")
    args = parser.parse_args()

    # create env
    env = wrap(envs.create(env_name=args.env), episode_length=args.episode_length)
    # create agent
    agent = AVG(args, env)
    # train agent
    agent = train(args, agent, env)
    # save agent
    if args.save_model:
        path = Path(args.model_dir)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        
        save_avg_checkpoint(path, agent)