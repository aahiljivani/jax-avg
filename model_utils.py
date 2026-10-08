import optax
import numpy as np
import jax
import flax.struct
from jax import numpy as jnp
from jax import random
from flax.training.train_state import TrainState
from avg_models import Actor, Q
from td_error_scalar import TDErrorScaler, sigma, update_td_error_scaler

@flax.struct.dataclass
class Transition:
    """One environment step for each of N envs.

    The leading axis is the env axis. N = 1 is the single-env case and
    N = num_envs is parallel training. `sample_key` has shape (N, 2) and is
    the key that produced `action` and `lprob`. The distribution is not stored.
    """
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray
    lprob: jnp.ndarray
    sample_key: jnp.ndarray

# model initializations
def init_Q(rng, 
lr,
obs_dim,
net_params,
action_dim,
layer_norm,
dtype,
):
    q = Q(net_params, obs_dim, action_dim, layer_norm, dtype)
    init_actions = jnp.zeros((1, action_dim))
    init_obs = jnp.zeros((1, obs_dim))
    rng, init_key, sample_key = random.split(rng, 3)
    params = q.init(init_key, sample_key, init_obs, init_actions)['params']
    tx=optax.adam(lr, b1=0.0, mu_dtype=dtype) # zero nans and clip by global norm? JSAC reference
    return rng, TrainState.create(apply_fn=q.apply, params=params, tx=tx)

def init_actor(rng,
lr,
obs_dim,
net_params,
action_dim,
layer_norm,
dtype,
):
    actor = Actor(net_params, action_dim, layer_norm, dtype)
    init_obs = jnp.zeros((1, obs_dim))
    rng, init_key, sample_key = random.split(rng, 3)
    params = actor.init(init_key, sample_key, init_obs)['params']
    tx=optax.adam(lr, b1=0.0, mu_dtype=dtype) # zero nans and clip by global norm? JSAC reference
    return rng, TrainState.create(apply_fn=actor.apply, params=params, tx=tx)


def critic_update(rng, 
                  actor, 
                  critic,  
                  temp, 
                  transition, 
                  discount,
                  td_error_scaler,
                  alpha):

    rng, key_ac, key_q, key_q2 = random.split(rng, 4)
    # reward entropy scaled
    r_ent = transition.reward - alpha * jax.lax.stop_gradient(transition.lprob)
    # one scaler step per env; a stacked reward would corrupt a scalar G
    td_error_scaler = jax.vmap(update_td_error_scaler, in_axes=(0, 0, None, 0))(
        td_error_scaler, r_ent, discount, transition.done)

    # JAX has no tape: grad has to call this, with the critic weights as `params`.
    def loss(params):
        q = critic.apply_fn(
            {'params': params}, 
            key_q,
            transition.obs,
            jax.lax.stop_gradient(transition.action),
            )
        next_actions, next_action_info = actor.apply_fn(
            {"params": actor.params}, 
            key_ac,
            transition.next_obs,
            temperature=1.0)
        
        q2 = critic.apply_fn(
            {'params': params}, 
            key_q2,
            transition.next_obs,
            next_actions,
            )
        target_V = jax.lax.stop_gradient(q2 - alpha * next_action_info['lprob'])
        delta = transition.reward + (1 - transition.done) * discount * target_V - q
        delta /= jax.lax.stop_gradient(sigma(td_error_scaler))
        return jnp.mean(delta ** 2)
    return rng, td_error_scaler, loss
    
def q_loss(critic, loss):
    grads = jax.grad(loss)(critic.params)
    critic = critic.apply_gradients(grads=grads)
    return critic
def actor_update(rng, 
                  critic,
                  transition,
                  actor,
                  alpha):
    rng, key_q = random.split(rng, 2)
    # Replay the key stored on the transition. The stored action is a constant,
    # so grad through it is zero.
    def loss(params):
        def one(key, obs_i):
            action, info = actor.apply_fn({"params": params}, key, obs_i, temperature=1.0)
            return action, info['lprob']
        action, lprob = jax.vmap(one)(transition.sample_key, transition.obs)
        q = critic.apply_fn({'params': critic.params}, key_q, transition.obs, action)
        return jnp.mean(alpha * lprob - q)
    grads = jax.grad(loss)(actor.params)
    return rng, grads

