from typing import Sequence, Any
import jax, jax.numpy as jnp
import flax.linen as nn
from brax.training.agents.sac import networks as sac_networks
from brax.training import distribution, networks, types
from tensorflow_probability.substrates import jax as tfp

#### for SAC -AVG tranfer policy heads in brax training ####
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

class AVGLayers(nn.Module):
    hidden: Sequence[int]
    @nn.compact
    def __call__(self,x):
        for i, h in enumerate(self.hidden):
            x = nn.Dense(h, name=f"hidden_{i}", kernel_init=jax.nn.initializers.orthogonal())(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.01)
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

class AVGActor(nn.Module):
    action_dim: int
    hidden: Sequence[int]

    @nn.compact
    def __call__(self, obs):
        phi = AVGLayers(self.hidden, name='phi')(obs)
        mu = nn.Dense(self.action_dim, name="mu")(phi)
        log_std =nn.Dense(self.action_dim, name="log_std")(phi)
        return jnp.concatenate([mu, log_std], axis =-1)

class NormalTanH(distribution.NormalTanhDistribution):

    def create_dist(self, parameters):
        mu, log_std = jnp.split(parameters, 2, axis=-1)
        log_std = jnp.clip(log_std,LOG_STD_MIN,LOG_STD_MAX)
         
        return distribution._NormalDistribution(loc=mu, scale=jnp.exp(0.5*log_std))


class AVGQLayer(nn.Module):
    hidden: Sequence[int]
    @nn.compact
    def __call__(self,x):
        return nn.Dense(1, name='q')(AVGLayers(self.hidden, name="phi")(x))

class AVGQ(nn.Module):
    hidden:Sequence[int]
    n_critics: int
    @nn.compact
    def __call__(self,obs, act):
        x = jnp.concatenate([obs,act], axis=-1)
        # critic and target critic num_critics should be 2
        return jnp.concatenate([AVGQLayer(self.hidden, name=f"q_{i}")(x)for i in range (self.n_critics)],-1)



def make_avg_sac_networks(
    observation_size: int,
    action_size: int,
    # this is the obs normalization
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    hidden_layer_sizes: Sequence[int] = (256, 256),
    n_critics: int = 2,
    # policy_network_layer_norm: bool = False,
    # q_network_layer_norm: bool = False,
) -> sac_networks.SACNetworks:
    actor = AVGActor(action_size, tuple(hidden_layer_sizes))
    q = AVGQ(tuple(hidden_layer_sizes), n_critics)
    return sac_networks.SACNetworks(
        policy_network=networks.FeedForwardNetwork(
            init=lambda k: actor.init(k, jnp.zeros((1, observation_size))),
            apply=lambda norm, p, obs: actor.apply(p, preprocess_observations_fn(obs, norm))),
        q_network=networks.FeedForwardNetwork(
            init=lambda k: q.init(k, jnp.zeros((1, observation_size)), jnp.zeros((1, action_size))),
            apply=lambda norm, p, obs, act: q.apply(p, preprocess_observations_fn(obs, norm), act)),
        parametric_action_distribution=NormalTanH(event_size=action_size),
        
    )
#### end of SAC -AVG tranfer policy heads in brax training ####



#### AVG ported from torch to jax ####
def ortho_init(scale=1.41418, dtype=jnp.float32): # leakyrelu scale
    return nn.initializers.orthogonal(scale, dtype=dtype)

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activate_final: int = True
    layer_norm: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=ortho_init(dtype=self.dtype))(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm(dtype=self.dtype, param_dtype=self.dtype)(x)
                x = nn.leaky_relu(x, negative_slope=0.01)
        return x

class Actor(nn.Module):
    net_params: dict # hidden dims AVG
    action_dim: int
    layer_norm: bool = False
    dtype: Any = jnp.float32


    @nn.compact
    def __call__(self,
    key,
    x,
    temperature = 1.0
    ):
        outputs = MLP(self.net_params['mlp'], 
                    activate_final=True, 
                    layer_norm=self.layer_norm,
                    dtype=self.dtype)(x)
        # penultimate norm
        outputs = outputs / jnp.linalg.norm(outputs, axis=-1, keepdims=True)
        init = ortho_init(1.41418, self.dtype)
        mu = nn.Dense(self.action_dim, kernel_init=init, dtype=self.dtype)(outputs)
        log_std = nn.Dense(self.action_dim, kernel_init=init, dtype=self.dtype)(outputs)
        log_std = jnp.clip(log_std, -20, 2) # AVG ranges

        base_dist = tfp.distributions.MultivariateNormalDiag(loc=mu,
                                            scale_diag=jnp.exp(0.5 * log_std)*temperature) # std is exp(0.5 log_std); 0 scale is deterministic 

        dist = tfp.distributions.TransformedDistribution(distribution=base_dist,
                                            bijector=tfp.bijectors.Tanh())
        action_pre = base_dist.sample(seed=key)
        lprob = base_dist.log_prob(action_pre)
        lprob -= dist.bijector.forward_log_det_jacobian(action_pre, event_ndims=1)
        action = jnp.tanh(action_pre)
        action_info = {'mu': mu, 'log_std': log_std, 'dist': dist, 'lprob': lprob, 'action_pre': action_pre}
        return action, action_info

class Q(nn.Module):
    net_params: dict # hidden dims AVG
    obs_dim: int
    action_dim: int
    layer_norm: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, key,obs, action):
        inputs = jnp.concatenate([obs, action], -1)
        outputs = MLP(self.net_params['mlp'], 
                    activate_final=True, 
                    layer_norm=self.layer_norm,
                    dtype=self.dtype)(inputs)
        # penultimate norm
        outputs = outputs / jnp.linalg.norm(outputs, axis=-1, keepdims=True)
        init = ortho_init(1.41418, self.dtype)
        q = nn.Dense(1, kernel_init=init, dtype=self.dtype)(outputs)
        return jnp.squeeze(q, -1)