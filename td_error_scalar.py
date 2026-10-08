import flax.struct
import jax
from jax import numpy as jnp

@flax.struct.dataclass
class RunningStats:
    n: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray

def init_running_stats():
    z = jnp.zeros(())
    return RunningStats(n=z, m=z, s=z)

def update_running_stats(rs, x):
    n = rs.n + 1.0
    prev_m = rs.m
    m = prev_m + (x - prev_m) / n
    s = rs.s + (x - prev_m) * (x - m)
    return RunningStats(n=n, m=m, s=s)

def rs_mean(rs):
    return jnp.where(rs.n > 0, rs.m, 0.0)

def rs_variance(rs):
    return jnp.where(rs.n > 0, rs.s / rs.n, 0.0)

@flax.struct.dataclass
class TDErrorScaler:
    gamma_rms: RunningStats
    return_sq_rms: RunningStats
    reward_rms: RunningStats
    return_rms: RunningStats
    G: jnp.ndarray

def init_td_error_scaler():
    return TDErrorScaler(
        gamma_rms=init_running_stats(),
        return_sq_rms=init_running_stats(),
        reward_rms=init_running_stats(),
        return_rms=init_running_stats(),
        G=jnp.zeros(()),
    )

def update_rs(rs, x, done):
    updated_rs = update_running_stats(rs, x)
    return jax.tree.map(lambda new, old: jnp.where(done, new, old), updated_rs, rs)

def update_td_error_scaler(state, reward, discount, done):
    G = state.G + reward
    gamma = jnp.where(done, 0.0, discount)
    return state.replace(
        reward_rms=update_running_stats(state.reward_rms, reward),
        gamma_rms=update_running_stats(state.gamma_rms, gamma),
        return_sq_rms=update_rs(state.return_sq_rms, G ** 2, done),
        return_rms=update_rs(state.return_rms, G, done),
        G=jnp.where(done, 0.0, G),
    )

def sigma(state):
    variance = jnp.maximum(
        rs_variance(state.reward_rms)
        + rs_variance(state.gamma_rms) * rs_mean(state.return_sq_rms),
        1e-4,
    )
    no_return_yet = (variance <= 0.01) & (state.return_sq_rms.n == 0)
    return jnp.where(no_return_yet, 1.0, jnp.sqrt(variance))
  