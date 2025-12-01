"""Tests for evaluator wrapper without flax dependency."""

import jax
import jax.numpy as jnp

from gymnax.experimental import rollout


def mlp_forward(params, x, key):
    """Tiny MLP forward: (3,) -> (1,) with tanh output."""
    W1, b1 = params["W1"], params["b1"]
    W2, b2 = params["W2"], params["b2"]
    x = jnp.tanh(x @ W1 + b1)
    x = jnp.tanh(x @ W2 + b2)
    return x.squeeze()


def test_rollout():
    """Test rollout wrapper."""
    key = jax.random.key(0)
    key1, key2 = jax.random.split(key)
    policy_params = {
        "W1": jax.random.normal(key1, (3, 8)) * 0.1,
        "b1": jnp.zeros((8,)),
        "W2": jax.random.normal(key2, (8, 1)) * 0.1,
        "b2": jnp.zeros((1,)),
    }
    manager = rollout.RolloutWrapper(
        mlp_forward, env_name="Pendulum-v1", num_env_steps=200
    )

    # Test simple single episode rollout
    obs, _, _, _, _, _ = manager.single_rollout(key, policy_params)
    assert obs.shape == (200, 3)

    # Test multiple rollouts for same network (different random numbers)
    key_batch = jax.random.split(key, 10)
    obs, _, _, _, _, _ = manager.batch_rollout(key_batch, policy_params)
    assert obs.shape == (10, 200, 3)

    # Test multiple rollouts for different networks
    batch_params = jax.tree.map(
        lambda x: jnp.tile(x, (5, 1)).reshape(5, *x.shape), policy_params
    )
    # print(jax.tree.map(lambda x: x.shape, policy_params))
    # print(jax.tree.map(lambda x: x.shape, batch_params))
    (
        obs,
        _,
        _,
        _,
        _,
        _,
    ) = manager.population_rollout(key_batch, batch_params)
    assert obs.shape == (5, 10, 200, 3)
