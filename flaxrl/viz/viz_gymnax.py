"""Inspired from https://github.com/RobertTLange/gymnax-blines/blob/main/visualize.py"""

import os
import argparse
import json
import gymnax
from gymnax.visualize import Visualizer
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
import numpy as np
from types import SimpleNamespace

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.make_env import make_env


def load_mode(train_args, model_path, which_ppo, observation_size, action_size, seed):

    if which_ppo == "ppo":
        from ppo import Actor

        abstract_actor = nnx.eval_shape(
            lambda: Actor(
                input_dim=observation_size,
                hidden_dim=train_args.actor_hidden_dim,
                num_layers=train_args.actor_num_layers,
                output_dim=action_size,
                rngs=nnx.Rngs(seed),
            )
        )
    elif which_ppo == "ppo_cnn":
        from ppo_cnn import Actor

        abstract_actor = nnx.eval_shape(
            lambda: Actor(
                in_features=observation_size[-1],
                output_dim=action_size,
                rngs=nnx.Rngs(seed),
            )
        )
    elif which_ppo == "ppo_continuous":
        from ppo_continuous import Actor

        abstract_actor = nnx.eval_shape(
            lambda: Actor(
                input_dim=observation_size,
                hidden_dim=train_args.actor_hidden_dim,
                num_layers=train_args.actor_num_layers,
                output_dim=action_size,
                log_std_init=train_args.log_std_init,
                rngs=nnx.Rngs(seed),
            )
        )

    graphdef, abstract_state = nnx.split(abstract_actor)
    checkpoint_path = os.path.abspath(f"{model_path}/policy")
    checkpointer = ocp.StandardCheckpointer()
    actor_state = checkpointer.restore(checkpoint_path, abstract_state)

    actor = nnx.merge(graphdef, actor_state)
    return actor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="Folder that contains the nets and the config")
    parser.add_argument("--output", default="rollouts", help="Output GIF path")
    parser.add_argument("--seed", type=int, default=120, help="Rollout seed")
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--which-ppo", type=str, default="ppo")
    args = parser.parse_args()
    # Load the model
    args_path = f"{args.model_path}/args.json"
    with open(args_path) as f:
        train_args = json.load(f)
    train_args = SimpleNamespace(**train_args)

    env = make_env(train_args)  # TODO save normalization stats
    env_, env_params = gymnax.make(train_args.env_name)
    # policy
    policy = load_mode(
        train_args, args.model_path, args.which_ppo, env.observation_size, env.action_size, args.seed
    )
    # collect steps
    key = jax.random.key(args.seed)
    key, reset_key = jax.random.split(key)
    reset_key = jax.random.split(reset_key, 1)
    obs, env_state = env.reset(reset_key)
    step = 0
    state_seq, reward_seq = [], []
    while True:
        key, key_step = jax.random.split(key)
        key_step = jax.random.split(key_step, 1)
        action = policy.get_action(obs)
        next_obs, next_env_state, reward, terminated, truncated, _ = env.step(key_step, env_state, action)
        done = terminated or truncated
        reward_seq.append(reward.squeeze(0))
        state_seq.append(jax.tree.map(lambda x: x.squeeze(0), env_state.env_state))
        if done or step == args.max_frames:
            break
        else:
            env_state = next_env_state
            obs = next_obs
    cum_rewards = np.cumsum(reward_seq)
    vis = Visualizer(env_, env_params, state_seq, cum_rewards)
    os.makedirs(f"{args.output}/{train_args.env_name}", exist_ok=True)
    vis.animate(f"{args.output}/{train_args.env_name}/ppo.gif")
