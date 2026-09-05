"""Inspired from https://github.com/RobertTLange/gymnax-blines/blob/main/visualize.py"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnax
import jax
import numpy as np
import orbax.checkpoint as ocp
from envs.gymnax import GymnaxInterface
from envs.wrappers import VecWrapper
from flax import nnx
from gymnax.visualize import Visualizer


def load_mode(train_args, model_path, which_algo, observation_size, action_size):

    if which_algo == "ppo":
        from ppo import Actor

        abstract_actor = nnx.eval_shape(
            lambda: Actor(
                input_dim=observation_size,
                hidden_dim=train_args.actor_hidden_dim,
                num_layers=train_args.actor_num_layers,
                output_dim=action_size,
                rngs=nnx.Rngs(0),
            )
        )
    elif which_algo == "ppo_cnn":
        from ppo_cnn import Actor

        abstract_actor = nnx.eval_shape(
            lambda: Actor(
                in_features=observation_size[-1],
                output_dim=action_size,
                rngs=nnx.Rngs(0),
            )
        )
    elif which_algo == "ppo_continuous":
        from ppo_continuous import Actor

        abstract_actor = nnx.eval_shape(
            lambda: Actor(
                input_dim=observation_size,
                hidden_dim=train_args.actor_hidden_dim,
                num_layers=train_args.actor_num_layers,
                output_dim=action_size,
                log_std_init=train_args.log_std_init,
                rngs=nnx.Rngs(0),
            )
        )
    elif which_algo == "ppo_rnn":
        from ppo_rnn import Actor

        abstract_actor = nnx.eval_shape(
            lambda: Actor(
                input_dim=observation_size,
                hidden_dim=train_args.actor_hidden_dim,
                output_dim=action_size,
                log_std_init=train_args.log_std_init,
                rngs=nnx.Rngs(0),
            )
        )
    elif which_algo == "dqn":
        from dqn import Qnetwork

        abstract_actor = nnx.eval_shape(
            lambda: Qnetwork(
                input_dim=observation_size,
                hidden_dim=train_args.hidden_dim,
                num_layers=train_args.num_layers,
                output_dim=action_size,
                rngs=nnx.Rngs(0),
            )
        )
    elif which_algo == "dqn_cnn":
        from dqn_cnn import Qnetwork

        abstract_actor = nnx.eval_shape(
            lambda: Qnetwork(in_features=observation_size[-1], output_dim=action_size, rngs=nnx.Rngs(0))
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
    parser.add_argument("--which-algo", type=str, default="ppo")
    args = parser.parse_args()
    # Load the model
    args_path = f"{args.model_path}/args.json"
    with open(args_path) as f:
        train_args = json.load(f)
    train_args = SimpleNamespace(**train_args)

    env = GymnaxInterface.make(train_args.env_name)
    env = VecWrapper(env)
    if train_args.normalize_obs:
        from envs.wrappers import NormalizeVecObservationEval

        normalization_state = np.load(os.path.abspath(f"{args.model_path}/obs_normalization.npz"))
        env = NormalizeVecObservationEval(
            env, obs_mean=normalization_state["mean"], obs_var=normalization_state["var"]
        )
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    env_, env_params = gymnax.make(train_args.env_name)
    # policy
    policy = load_mode(train_args, args.model_path, args.which_algo, env.observation_size, env.action_size)
    if "rnn" in args.which_algo:
        lstm_carry = policy.initialize_carry(1, train_args.actor_hidden_dim)

        def get_action(carry, x):
            return policy.get_action(carry, x)

    else:
        lstm_carry = None

        def get_action(carry, x):
            return (carry, policy.get_action(x))

    # collect steps
    key = jax.random.key(args.seed)
    key, reset_key = jax.random.split(key)
    reset_key = jax.random.split(reset_key, 1)
    obs, env_state = jitted_reset(reset_key)
    step = 0
    state_seq, reward_seq = [], []
    while True:
        key, key_step = jax.random.split(key)
        key_step = jax.random.split(key_step, 1)
        lstm_carry, action = get_action(lstm_carry, obs)
        next_obs, next_env_state, reward, terminated, truncated, _ = jitted_step(key_step, env_state, action)
        done = terminated or truncated
        reward_seq.append(reward.squeeze(0))
        state_seq.append(jax.tree.map(lambda x: x.squeeze(0), env_state))
        if done or step == args.max_frames:
            break
        else:
            env_state = next_env_state
            obs = next_obs
    cum_rewards = np.cumsum(reward_seq)
    vis = Visualizer(env_, env_params, state_seq, cum_rewards)
    os.makedirs(f"{args.output}/{train_args.env_name}", exist_ok=True)
    vis.animate(f"{args.output}/{train_args.env_name}/ppo.gif")
