import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import numpy as np
import orbax.checkpoint as ocp
from brax.io import html
import imageio.v2 as imageio
from envs.brax import BraxInterface
from envs.wrappers import VecWrapper
from flax import nnx


def load_mode(train_args, model_path, which_ppo, observation_size, action_size, seed):

    if which_ppo == "ppo_continuous":
        from ppo_continuous import Actor

        abstract_actor = Actor(
            input_dim=observation_size,
            hidden_dim=train_args.actor_hidden_dim,
            num_layers=train_args.actor_num_layers,
            output_dim=action_size,
            log_std_init=train_args.log_std_init,
            rngs=nnx.Rngs(seed),
        )
    elif which_ppo == "ppo_rnn":
        from ppo_rnn import Actor

        abstract_actor = nnx.eval_shape(
            lambda: Actor(
                input_dim=observation_size,
                hidden_dim=train_args.actor_hidden_dim,
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

    # env
    env = BraxInterface.make(env_name=train_args.env_name, backend=train_args.backend)
    env = VecWrapper(env)
    if train_args.normalize_obs:
        from envs.wrappers import NormalizeVecObservationEval

        normalization_state = np.load(os.path.abspath(f"{args.model_path}/obs_normalization.npz"))
        env = NormalizeVecObservationEval(
            env, obs_mean=normalization_state["mean"], obs_var=normalization_state["var"]
        )
    print("++ ENV LOADED")
    # Policy
    policy = load_mode(
        train_args, args.model_path, args.which_ppo, env.observation_size, env.action_size, args.seed
    )
    if "rnn" in args.which_ppo:
        lstm_carry = policy.initialize_carry(1, train_args.actor_hidden_dim)

        def get_action(carry, x):
            return policy.get_action(carry, x)

    else:
        lstm_carry = None

        def get_action(carry, x):
            return (carry, policy.get_action(x))

    print("++ POLICY LOADED")
    # collect steps
    key = jax.random.key(args.seed)
    key, reset_key = jax.random.split(key)
    reset_key = jax.random.split(reset_key, 1)
    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)
    obs, env_state = jitted_reset(reset_key)
    step = 0
    state_seq = []
    while True:
        key, key_step = jax.random.split(key)
        key_step = jax.random.split(key_step, 1)
        lstm_carry, action = get_action(lstm_carry, obs)
        next_obs, next_env_state, reward, terminated, truncated, _ = jitted_step(key_step, env_state, action)
        done = terminated or truncated
        state_seq.append(jax.tree.map(lambda x: x.squeeze(0), env_state.pipeline_state))
        if done or step == args.max_frames:
            break
        else:
            env_state = next_env_state
            obs = next_obs
            step += 1
    print("++ STATES COLLECTED")
    # Rendering
    html_str = html.render(
        env.unwrapped.env.sys.tree_replace({"opt.timestep": env.unwrapped.env.dt}),
        state_seq,
    )
    os.makedirs(f"{args.output}/{train_args.env_name}", exist_ok=True)
    with open(f"{args.output}/{train_args.env_name}/ppo.html", "w") as f:
        f.write(html_str)
