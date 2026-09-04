import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import numpy as np
import mediapy as media
import orbax.checkpoint as ocp
from envs.mujoco_playground import PlaygroundInterface
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
    elif which_ppo == "ppo_cnn":
        pass

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
    env = PlaygroundInterface.make(env_name=train_args.env_name)
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
        action = policy.get_action(obs)
        next_obs, next_env_state, reward, terminated, truncated, _ = jitted_step(key_step, env_state, action)
        done = terminated or truncated
        state_seq.append(jax.tree.map(lambda x: x.squeeze(0), env_state))
        if done or step == args.max_frames:
            break
        else:
            env_state = next_env_state
            obs = next_obs
            step += 1
    print("++ STATES COLLECTED")
    # Rendering
    frames = env.unwrapped.env.render(state_seq)
    os.makedirs(f"{args.output}/{train_args.env_name}", exist_ok=True)
    fps = 1.0 / env.unwrapped.env.dt / 2
    print(fps)
    media.write_video(f"{args.output}/{train_args.env_name}/vid.mp4", frames, fps=fps)
    # media.show_video(frames, fps=1.0 / env.unwrapped.env.dt)
