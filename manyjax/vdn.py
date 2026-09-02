import datetime
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import flashbax as fbx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
from envs.make_env import make_marl_env
from flax import nnx
from tensorboardX import SummaryWriter


@dataclass
class Args:
    # Environment
    env_type: str = "smax"
    """ smax, mpe """
    env_name: str = "3m"
    """ Discrete envs only """
    reward_aggr: str = "mean"
    """ Aggregate rewards: mean, sum, or none"""
    # Network
    hidden_dim: int = 64
    """ Hidden dimension"""
    num_layers: int = 1
    """ Number of hidden layers"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    num_envs: int = 10
    """num envs"""
    buffer_size: int = 5000
    """ Buffer size"""
    batch_size: int = 64
    """ Batch size """
    train_freq: int = 10
    """ Train the network every train_freq environment steps"""
    learning_starts: int = 70
    """ Number of env steps to initialize the replay buffer"""
    optimizer: str = "adam"
    """ The optimizer"""
    learning_rate: float = 0.0008
    """ Learning rate for the actor"""
    gamma: float = 0.99
    """ Discount factor"""
    clip_gradients: float = -1
    """ Disable gradient clipping when <= 0; otherwise clip at this value"""
    target_network_update_freq: int = 10
    """ Update the target network every target_network_update_freq step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient for target network update"""
    start_e: float = 1
    """ The starting value of epsilon, for exploration"""
    end_e: float = 0.05
    """ The end value of epsilon, for exploration"""
    exploration_fraction: float = 0.05
    """ Fraction of total_timesteps over which epsilon decreases from start_e to end_e"""
    seed: int = 1
    """ Random seed"""
    # Logging and eval
    log: bool = True
    """ Log data at the end """
    eval: bool = True
    """ evaluate at the end """
    save_model: bool = False
    """ If True, save the weights of the agents and hyperparameters"""
    work_dir: str = "runs"
    """ Folder to save logs, weights ..."""
    exp_name: str = "v1"
    """ Used for logging"""
    log_every: int = 10
    """ Logging steps """
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""


# -------- Q(s,a) network --------
class Qnetwork(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        layers = [nnx.Linear(input_dim, hidden_dim, rngs=rngs), nnx.relu]
        for _ in range(num_layers - 1):
            layers.extend([nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), nnx.relu])
        layers.append(nnx.Linear(hidden_dim, output_dim, rngs=rngs))
        self.qnet = nnx.Sequential(*layers)

    def __call__(self, obs: jnp.ndarray, avail_actions: jnp.ndarray):
        qvals = self.qnet(obs)
        qvals = jnp.where(avail_actions, qvals, -1e10)
        return qvals


# -------- States --------
class RolloutState(NamedTuple):
    """The necessary information to step the environment"""

    obs: jax.Array
    env_state: Any
    avail_actions: jax.Array
    step: int
    key: jax.Array


class EpisodeStats(NamedTuple):
    """Episodic statistics."""

    episode_return: jax.Array
    episode_length: jax.Array
    battle_won: jax.Array
    ep_done: jax.Array


class TimeStep(NamedTuple):
    """the transition saved to the replay buffer"""

    obs: jax.Array
    avail_actions: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array


def train(args):  # noqa: PLR0915
    # Rng keys
    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)
    key, reset_key, eval_key = jax.random.split(key, 3)
    # Import the environment
    env = make_marl_env(args)
    # Vec training params
    num_updates = args.total_timesteps // args.num_envs
    assert args.train_freq % args.num_envs == 0, "args.train_freq % args.num_envs != 0"
    assert args.target_network_update_freq % args.num_envs == 0, (
        "args.target_network_update_freq % args.num_envs == 0 "
    )
    assert args.reward_aggr != "none", "VDN works only for common reward"
    # Prepare networks + optimizer
    qnetwork = Qnetwork(
        input_dim=env.observation_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=env.action_size,
        rngs=rngs,
    )
    target_qnetwork = nnx.clone(qnetwork)
    optim = getattr(optax, args.optimizer)
    optimizer = optim(learning_rate=args.learning_rate)
    if args.clip_gradients > 0:
        optimizer = optax.chain(optax.clip_by_global_norm(args.clip_gradients), optimizer)
    optimizer = nnx.Optimizer(qnetwork, optimizer, wrt=nnx.Param)
    # Logging
    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_name}__{args.exp_name}__{time_token}"
    log_dir = f"{args.work_dir}/VDN-{run_name}"
    # Reset the env
    reset_key = jax.random.split(reset_key, args.num_envs)
    obs, _, env_state = env.reset(key=reset_key)
    avail_actions = env.get_avail_actions(env_state)
    # Replay buffer
    rb_fun = fbx.make_flat_buffer(
        max_length=args.buffer_size,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
        add_batch_size=args.num_envs,
    )
    buffer = rb_fun.init(  # initialize the buffer
        TimeStep(
            obs=jnp.zeros((env.num_agents, env.observation_size)),
            avail_actions=jnp.zeros((env.num_agents, env.action_size)).astype(bool),
            action=jnp.zeros(env.num_agents, dtype=jnp.int32),
            reward=jnp.array(1.0),
            done=jnp.array(True),
        )
    )

    rollout_state = RolloutState(obs=obs, env_state=env_state, avail_actions=avail_actions, step=0, key=key)
    # Eps scheduler + polyak
    eps_scheduler = partial(
        linear_schedule_, args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps
    )
    polyak_update = partial(polyak_update_, tau=args.polyak)

    # update_step: step the envs + periodically update the qnetwork
    @nnx.jit
    @nnx.scan(
        length=num_updates,
        in_axes=(nnx.Carry, None),
        out_axes=(nnx.Carry, 0),
    )
    def update_step(update_state: tuple, _):  # noqa: PLR0915

        qnetwork, optimizer, target_qnetwork, rollout_state, buffer = update_state

        # ------ Collect env steps ------
        obs, env_state, avail_actions, step, key = rollout_state
        # Keys for exploration and env.step
        key, key_eps, key_sample, key_step = jax.random.split(key, 4)
        key_step = jax.random.split(key_step, args.num_envs)
        key_sample = jax.random.split(key_sample, args.num_envs)
        # Get action
        q_vals = qnetwork(obs=obs, avail_actions=avail_actions)
        action = jnp.argmax(q_vals, axis=-1)
        eps = eps_scheduler(t=step)
        p_explore = jax.random.uniform(key_eps, (args.num_envs, env.num_agents))
        action = jnp.where(p_explore < eps, env.sample(key_sample, env_state), action)
        # step the env
        next_obs, _, next_env_state, reward, terminated, truncated, info = env.step(
            key_step, env_state, action
        )
        next_avail_actions = env.get_avail_actions(next_env_state)
        step += args.num_envs
        done = jnp.logical_or(terminated, truncated).squeeze(-1)
        # Record episodic return and length
        episode_stats = EpisodeStats(
            episode_return=info["episode_return"],
            episode_length=info["episode_length"],
            battle_won=reward >= 1,
            ep_done=info["__all__"],
        )
        reward = reward.squeeze(-1)  # vdn works for common reward only
        # Add the step in to the replay buffer
        timestep = TimeStep(obs=obs, avail_actions=avail_actions, action=action, reward=reward, done=done)
        buffer = rb_fun.add(buffer, timestep)

        # ------ Update the q network ------
        def update_qnetwork(qnetwork, optimizer, key_sample):
            # Sample a batch
            batch = rb_fun.sample(buffer, key_sample).experience
            # Compute targets
            qvals_agents_next = target_qnetwork(
                obs=batch.second.obs, avail_actions=batch.second.avail_actions
            )
            qvals_agents_next = jnp.max(qvals_agents_next, axis=-1)
            qvals_vdn_next = jnp.sum(qvals_agents_next, axis=-1)
            targets = batch.first.reward + args.gamma * (1 - batch.first.done) * qvals_vdn_next

            def vdn_loss(qnetwork):
                qvals = jnp.take_along_axis(
                    arr=qnetwork(obs=batch.first.obs, avail_actions=batch.first.avail_actions),
                    indices=jnp.expand_dims(batch.first.action, axis=-1),
                    axis=-1,
                ).squeeze(axis=-1)
                qvals_vdn = jnp.sum(qvals, axis=-1)
                return optax.l2_loss(targets, qvals_vdn).mean()

            # Update the q_network
            loss, grads = nnx.value_and_grad(vdn_loss)(qnetwork)
            optimizer.update(qnetwork, grads)
            return loss

        # Decide if it is time to update
        update_event = jnp.logical_and(
            step > jnp.maximum(args.batch_size, args.learning_starts), step % args.train_freq == 0
        )
        key, key_sample = jax.random.split(key)
        # It is important to use nnx.cond not jax.lax.cond. Otherwise the network will not be updated
        # Rule of thumb: if the transformation modified the params of the network\optimize,
        # always use nnx. transformations
        loss = nnx.cond(
            update_event,
            update_qnetwork,
            lambda *_: jnp.array(0.0),
            qnetwork,
            optimizer,
            key_sample,
        )
        # ------ Update the target q network ------
        # Decide if it is time to update the target network
        update_target_qnet_event = step % args.target_network_update_freq == 0
        nnx.cond(
            update_target_qnet_event,
            polyak_update,
            lambda *_: None,
            qnetwork,
            target_qnetwork,
        )
        # ------ Prepare the next updating step ------
        # update rollout_state
        rollout_state = RolloutState(
            obs=next_obs, env_state=next_env_state, avail_actions=next_avail_actions, step=step, key=key
        )
        update_state = qnetwork, optimizer, target_qnetwork, rollout_state, buffer
        # Metrics: we save "update_event" to log the losses, "done" to log the episode stats
        metrics = (
            loss,
            update_event,
            done,
            episode_stats.episode_return,
            episode_stats.episode_length,
            episode_stats.battle_won,
            eps,
        )
        return update_state, metrics

    # ------ Run VDN ------
    update_state = qnetwork, optimizer, target_qnetwork, rollout_state, buffer
    start_time = time.perf_counter()
    update_state, metrics = update_step(update_state, None)
    jax.block_until_ready(metrics)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time / 60:.2f} min ({elapsed_time:.2f} sec)")
    qnetwork, _, _, rollout_state, _ = update_state
    # ------ Evaluate + tensorboard logging + checkpoints ------
    if args.eval:
        evaluate(args, qnetwork, eval_key)
    if args.log:
        tb_logger(args, metrics, log_dir)
    if args.save_model:
        qnetwork, *_ = update_state
        _, qnetwork_state = nnx.split(qnetwork)
        network_states = {"qnetwork": qnetwork_state}
        checkpoint_path = (Path(log_dir) / "networks").resolve()
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(checkpoint_path, network_states)
        print(f"Networks saved to {checkpoint_path}")
        with open(Path(log_dir) / "args.json", "w") as file:
            json.dump(vars(args), file, indent=2)


def evaluate(args, qnetwork, eval_key):
    eval_env = make_marl_env(args)
    eval_key, reset_key = jax.random.split(eval_key)
    reset_key = jax.random.split(reset_key, args.num_eval_ep)
    obs, _, env_state = eval_env.reset(reset_key)
    avail_actions = eval_env.get_avail_actions(env_state)
    # Initialize metrics
    eval_ep_returns = jnp.zeros(args.num_eval_ep)
    eval_ep_lengths = jnp.zeros(args.num_eval_ep, dtype=jnp.int32)
    ep_dones = jnp.zeros(args.num_eval_ep, dtype=jnp.bool_)
    eval_ep_battle_win = jnp.zeros(args.num_eval_ep, dtype=bool)
    evaluation_state = (
        obs,
        avail_actions,
        env_state,
        eval_ep_returns,
        eval_ep_battle_win,
        eval_ep_lengths,
        ep_dones,
        eval_key,
    )

    # Stops once all envs are done
    def cond_fun(evaluation_state):
        return ~jnp.all(evaluation_state[-2])

    # Step the eval envs
    def eval_fun(evaluation_state):
        (
            obs,
            avail_actions,
            env_state,
            eval_ep_returns,
            eval_ep_battle_win,
            eval_ep_lengths,
            ep_dones,
            eval_key,
        ) = evaluation_state
        eval_key, key_step = jax.random.split(eval_key)
        key_step = jax.random.split(key_step, args.num_eval_ep)
        q_vals = qnetwork(obs=obs, avail_actions=avail_actions)
        action = jnp.argmax(q_vals, axis=-1)
        next_obs, _, next_state, reward, _, _, infos = eval_env.step(key_step, env_state, action)
        avail_actions = eval_env.get_avail_actions(next_state)
        reward = jnp.mean(reward, axis=-1)
        active = ~ep_dones.astype(bool)
        eval_ep_returns = eval_ep_returns + jnp.where(active, reward, 0.0)
        eval_ep_lengths = eval_ep_lengths + active.astype(jnp.int32)
        ep_dones = ep_dones.astype(bool) | infos["__all__"].astype(bool)
        eval_ep_battle_win += (reward >= 1) & active & infos["__all__"].astype(bool)
        return (
            next_obs,
            avail_actions,
            next_state,
            eval_ep_returns,
            eval_ep_battle_win,
            eval_ep_lengths,
            ep_dones,
            eval_key,
        )

    # Evaluation loop
    _, _, _, eval_ep_returns, eval_ep_battle_win, eval_ep_lengths, ep_dones, _ = jax.lax.while_loop(
        cond_fun=cond_fun, body_fun=eval_fun, init_val=evaluation_state
    )
    # Print the results
    eval_ep_returns, eval_ep_battle_win, eval_ep_lengths = jax.device_get(
        (eval_ep_returns, eval_ep_battle_win, eval_ep_lengths)
    )
    mean_return = float(eval_ep_returns.mean())
    std_return = float(eval_ep_returns.std())
    mean_length = float(eval_ep_lengths.mean())
    std_length = float(eval_ep_lengths.std())
    print(f"Evaluation over {args.num_eval_ep} episodes")
    print(f"Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Episode length: {mean_length:.1f} ± {std_length:.1f}")
    if args.env_type == "smax":
        mean_battle_won = float(eval_ep_battle_win.mean())
        std_battle_won = float(eval_ep_battle_win.std())
        print(f"Battle won: {mean_battle_won:.2f} ± {std_battle_won:.2f}")


def tb_logger(args, metrics, log_dir):
    metrics = jax.device_get(metrics)
    losses, update_events, dones, episode_returns, episode_lengths, battle_win, epses = metrics
    losses = losses.reshape(-1)
    epses = epses.reshape(-1)
    update_events = update_events.reshape(-1)
    dones = dones.reshape(-1)
    episode_returns = episode_returns.reshape(-1)
    episode_lengths = episode_lengths.reshape(-1)
    if args.env_type == "smax":
        battle_win = jnp.mean(battle_win, axis=-1).reshape(-1)
    writer = SummaryWriter(log_dir)
    completed_steps = np.flatnonzero(dones)
    for start in range(0, len(completed_steps), args.log_every):
        episode_steps = completed_steps[start : start + args.log_every]
        step = int(episode_steps[-1]) + 1
        mean_episode_return = episode_returns[episode_steps].mean()
        mean_episode_length = episode_lengths[episode_steps].mean()
        writer.add_scalar("rollout/ep_reward", float(mean_episode_return), step)
        writer.add_scalar("rollout/ep_length", float(mean_episode_length), step)
        if args.env_type == "smax":
            mean_battle_won = battle_win[episode_steps].mean()
            writer.add_scalar("rollout/battle_won", float(mean_battle_won), step)
    updates_steps = np.flatnonzero(update_events)
    for start in range(0, len(updates_steps), args.log_every):
        train_steps = updates_steps[start : start + args.log_every]
        step = (int(train_steps[-1]) + 1) * args.num_envs
        loss = losses[train_steps].mean()
        writer.add_scalar("losses/q_loss", float(loss), step)
        writer.add_scalar("rollout/eps", float(epses[start]), step)
    writer.close()


def linear_schedule_(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return jnp.clip(slope * t + start_e, end_e)


# polyak_update
def polyak_update_(network, target_network, tau):
    params = nnx.state(network, nnx.Param)
    target_params = nnx.state(target_network, nnx.Param)

    updated_params = jax.tree.map(
        lambda target, online: (1.0 - tau) * target + tau * online, target_params, params
    )
    nnx.update(target_network, updated_params)


if __name__ == "__main__":
    train(tyro.cli(Args))
