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
from envs.make_env import make_env
from flax import nnx
from tensorboardX import SummaryWriter


@dataclass
class Args:
    # Environment
    env_type: str = "gymnax"
    """ gymnax """
    env_name: str = "CartPole-v1"
    """ Discrete envs only """
    normalize_obs: bool = False
    """ Normalize the observations if True"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
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
    num_steps: int = 50
    """ Max episode steps"""
    buffer_size: int = 5000
    """ Buffer size"""
    batch_size: int = 64
    """ Batch size """
    n_epochs: int = 3
    """ Number of training epochs"""
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
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed = nnx.Sequential(nnx.Linear(input_dim, hidden_dim, rngs=rngs), nnx.relu)
        self.lstm = nnx.LSTMCell(in_features=hidden_dim, hidden_features=hidden_dim, rngs=rngs)
        self.qnet = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, carry, obs):
        x = self.embed(obs)
        carry, x = self.lstm(carry, x)
        values = self.qnet(x)
        return carry, values

    def sequence(self, carry, obs, episode_start):
        x = self.embed(obs)

        def lstm_t(carry, xs):
            x, episode_start = xs
            carry = jax.tree.map(
                lambda x: jnp.where(episode_start[:, None].astype(bool), jnp.zeros_like(x), x), carry
            )
            carry, x = self.lstm(carry, x)
            return carry, x

        carry, x = nnx.scan(
            lstm_t,
            in_axes=(nnx.Carry, 0),
            out_axes=(nnx.Carry, 0),
        )(carry, (x, episode_start))

        values = self.qnet(x)
        return values

    def get_action(self, carry, obs) -> jnp.ndarray:
        x = self.embed(obs)
        carry, x = self.lstm(carry, x)
        values = self.qnet(x)
        return carry, jnp.argmax(values, axis=-1)

    def initialize_carry(self, num_envs, hidden_dim):
        return self.lstm.initialize_carry((num_envs, hidden_dim), rngs=nnx.Rngs(0))


# -------- States --------
class RolloutState(NamedTuple):
    """The necessary information to step the environment"""

    obs: jax.Array
    env_state: Any
    lstm_carry: Any
    episode_start: jax.Array
    step: int
    key: jax.Array


class TimeStep(NamedTuple):
    """the transition saved to the replay buffer"""

    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    episode_start: jax.Array


class EpisodeStats(NamedTuple):
    """Episodic statistics"""

    episode_return: jax.Array
    episode_length: jax.Array
    ep_done: jax.Array


def train(args):
    # Vec training params
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    assert args.train_freq % args.num_envs == 0, "args.train_freq % args.num_envs != 0"
    # Rng keys
    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)
    key, reset_key, eval_key = jax.random.split(key, 3)
    # TODO Do I need to keep this one
    reset_key = jax.random.split(reset_key, args.num_envs)
    # Import the environment
    env = make_env(args)
    # Prepare networks + optimizer
    qnetwork = Qnetwork(
        input_dim=env.observation_size,
        hidden_dim=args.hidden_dim,
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
    log_dir = f"{args.work_dir}/DQN-{run_name}"
    # Replay buffer
    rb_fun = fbx.make_trajectory_buffer(
        max_length_time_axis=args.buffer_size,
        min_length_time_axis=args.num_steps,
        sample_batch_size=args.batch_size,
        add_batch_size=args.num_envs,
        sample_sequence_length=args.num_steps,
        period=args.num_steps,
    )
    buffer = rb_fun.init(  # initialize the buffer
        TimeStep(
            obs=jnp.zeros(env.observation_size),
            action=jnp.array(1, dtype=jnp.int32),
            reward=jnp.array(1.0),
            done=jnp.array(False),
            episode_start=jnp.array(True),
        )
    )
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

        qnetwork, optimizer, target_qnetwork, buffer, step, key = update_state

        # ------ Collect env steps ------
        def collect_rollout(qnetwork: nnx.Module, rollout_state: RolloutState):

            # env_one_step : one step for each environment
            def env_one_step(carry, x):
                # Last rollout state
                obs, state, lstm_carry, episode_start, step, key = carry
                # Reset hidden states + random keys
                lstm_carry = jax.tree.map(
                    lambda x: jnp.where(episode_start[:, None].astype(bool), jnp.zeros_like(x), x), lstm_carry
                )
                key, key_eps, key_sample, key_step = jax.random.split(key, 4)
                key_step = jax.random.split(key_step, args.num_envs)
                key_sample = jax.random.split(key_sample, args.num_envs)
                # Get action
                lstm_carry, q_vals = qnetwork(carry=lstm_carry, obs=obs)
                action = jnp.argmax(q_vals, axis=-1)
                eps = eps_scheduler(t=step)
                p_explore = jax.random.uniform(key_eps, args.num_envs)
                action = jnp.where(p_explore < eps, env.sample(key_sample), action)
                # step the env
                next_obs, next_state, reward, terminated, truncated, info = env.step(key_step, state, action)
                done = jnp.logical_or(terminated, truncated)
                step += args.num_envs
                # Record its episodic return and length
                episode_stats = EpisodeStats(
                    episode_return=info["episode_return"], episode_length=info["episode_length"], ep_done=done
                )
                # Store date needed for training
                timestep = TimeStep(
                    obs=obs, action=action, reward=reward, done=done, episode_start=episode_start
                )
                # Prepare the next rollout_state
                rollout_state = RolloutState(
                    obs=next_obs,
                    env_state=next_state,
                    lstm_carry=lstm_carry,
                    episode_start=done,
                    step=step,
                    key=key,
                )
                return rollout_state, (timestep, episode_stats)

            rollout_state, (transitions, episode_stats) = jax.lax.scan(
                f=env_one_step, init=rollout_state, xs=None, length=args.num_steps
            )
            return rollout_state, transitions, episode_stats

        # We always restart the env to 'safely' initialize the LSTM hidden state to zero during training
        # key, reset_key = jax.random.split(key)
        # reset_key = jax.random.split(reset_key, args.num_envs)
        # Reset env + lstm + rollout
        obs, env_state = env.reset(key=reset_key)
        lstm_carry = qnetwork.initialize_carry(args.num_envs, args.hidden_dim)
        rollout_state = RolloutState(
            obs=obs,
            env_state=env_state,
            lstm_carry=lstm_carry,
            episode_start=jnp.zeros(args.num_envs).astype(bool),
            step=step,
            key=key,
        )
        rollout_state, timesteps, episode_stats = collect_rollout(qnetwork, rollout_state)
        key = rollout_state.key
        step = rollout_state.step

        # The replay buffer is if shape (add_batch_size,max_length_time_axis)
        timesteps = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), timesteps)
        buffer = rb_fun.add(buffer, timesteps)

        # ------ Update the q network ------
        @nnx.scan(length=args.n_epochs, in_axes=(nnx.Carry, None), out_axes=(nnx.Carry, 0))
        def update_qnetwork(carry, x):
            qnetwork, optimizer, key = carry
            key, key_sample = jax.random.split(key)
            # Sample a batch
            batch = rb_fun.sample(buffer, key_sample).experience
            batch = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), batch)
            # Compute targets
            lstm_carry_init = target_qnetwork.initialize_carry(args.batch_size, args.hidden_dim)
            q_vals_next = target_qnetwork.sequence(
                lstm_carry_init, obs=batch.obs, episode_start=batch.episode_start
            )
            q_vals_next = jnp.max(q_vals_next, axis=-1)
            targets = batch.reward[:-1] + args.gamma * (1 - batch.done[:-1]) * q_vals_next[1:]

            # dqn_loss: dqn loss
            def dqn_loss(qnetwork):
                q_values = jnp.take_along_axis(
                    arr=qnetwork.sequence(lstm_carry_init, obs=batch.obs, episode_start=batch.episode_start),
                    indices=jnp.expand_dims(batch.action, axis=-1),
                    axis=-1,
                ).squeeze(axis=-1)
                return optax.l2_loss(targets, q_values[:-1]).mean()

            # Update the q_network
            loss, grads = nnx.value_and_grad(dqn_loss)(qnetwork)
            optimizer.update(qnetwork, grads)
            carry = qnetwork, optimizer, key
            return carry, loss

        # Decide if it is time to update
        update_event = jnp.logical_and(rb_fun.can_sample(buffer), rollout_state.step >= args.learning_starts)

        # It is important to use nnx.cond not jax.lax.cond. Otherwise the network will not be updated
        # Rule of thumb: if the transformation modified the params of the network\optimize,
        # always use nnx. transformations
        (qnetwork, optimizer, key), loss = nnx.cond(
            update_event,
            update_qnetwork,
            lambda x, y: (x, jnp.array([0.0] * args.n_epochs)),
            (qnetwork, optimizer, key),
            buffer,
        )
        loss = jnp.mean(loss)
        # ------ Update the target q network ------
        polyak_update(qnetwork, target_qnetwork)
        # ------ Prepare the next updating step ------
        # update rollout_state
        update_state = qnetwork, optimizer, target_qnetwork, buffer, step, key
        # Metrics: we save "update_event" to log the losses, "done" to log the episode stats
        metrics = (
            loss,
            update_event,
            episode_stats.episode_return,
            episode_stats.episode_length,
            episode_stats.ep_done,
        )
        return update_state, metrics

    # ------ Run DQN ------
    update_state = qnetwork, optimizer, target_qnetwork, buffer, 0, key
    start_time = time.perf_counter()
    update_state, metrics = update_step(update_state, None)
    jax.block_until_ready(metrics)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time / 60:.2f} min ({elapsed_time:.2f} sec)")
    qnetwork, _, _, rollout_state, *_ = update_state
    # ------ Evaluate + tensorboard logging + checkpoints ------
    if args.eval:
        evaluate(args, qnetwork, rollout_state, eval_key)
    if args.log:
        tb_logger(args, metrics, log_dir, num_updates)
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


def evaluate(args, qnetwork, rollout_state, eval_key):
    eval_env = make_env(args, eval=True, rollout_state=rollout_state)
    eval_key, reset_key = jax.random.split(eval_key)
    reset_key = jax.random.split(reset_key, args.num_eval_ep)
    obs, env_state = eval_env.reset(reset_key)
    # Initialize hidden state
    eval_lstm_carry = qnetwork.initialize_carry(args.num_eval_ep, args.hidden_dim)
    # Initialize metrics
    eval_ep_returns = jnp.zeros(args.num_eval_ep)
    eval_ep_lengths = jnp.zeros(args.num_eval_ep, dtype=jnp.int32)
    ep_dones = jnp.zeros(args.num_eval_ep, dtype=jnp.bool_)
    evaluation_state = (obs, env_state, eval_ep_returns, eval_ep_lengths, ep_dones, eval_lstm_carry, eval_key)

    # Stops once all envs are done
    def cond_fun(evaluation_state):
        return ~jnp.all(evaluation_state[-3])

    # Step the eval envs
    def eval_fun(evaluation_state):
        obs, env_state, eval_ep_returns, eval_ep_lengths, ep_dones, eval_lstm_carry, eval_key = (
            evaluation_state
        )
        eval_key, key_step = jax.random.split(eval_key)
        key_step = jax.random.split(key_step, args.num_eval_ep)
        eval_lstm_carry, action = qnetwork.get_action(eval_lstm_carry, obs)
        next_obs, next_state, reward, terminated, truncated, _ = eval_env.step(key_step, env_state, action)
        active = ~ep_dones.astype(bool)
        eval_ep_returns = eval_ep_returns + jnp.where(active, reward, 0.0)
        eval_ep_lengths = eval_ep_lengths + active.astype(jnp.int32)
        ep_dones = ep_dones.astype(bool) | terminated.astype(bool) | truncated.astype(bool)
        return next_obs, next_state, eval_ep_returns, eval_ep_lengths, ep_dones, eval_lstm_carry, eval_key

    # Evaluation loop
    _, _, eval_ep_returns, eval_ep_lengths, ep_dones, *_ = jax.lax.while_loop(
        cond_fun=cond_fun, body_fun=eval_fun, init_val=evaluation_state
    )
    # Display the results
    eval_ep_returns, eval_ep_lengths = jax.device_get((eval_ep_returns, eval_ep_lengths))
    mean_return = float(eval_ep_returns.mean())
    std_return = float(eval_ep_returns.std())
    mean_length = float(eval_ep_lengths.mean())
    std_length = float(eval_ep_lengths.std())
    print(f"Evaluation over {args.num_eval_ep} episodes")
    print(f"Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Episode length: {mean_length:.1f} ± {std_length:.1f}")


def tb_logger(args, metrics, log_dir, num_updates):
    metrics = jax.device_get(metrics)
    losses, update_events, episode_returns, episode_lengths, ep_dones = metrics
    losses = losses.reshape(-1)
    update_events = update_events.reshape(-1)
    ep_dones = ep_dones.reshape(-1)
    episode_returns = episode_returns.reshape(-1)
    episode_lengths = episode_lengths.reshape(-1)
    writer = SummaryWriter(log_dir)
    completed_steps = np.flatnonzero(ep_dones)
    for start in range(0, len(completed_steps), args.log_every):
        episode_steps = completed_steps[start : start + args.log_every]
        step = int(episode_steps[-1]) + 1
        mean_episode_return = episode_returns[episode_steps].mean()
        mean_episode_length = episode_lengths[episode_steps].mean()
        writer.add_scalar("rollout/ep_reward", float(mean_episode_return), step)
        writer.add_scalar("rollout/ep_length", float(mean_episode_length), step)
    updates_steps = np.flatnonzero(update_events)
    for start in range(0, len(updates_steps), args.log_every):
        train_steps = updates_steps[start : start + args.log_every]
        step = (int(train_steps[-1]) + 1) * args.num_envs * args.num_steps
        loss = losses[train_steps].mean()
        writer.add_scalar("losses/q_loss", float(loss), step)
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
