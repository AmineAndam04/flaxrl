import datetime
import json
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
from flax import nnx
from tensorboardX import SummaryWriter

from envs.make_env import make_env


@dataclass
class Args:
    # Environment
    env_type: str = "brax"
    """ gymnax, brax, playground """
    env_name: str = "hopper"
    """ Name of the environment """
    normalize_obs: bool = False
    """ Normalize the observations if True"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    max_episode_steps: int = 1000
    """ Maximum steps per episode"""
    clip_actions: bool = False
    """ Clip actions before env.step """
    backend: str = "generalized"  # brax
    """ For brax envs: generalized", positional, spring """
    impl: str = "jax"
    """ For playground: jax , warp"""
    # Network
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 32
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    num_envs: int = 10
    """num envs"""
    buffer_size: int = 5000
    """ Buffer size"""
    batch_size: int = 64
    """ Batch size """
    train_freq_critic: int = 10
    """ Train the critics every train_freq environment steps"""
    train_freq_actor: int = 20
    """ Train the actor every train_freq environment steps"""
    learning_starts: int = 70
    """ Number of env steps to initialize the replay buffer"""
    optimizer: str = "adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    gamma: float = 0.99
    """ Discount factor"""
    clip_gradients: float = -1
    """ Disable gradient clipping when <= 0; otherwise clip at this value"""
    target_network_update_freq: int = 10
    """ Update the target network every target_network_update_freq step in the environment"""
    polyak: float = 0.005
    """ Polyak coefficient for target network update"""
    act_noise: float = 0.1
    """ Stddev for Gaussian exploration noise added to policy at training time"""
    target_noise: float = 0.2
    """  Stddev of noise added to target policy"""
    noise_clip: float = 0.5
    """  Clip added noise to target policy"""
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


class Actor(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        action_low,
        action_high,
        *,
        rngs: nnx.Rngs,
    ):
        layers = [nnx.Linear(input_dim, hidden_dim, rngs=rngs), nnx.relu]
        for _ in range(num_layers - 1):
            layers.extend([nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), nnx.relu])
        self.enc = nnx.Sequential(*layers)

        self.mean = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        # action scaling
        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

    def __call__(self, obs: jnp.ndarray):
        x = self.enc(obs)
        mean = self.mean(x)
        return jnp.tanh(mean) * self.action_scale + self.action_bias


class Critic(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        layers = [nnx.Linear(input_dim, hidden_dim, rngs=rngs), nnx.relu]
        for _ in range(num_layers - 1):
            layers.extend([nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), nnx.relu])
        layers.append(nnx.Linear(hidden_dim, 1, rngs=rngs))
        self.critic = nnx.Sequential(*layers)

    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return self.critic(jnp.concat([obs, action], axis=-1)).squeeze(-1)


# States
## RolloutState contains the necessary information to step the environment
## it includes - obs: the current observation, used to take the action
##             - env_state: the current state of the environment
##             - key: to sample actions
class RolloutState(NamedTuple):
    obs: jax.Array
    env_state: Any
    step: int
    key: jax.Array


## TimeStep is the transition saved to the replay buffer
class TimeStep(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array


## EpisodeStats keeps track of the episodic returns and length
## When an episode is done, we record it is episode length and episodic return
class EpisodeStats(NamedTuple):
    episode_return: jax.Array
    episode_length: jax.Array


# polyak_update
def polyak_update(network, target_network, tau):
    params = nnx.state(network, nnx.Param)
    target_params = nnx.state(target_network, nnx.Param)

    updated_params = jax.tree.map(
        lambda target, online: (1.0 - tau) * target + tau * online,
        target_params,
        params,
    )
    nnx.update(target_network, updated_params)


if __name__ == "__main__":
    args = tyro.cli(Args)
    # Vec training params
    num_updates = args.total_timesteps // args.num_envs
    assert args.train_freq_critic % args.num_envs == 0, "args.train_freq_critic % args.num_envs != 0"
    assert args.train_freq_actor % args.num_envs == 0, "args.train_freq_actor % args.num_envs != 0"
    assert args.train_freq_actor >= args.train_freq_critic, "args.train_freq_actor < args.train_freq_critic"
    assert args.train_freq_actor % args.train_freq_critic == 0, (
        "args.train_freq_actor % args.train_freq_critic != 0"
    )
    # Rng keys
    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)
    key, reset_key, eval_key = jax.random.split(key, 3)
    # Import the environment
    env = make_env(args)
    action_low = env.action_space.low
    action_high = env.action_space.high
    # Prepare networks
    actor = Actor(
        input_dim=env.observation_size,
        hidden_dim=args.actor_hidden_dim,
        num_layers=args.actor_num_layers,
        output_dim=env.action_size,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        rngs=rngs,
    )
    target_actor = nnx.clone(actor)
    qnet1 = Critic(
        input_dim=env.observation_size + env.action_size,
        hidden_dim=args.critic_hidden_dim,
        num_layers=args.critic_num_layers,
        rngs=nnx.Rngs(args.seed),
    )
    qnet2 = Critic(
        input_dim=env.observation_size + env.action_size,
        hidden_dim=args.critic_hidden_dim,
        num_layers=args.critic_num_layers,
        rngs=nnx.Rngs(args.seed + 1),
    )
    critic = nnx.Dict({"qnet1": qnet1, "qnet2": qnet2})
    target_qnet1 = nnx.clone(qnet1)
    target_qnet2 = nnx.clone(qnet2)
    target_critic = nnx.Dict({"qnet1": target_qnet1, "qnet2": target_qnet2})
    # Optimizers
    optim = getattr(optax, args.optimizer)
    actor_optimizer = optim(learning_rate=args.learning_rate_actor)
    critic_optimizer = optim(learning_rate=args.learning_rate_critic)
    if args.clip_gradients > 0:
        actor_optimizer = optax.chain(optax.clip_by_global_norm(args.clip_gradients), actor_optimizer)
        critic_optimizer = optax.chain(optax.clip_by_global_norm(args.clip_gradients), critic_optimizer)

    actor_optimizer = nnx.Optimizer(actor, actor_optimizer, wrt=nnx.Param)
    critic_optimizer = nnx.Optimizer(critic, critic_optimizer, wrt=nnx.Param)
    # Logging
    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_name}__{args.exp_name}__{time_token}"
    log_dir = f"{args.work_dir}/TD3-{run_name}"
    # Reset the env
    reset_key = jax.random.split(reset_key, args.num_envs)
    obs, env_state = env.reset(key=reset_key)
    # Replay buffer
    rb_fun = fbx.make_flat_buffer(
        max_length=args.buffer_size,
        min_length=args.batch_size,
        sample_batch_size=args.batch_size,
        add_batch_size=args.num_envs,
    )
    buffer = rb_fun.init(  # initialize the buffer
        TimeStep(
            obs=jnp.zeros(env.observation_size),
            action=jnp.zeros(env.action_size),
            reward=jnp.array(1.0),
            done=jnp.array(True),
        )
    )
    # Prepare the rollout state
    rollout_state = RolloutState(obs=obs, env_state=env_state, step=0, key=key)
    # polyak func
    polyak_update = partial(polyak_update, tau=args.polyak)

    # update_step: step the envs + periodically update the networks
    @nnx.jit
    @nnx.scan(
        length=num_updates,
        in_axes=(nnx.Carry, None),
        out_axes=(nnx.Carry, 0),
    )
    def update_step(update_state: tuple, _):  # noqa: PLR0915

        (
            actor,
            target_actor,
            actor_optimizer,
            critic,
            target_critic,
            critic_optimizer,
            rollout_state,
            buffer,
        ) = update_state

        # ------ Collect env steps ------
        obs, state, step, key = rollout_state
        # Keys for exploration and env.step
        key, key_noise, key_step = jax.random.split(key, 3)
        key_step = jax.random.split(key_step, args.num_envs)
        # Get action
        action = actor(obs=obs)
        noise = jax.random.normal(key_noise, action.shape) * actor.action_scale * args.act_noise
        action = jnp.clip(action + noise, action_low, action_high)
        # step the env
        next_obs, next_state, reward, terminated, truncated, info = env.step(key_step, state, action)
        step += args.num_envs
        done = jnp.logical_or(terminated, truncated)
        # Record episodic return and length
        episode_stats = EpisodeStats(
            episode_return=info["episode_return"], episode_length=info["episode_length"]
        )
        # Add the step in to the replay buffer
        timestep = TimeStep(obs=obs, action=action, reward=reward, done=done)
        buffer = rb_fun.add(buffer, timestep)

        # ------ Update critics and the  ------
        def update_actor_and_critic(
            actor,
            actor_optimizer,
            critic,
            target_critic,
            critic_optimizer,
            update_actor_event,
            key_sample,
            key_noise,
        ):
            # Sample a batch
            batch = rb_fun.sample(buffer, key_sample).experience
            # ---- Update critics
            # compute targets
            next_action = target_actor(obs=batch.second.obs)
            next_noise = jax.random.normal(key_noise, next_action.shape) * args.target_noise
            next_noise = jnp.clip(next_noise, -args.noise_clip, args.noise_clip) * target_actor.action_scale
            next_action = jnp.clip(next_action + next_noise, action_low, action_high)
            q1_vals_next = target_critic["qnet1"](obs=batch.second.obs, action=next_action)
            q2_vals_next = target_critic["qnet2"](obs=batch.second.obs, action=next_action)
            q_vals_next = jnp.minimum(q1_vals_next, q2_vals_next)
            targets = batch.first.reward + args.gamma * (1 - batch.first.done) * q_vals_next

            # critic_loss
            def critic_loss(critic):
                q1_values = critic["qnet1"](obs=batch.first.obs, action=batch.first.action)
                q1_loss = optax.l2_loss(targets, q1_values).mean()
                q2_values = critic["qnet2"](obs=batch.first.obs, action=batch.first.action)
                q2_loss = optax.l2_loss(targets, q2_values).mean()
                return q1_loss + q2_loss

            # Update the critics
            cr_loss, grads = nnx.value_and_grad(critic_loss)(critic)
            critic_optimizer.update(critic, grads)

            # ---- Update the actor
            def update_actor(actor, actor_optimizer, obs):
                # actor_loss
                def actor_loss(actor):
                    action = actor(obs=obs)
                    q_values = critic["qnet1"](obs=obs, action=action)
                    ac_loss = -q_values.mean()
                    return ac_loss

                ac_loss, grads = nnx.value_and_grad(actor_loss)(actor)
                actor_optimizer.update(actor, grads)
                return ac_loss

            # Update the actor if update_actor_event==True
            ac_loss = nnx.cond(
                update_actor_event,
                update_actor,
                lambda *_: jnp.array(0.0),
                actor,
                actor_optimizer,
                batch.first.obs,
            )
            return cr_loss, ac_loss

        # Decide if it is time to update
        update_critic_event = jnp.logical_and(
            step > jnp.maximum(args.batch_size, args.learning_starts), step % args.train_freq_critic == 0
        )
        update_actor_event = jnp.logical_and(
            step > jnp.maximum(args.batch_size, args.learning_starts), step % args.train_freq_actor == 0
        )
        key, key_sample, key_noise = jax.random.split(key, 3)
        # It is important to use nnx.cond not jax.lax.cond. Otherwise the network will not be updated
        # Rule of thumb: if the transformation modified the params of the network\optimize,
        # always use nnx. transformations
        cr_loss, ac_loss = nnx.cond(
            update_critic_event,
            update_actor_and_critic,
            lambda *_: (jnp.array(0.0), jnp.array(0.0)),
            actor,
            actor_optimizer,
            critic,
            target_critic,
            critic_optimizer,
            update_actor_event,
            key_sample,
            key_noise,
        )

        # ------ Update the target networks ------
        def update_targets(actor, target_actor, critic, target_critic):
            polyak_update(actor, target_actor)
            polyak_update(critic, target_critic)

        nnx.cond(
            update_actor_event,
            update_targets,
            lambda *_: None,
            actor,
            target_actor,
            critic,
            target_critic,
        )
        # update rollout_state
        rollout_state = RolloutState(obs=next_obs, env_state=next_state, step=step, key=key)
        update_state = (
            actor,
            target_actor,
            actor_optimizer,
            critic,
            target_critic,
            critic_optimizer,
            rollout_state,
            buffer,
        )
        # Metrics: we save "update_event" to log the losses, "done" to log the episode stats
        metrics = (
            cr_loss,
            ac_loss,
            update_actor_event,
            update_critic_event,
            done,
            episode_stats.episode_return,
            episode_stats.episode_length,
        )
        return update_state, metrics

    # Train
    update_state = (
        actor,
        target_actor,
        actor_optimizer,
        critic,
        target_critic,
        critic_optimizer,
        rollout_state,
        buffer,
    )
    update_state, metrics = update_step(update_state, None)
    jax.block_until_ready(metrics)
    # Evaluate
    if args.eval:
        # Create eval env
        actor, _, _, _, _, _, rollout_state, _ = update_state
        eval_env = make_env(args, eval=True, rollout_state=rollout_state)
        # Reset eval_env
        eval_key, reset_key = jax.random.split(eval_key)
        reset_key = jax.random.split(reset_key, args.num_eval_ep)
        obs, env_state = eval_env.reset(reset_key)
        # Initialize metrics
        eval_ep_returns = jnp.zeros(args.num_eval_ep)
        eval_ep_lengths = jnp.zeros(args.num_eval_ep, dtype=jnp.int32)
        ep_dones = jnp.zeros(args.num_eval_ep, dtype=jnp.bool_)
        # Eval state
        evaluation_state = (obs, env_state, eval_ep_returns, eval_ep_lengths, ep_dones, eval_key)

        # Stops once all envs are done
        def cond_fun(evaluation_state):
            return ~jnp.all(evaluation_state[-2])

        # Step the eval envs
        def eval_fun(evaluation_state):
            obs, env_state, eval_ep_returns, eval_ep_lengths, ep_dones, eval_key = evaluation_state
            eval_key, key_step = jax.random.split(eval_key)
            key_step = jax.random.split(key_step, args.num_eval_ep)
            action = actor(obs=obs)
            next_obs, next_state, reward, terminated, truncated, _ = eval_env.step(
                key_step, env_state, action
            )
            active = ~ep_dones.astype(bool)
            eval_ep_returns = eval_ep_returns + jnp.where(active, reward, 0.0)
            eval_ep_lengths = eval_ep_lengths + active.astype(jnp.int32)
            ep_dones = ep_dones.astype(bool) | terminated.astype(bool) | truncated.astype(bool)
            return next_obs, next_state, eval_ep_returns, eval_ep_lengths, ep_dones, eval_key

        # Evaluation loop
        _, _, eval_ep_returns, eval_ep_lengths, ep_dones, _ = jax.lax.while_loop(
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
    # tensorboard logging
    if args.log:
        metrics = jax.device_get(metrics)
        (
            cr_losses,
            ac_losses,
            update_actor_event,
            update_critic_event,
            dones,
            episode_returns,
            episode_lengths,
        ) = metrics
        cr_losses = cr_losses.reshape(-1)
        ac_losses = ac_losses.reshape(-1)
        update_actor_event = update_actor_event.reshape(-1)
        update_critic_event = update_critic_event.reshape(-1)
        dones = dones.reshape(-1)
        episode_returns = episode_returns.reshape(-1)
        episode_lengths = episode_lengths.reshape(-1)
        writer = SummaryWriter(log_dir)
        completed_steps = np.flatnonzero(dones)
        for start in range(0, len(completed_steps), args.log_every):
            episode_steps = completed_steps[start : start + args.log_every]
            step = int(episode_steps[-1]) + 1
            mean_episode_return = episode_returns[episode_steps].mean()
            mean_episode_length = episode_lengths[episode_steps].mean()
            writer.add_scalar("rollout/ep_reward", float(mean_episode_return), step)
            writer.add_scalar("rollout/ep_length", float(mean_episode_length), step)
        updates_steps = np.flatnonzero(update_critic_event)
        for start in range(0, len(updates_steps), args.log_every):
            train_steps = updates_steps[start : start + args.log_every]
            step = (int(train_steps[-1]) + 1) * args.num_envs
            cr_loss = cr_losses[train_steps].mean()
            writer.add_scalar("losses/cr_loss", float(cr_loss), step)
        updates_steps = np.flatnonzero(update_actor_event)
        for start in range(0, len(updates_steps), args.log_every):
            train_steps = updates_steps[start : start + args.log_every]
            step = (int(train_steps[-1]) + 1) * args.num_envs
            ac_loss = ac_losses[train_steps].mean()
            writer.add_scalar("losses/ac_loss", float(ac_loss), step)
        writer.close()
    # Save the weights
    if args.save_model:
        actor, *_ = update_state
        _, actor_state = nnx.split(actor)
        network_states = {"actor": actor_state}
        checkpoint_path = (Path(log_dir) / "networks").resolve()
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(checkpoint_path, network_states)
        print(f"Networks saved to {checkpoint_path}")
        with open(Path(log_dir) / "args.json", "w") as file:
            json.dump(vars(args), file, indent=2)
