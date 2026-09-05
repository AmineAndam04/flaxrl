"""PPO with recurrent policies"""

import datetime
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import distrax
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
    critic_hidden_dim: int = 32
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
    """ Number of hidden layers of critic network"""
    log_std_init: float = -1
    """ Initial log std """
    # Training
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    num_envs: int = 32
    """num envs"""
    num_steps: int = 2048
    """ Number of collected steps"""
    batch_size: int = 8
    """Batch size """
    n_epochs: int = 3
    """ Number of training epochs"""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    normalize_advantage: bool = False
    """ Normalize the advantage if True"""
    optimizer: str = "adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    gamma: float = 0.99
    """ Discount factor"""
    gae_lambda: float = 0.95
    """ GAE discount factor"""
    clip_gradients: float = -1
    """ Disable gradient clipping when <= 0; otherwise clip at this value"""
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


# -------- Actor and critic nets --------
class Actor(nnx.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, log_std_init: float, *, rngs: nnx.Rngs
    ):
        self.embed = nnx.Sequential(nnx.Linear(input_dim, hidden_dim, rngs=rngs), nnx.relu)
        self.lstm = nnx.LSTMCell(in_features=hidden_dim, hidden_features=hidden_dim, rngs=rngs)
        self.mean = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros(output_dim) + log_std_init)

    def __call__(self, carry, obs):
        x = self.embed(obs)
        carry, x = self.lstm(carry, x)
        mean = self.mean(x)
        log_std = jnp.broadcast_to(self.log_std, mean.shape)
        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        return carry, pi

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

        mean = self.mean(x)
        log_std = jnp.broadcast_to(self.log_std, mean.shape)
        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        return pi

    def get_action(self, carry, obs):
        x = self.embed(obs)
        carry, x = self.lstm(carry, x)
        mean = self.mean(x)
        return carry, mean

    def initialize_carry(self, num_envs, actor_hidden_dim):
        return self.lstm.initialize_carry((num_envs, actor_hidden_dim), rngs=nnx.Rngs(0))


class Critic(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, *, rngs: nnx.Rngs):
        layers = [nnx.Linear(input_dim, hidden_dim, rngs=rngs), nnx.relu]
        for _ in range(num_layers - 1):
            layers.extend([nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), nnx.relu])
        layers.append(nnx.Linear(hidden_dim, 1, rngs=rngs))
        self.critic = nnx.Sequential(*layers)

    def __call__(self, obs: jnp.ndarray):
        return self.critic(obs).squeeze(-1)


# -------- States --------
class RolloutState(NamedTuple):
    """The necessary information to step the environment"""

    obs: jax.Array
    env_state: Any
    lstm_carry: Any
    episode_start: jax.Array
    key: jax.Array


class EpisodeStats(NamedTuple):
    """Episodic statistics"""

    episode_return: jax.Array
    episode_length: jax.Array


class Transition(NamedTuple):
    """Transitions to update networks"""

    obs: jax.Array
    action: jax.Array
    log_prob: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array
    episode_start: jax.Array


def train(args):
    # Vec training params
    num_steps = args.num_envs * args.num_steps
    assert args.num_envs % args.batch_size == 0, "args.num_envs % args.batch_size != 0"
    num_batches = args.num_envs // args.batch_size
    num_updates = args.total_timesteps // num_steps
    # Rng keys
    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)
    key, reset_key, eval_key = jax.random.split(key, 3)
    # Import the environment
    env = make_env(args)
    # Prepare networks + optimizes
    actor = Actor(
        input_dim=env.observation_size,
        hidden_dim=args.actor_hidden_dim,
        output_dim=env.action_size,
        log_std_init=args.log_std_init,
        rngs=rngs,
    )
    critic = Critic(
        input_dim=env.observation_size,
        hidden_dim=args.critic_hidden_dim,
        num_layers=args.critic_num_layers,
        rngs=rngs,
    )
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
    log_dir = f"{args.work_dir}/PPO-{run_name}"
    # Reset the env
    reset_key = jax.random.split(reset_key, args.num_envs)
    obs, env_state = env.reset(key=reset_key)
    #! Initialize LSTM Hidden state
    # its shape is the input shape except the sequence dimension. The best example is in the docstring og flax.linen.scan
    lstm_carry = actor.initialize_carry(args.num_envs, args.actor_hidden_dim)
    episode_start = jnp.zeros(args.num_envs).astype(bool)
    rollout_state = RolloutState(
        obs=obs, env_state=env_state, lstm_carry=lstm_carry, episode_start=episode_start, key=key
    )

    # update_step: 1 PPO update = collect args.num_steps*num_env steps + compute GAE + PPO updates for n_epochs
    @nnx.jit
    @nnx.scan(
        length=num_updates,
        in_axes=(nnx.Carry, None),
        out_axes=(nnx.Carry, 0),
    )
    def update_step(update_state: tuple, _):  # noqa: PLR0915

        actor, actor_optimizer, critic, critic_optimizer, rollout_state = update_state
        initial_lstm_carry = rollout_state.lstm_carry

        # ------ Collect env steps ------
        def collect_rollout(
            actor: nnx.Module, critic: nnx.Module, rollout_state: RolloutState
        ) -> tuple[RolloutState, Transition, EpisodeStats]:
            # env_one_step : one step for each environment
            def env_one_step(carry, x):
                # Last rollout state
                obs, state, lstm_carry, episode_start, key = carry
                # Reset hidden states
                lstm_carry = jax.tree.map(
                    lambda x: jnp.where(episode_start[:, None].astype(bool), jnp.zeros_like(x), x), lstm_carry
                )
                # Keys for action-sampling and env.step
                key, key_act, key_step = jax.random.split(key, 3)
                key_step = jax.random.split(key_step, args.num_envs)
                # Get action, log_prob and value
                lstm_carry, pi = actor(carry=lstm_carry, obs=obs)
                value = critic(obs=obs)
                action = pi.sample(seed=key_act)
                log_prob = pi.log_prob(action)
                # Step the env
                next_obs, next_state, reward, terminated, truncated, info = env.step(key_step, state, action)
                done = jnp.logical_or(terminated, truncated)
                # Record its episodic return and length
                episode_stats = EpisodeStats(
                    episode_return=info["episode_return"], episode_length=info["episode_length"]
                )
                # Store date needed for training
                transition = Transition(
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                    value=value,
                    episode_start=episode_start,
                )
                # Prepare the next rollout_state
                rollout_state = RolloutState(
                    obs=next_obs, env_state=next_state, lstm_carry=lstm_carry, episode_start=done, key=key
                )
                return rollout_state, (transition, episode_stats)

            rollout_state, (transitions, episode_stats) = jax.lax.scan(
                f=env_one_step, init=rollout_state, xs=None, length=args.num_steps
            )
            return rollout_state, transitions, episode_stats

        rollout_state, transitions, episode_stats = collect_rollout(actor, critic, rollout_state)

        # ------ Compute GAE advantages and returns ------
        # Compute the value of the last steps
        next_value = critic(obs=rollout_state.obs)

        def compute_advantage_and_return(
            transition: Transition, next_value: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            # gae_t: one step GAE
            def gae_t(carry, transition):
                gae, next_value = carry
                reward, done, value = transition.reward, transition.done, transition.value
                delta = reward + args.gamma * (1 - done) * next_value - value
                gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                return (gae, value), gae

            # advantages loop: start the transitions from the end (reverse=True)
            _, advantages = jax.lax.scan(
                init=(jnp.zeros_like(next_value), next_value), f=gae_t, xs=transition, reverse=True
            )
            return advantages, advantages + transition.value

        advantages, returns = compute_advantage_and_return(transition=transitions, next_value=next_value)
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ------ Update the networks ------
        # ppo_epochs: run PPO updates for args.n_epochs epochs
        # @nnx.jit(donate_argnums=(0,))
        @nnx.scan(length=args.n_epochs, in_axes=(nnx.Carry, None), out_axes=(nnx.Carry, 0))
        def ppo_epoch(carry, batches):
            actor, actor_optimizer, critic, critic_optimizer = carry

            # ppo_batch: run one (mini)-batch update
            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def ppo_batch(carry, batch):
                batch_transition, batch_advantages, batch_returns, lstm_carry = batch
                actor, actor_optimizer, critic, critic_optimizer = carry

                def ppo_actor_loss(actor, b_obs, b_action, b_log_probs, b_adv, lstm_carry, b_episode_start):
                    pi = actor.sequence(carry=lstm_carry, obs=b_obs, episode_start=b_episode_start)
                    b_new_log_prob = pi.log_prob(b_action)
                    ratio = jnp.exp(b_new_log_prob - b_log_probs)
                    pg_loss1 = -b_adv * ratio
                    pg_loss2 = -b_adv * jnp.clip(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip)
                    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
                    entropy = pi.entropy().mean()
                    ac_loss = pg_loss - args.entropy_coef * entropy
                    return ac_loss, entropy

                def ppo_critic_loss(critic, b_obs, b_returns):
                    values = critic(obs=b_obs)
                    cr_loss = optax.l2_loss(values, b_returns).mean()
                    return cr_loss

                # Compute actor loss and gradients
                (ac_loss, entropy), ac_grads = nnx.value_and_grad(ppo_actor_loss, has_aux=True)(
                    actor,
                    batch_transition.obs,
                    batch_transition.action,
                    batch_transition.log_prob,
                    batch_advantages,
                    lstm_carry,
                    batch_transition.episode_start,
                )
                # Update the actor
                actor_optimizer.update(actor, ac_grads)
                # Compute critic loss and gradients
                cr_loss, cr_grads = nnx.value_and_grad(ppo_critic_loss)(
                    critic, batch_transition.obs, batch_returns
                )
                # Update the critic
                critic_optimizer.update(critic, cr_grads)
                carry = (actor, actor_optimizer, critic, critic_optimizer)
                return carry, (ac_loss, entropy, cr_loss)

            (actor, actor_optimizer, critic, critic_optimizer), losses = ppo_batch(
                (actor, actor_optimizer, critic, critic_optimizer),
                batches,
            )
            losses = jax.tree.map(lambda x: x.mean(), losses)
            return (actor, actor_optimizer, critic, critic_optimizer), losses

        # ------ Prepare training batches ------
        # We batch over episodes: consume 'batch_size' env every mini-batch update
        # From (args.num_steps, num_envs,**) to (args.num_steps,num_batches,batch_size, **)
        # We then switch the first two dims
        dones = transitions.done
        transitions, advantages, returns = jax.tree.map(
            lambda x: jnp.swapaxes(
                x.reshape((args.num_steps, num_batches, args.batch_size) + x.shape[2:]), 0, 1
            ),
            (transitions, advantages, returns),
        )
        # We do the same for the initial lstm
        initial_lstm_carry = jax.tree.map(
            lambda x: x.reshape(num_batches, args.batch_size, x.shape[-1]),
            initial_lstm_carry,
        )
        batches = (transitions, advantages, returns, initial_lstm_carry)
        # Train for n_epochs
        (actor, actor_optimizer, critic, critic_optimizer), losses = ppo_epoch(
            (actor, actor_optimizer, critic, critic_optimizer), batches
        )
        update_state = actor, actor_optimizer, critic, critic_optimizer, rollout_state
        losses = jax.tree.map(lambda x: x.mean(), losses)
        metrics = *losses, dones, episode_stats.episode_return, episode_stats.episode_length
        return update_state, metrics

    # ------ Run PPO ------
    update_state = actor, actor_optimizer, critic, critic_optimizer, rollout_state
    start_time = time.perf_counter()
    update_state, metrics = update_step(update_state, None)
    jax.block_until_ready(metrics)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time / 60:.2f} min ({elapsed_time:.2f} sec)")
    actor, _, critic, _, rollout_state = update_state
    # ------ Evaluate + tensorboard logging + checkpoints ------
    if args.eval:
        evaluate(args, actor, rollout_state, eval_key)
    if args.log:
        tb_logger(args, metrics, log_dir, num_updates)
    if args.save_model:
        _, actor_state = nnx.split(actor)
        checkpointer = ocp.StandardCheckpointer()
        checkpoint_path = os.path.abspath(f"{log_dir}/policy")
        checkpointer.save(checkpoint_path, actor_state)
        checkpointer.wait_until_finished()
        print(f"Networks saved to {checkpoint_path}")
        # Save normalization statistics
        if args.normalize_obs:
            from envs.make_env import get_state
            from envs.wrappers import NormalizeVecObservationState

            normalization_state = get_state(rollout_state, NormalizeVecObservationState)
            np.savez(
                Path(log_dir) / "obs_normalization.npz",
                mean=np.asarray(normalization_state.mean),
                var=np.asarray(normalization_state.var),
            )
        with open(Path(log_dir) / "args.json", "w") as file:
            json.dump(vars(args), file, indent=2)


def evaluate(args, actor, rollout_state, eval_key):
    eval_env = make_env(args, eval=True, rollout_state=rollout_state)
    eval_key, reset_key = jax.random.split(eval_key)
    reset_key = jax.random.split(reset_key, args.num_eval_ep)
    obs, env_state = eval_env.reset(reset_key)
    # Initialize hidden state
    eval_lstm_carry = actor.initialize_carry(args.num_eval_ep, args.actor_hidden_dim)
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
        eval_lstm_carry, action = actor.get_action(eval_lstm_carry, obs)
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
    """Log results to tensorboard"""
    metrics = jax.device_get(metrics)
    actor_losses, entropies, critic_losses, dones, episode_returns, episode_lengths = metrics
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
    for update in range(0, num_updates, args.log_every):
        step = (update + 1) * args.num_steps * args.num_envs
        writer.add_scalar("losses/actor_loss", float(actor_losses[update]), step)
        writer.add_scalar("losses/entropy", float(entropies[update]), step)
        writer.add_scalar("losses/critic_loss", float(critic_losses[update]), step)
    writer.close()


if __name__ == "__main__":
    train(tyro.cli(Args))
