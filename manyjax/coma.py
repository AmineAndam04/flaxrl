import datetime
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
from envs.make_env import make_marl_env
from flax import nnx
from tensorboardX import SummaryWriter


# TODO support reward normalization
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
    num_envs: int = 8
    """num envs"""
    num_steps: int = 2048
    """ Number of collected steps"""
    batch_size: int = 64
    """Batch size """
    optimizer: str = "adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    gamma: float = 0.99
    """ Discount factor"""
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    td_lambda: float = 0.8
    """ TD(λ) parameter"""
    polyak: float = 0.2
    """ Polyak coefficient when using polyak averaging for target network update"""
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
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        kernel_init = nnx.initializers.orthogonal(np.sqrt(2))
        layers = [nnx.Linear(input_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs), nnx.relu]
        for _ in range(num_layers - 1):
            layers.extend([nnx.Linear(hidden_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs), nnx.relu])
        layers.append(
            nnx.Linear(hidden_dim, output_dim, kernel_init=nnx.initializers.orthogonal(0.01), rngs=rngs)
        )
        self.logits = nnx.Sequential(*layers)

    def __call__(self, obs: jnp.ndarray, avail_actions: jnp.ndarray) -> distrax.Categorical:
        logits = self.logits(obs)
        logits = jnp.where(avail_actions, logits, -1e10)
        pi = distrax.Categorical(logits)
        return pi

    def get_action(self, obs: jnp.ndarray, avail_actions: jnp.ndarray) -> jnp.ndarray:
        logits = self.logits(obs)
        logits = jnp.where(avail_actions, logits, -1e10)
        return jnp.argmax(logits, axis=-1)


class Critic(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        s_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        num_agents: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.output_dim = output_dim
        self.num_agents = num_agents
        # cache other agents indices
        self.other_agent_ids = jnp.array(
            [[j for j in range(num_agents) if i != j] for i in range(num_agents)]
        )
        kernel_init = nnx.initializers.orthogonal(np.sqrt(2))
        self.state_norm = nnx.LayerNorm(s_dim, rngs=rngs)
        layers = [nnx.Linear(input_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs), nnx.relu]
        for _ in range(num_layers - 1):
            layers.extend([nnx.Linear(hidden_dim, hidden_dim, kernel_init=kernel_init, rngs=rngs), nnx.relu])
        layers.append(
            nnx.Linear(hidden_dim, output_dim, kernel_init=nnx.initializers.orthogonal(1), rngs=rngs)
        )
        self.critic = nnx.Sequential(*layers)

    def __call__(self, state: jnp.ndarray, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        state = self.state_norm(state)
        coma_input = self._coma_input(state, obs, action)
        return self.critic(coma_input)

    def _coma_input(self, state: jnp.ndarray, obs: jnp.ndarray, action: jnp.ndarray):
        action = jax.nn.one_hot(action, self.output_dim)
        action = action[:, self.other_agent_ids].reshape(state.shape[0], self.num_agents, -1)
        state = jnp.repeat(state[:, None, :], self.num_agents, axis=1)
        return jnp.concat([state, obs, action], axis=-1)


# -------- States --------
class RolloutState(NamedTuple):
    """The necessary information to step the environment"""

    obs: jax.Array
    state: jax.Array
    env_state: Any
    avail_actions: jax.Array
    key: jax.Array


class EpisodeStats(NamedTuple):
    """Episodic statistics."""

    episode_return: jax.Array
    episode_length: jax.Array
    battle_won: jax.Array
    ep_done: jax.Array


class Transition(NamedTuple):
    """Transitions to update networks"""

    obs: jax.Array
    state: jax.Array
    action: jax.Array
    avail_actions: jax.Array
    reward: jax.Array
    done: jax.Array
    value: jax.Array


# -------- Training loop --------
def train(args):
    # Rng keys
    key = jax.random.key(args.seed)
    rngs = nnx.Rngs(args.seed)
    key, reset_key, shuffle_key, eval_key = jax.random.split(key, 4)
    # Import the environment
    env = make_marl_env(args)
    # Vec training params
    num_steps = args.num_envs * args.num_steps
    assert num_steps % args.batch_size == 0, "(args.num_envs * args.num_steps) % args.batch_size != 0"
    num_batches = num_steps // args.batch_size
    num_updates = args.total_timesteps // num_steps
    # Prepare networks + optimizers
    actor = Actor(
        input_dim=env.observation_size,
        hidden_dim=args.actor_hidden_dim,
        num_layers=args.actor_num_layers,
        output_dim=env.action_size,
        rngs=rngs,
    )
    critic = Critic(
        input_dim=env.state_size + env.observation_size + env.action_size * (env.num_agents - 1),
        s_dim=env.state_size,
        hidden_dim=args.critic_hidden_dim,
        num_layers=args.critic_num_layers,
        output_dim=env.action_size,
        num_agents=env.num_agents,
        rngs=rngs,
    )
    target_critic = nnx.clone(critic)
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
    log_dir = f"{args.work_dir}/COMA-{run_name}"
    # Reset the env
    reset_key = jax.random.split(reset_key, args.num_envs)
    obs, state, env_state = env.reset(key=reset_key)
    avail_actions = env.get_avail_actions(env_state)
    rollout_state = RolloutState(
        obs=obs, state=state, env_state=env_state, avail_actions=avail_actions, key=key
    )
    polyak_update = partial(polyak_update_, tau=args.polyak)

    # update_step: 1 COMA update = collect num_steps*num_env steps + TD-lambda + Update networks
    @nnx.jit
    @nnx.scan(
        length=num_updates,
        in_axes=(nnx.Carry, None),
        out_axes=(nnx.Carry, 0),
    )
    def update_step(update_state: tuple, _):  # noqa: PLR0915

        actor, actor_optimizer, critic, target_critic, critic_optimizer, rollout_state, shuffle_key = (
            update_state
        )

        # ------ Collect env steps ------
        def collect_rollout(actor: nnx.Module, target_critic: nnx.Module, rollout_state: RolloutState):
            # env_one_step : one step for each environment
            def env_one_step(carry, _):
                # Last rollout state
                obs, state, env_state, avail_actions, key = carry
                # Keys for action-sampling and env.step
                key, key_act, key_step = jax.random.split(key, 3)
                key_step = jax.random.split(key_step, args.num_envs)
                # Get action
                pi = actor(obs=obs, avail_actions=avail_actions)
                action = pi.sample(seed=key_act)
                # Get action value
                values = target_critic(state=state, obs=obs, action=action)
                value = jnp.take_along_axis(
                    arr=values, indices=jnp.expand_dims(action, axis=-1), axis=-1
                ).squeeze(axis=-1)
                # Step the env
                next_obs, next_state, next_env_state, reward, terminated, truncated, info = env.step(
                    key_step, env_state, action
                )
                next_avail_actions = env.get_avail_actions(next_env_state)
                done = terminated.astype(bool) | info["__all__"][:, None]
                # Record its episodic return and length
                episode_stats = EpisodeStats(
                    episode_return=info["episode_return"],
                    episode_length=info["episode_length"],
                    battle_won=reward >= 1,
                    ep_done=info["__all__"],
                )
                # Store data needed for training
                transition = Transition(
                    obs=obs,
                    state=state,
                    action=action,
                    avail_actions=avail_actions,
                    reward=reward,
                    done=done,
                    value=value,
                )
                # Prepare the next rollout_state
                rollout_state = RolloutState(
                    obs=next_obs,
                    state=next_state,
                    env_state=next_env_state,
                    avail_actions=next_avail_actions,
                    key=key,
                )
                return rollout_state, (transition, episode_stats)

            # Collect args.num_steps env steps
            rollout_state, (transitions, episode_stats) = jax.lax.scan(
                f=env_one_step, init=rollout_state, xs=None, length=args.num_steps
            )

            return rollout_state, transitions, episode_stats

        rollout_state, transitions, episode_stats = collect_rollout(actor, target_critic, rollout_state)
        # ------ Compute TD-lambda advantages and returns ------
        # Compute the value of the last steps
        _, key_act = jax.random.split(rollout_state.key)
        action = actor(rollout_state.obs, rollout_state.avail_actions).sample(seed=key_act)
        next_value = target_critic(state=rollout_state.state, obs=rollout_state.obs, action=action)
        next_value = jnp.take_along_axis(
            arr=next_value, indices=jnp.expand_dims(action, axis=-1), axis=-1
        ).squeeze(axis=-1)

        def compute_td_returns(transition, next_value):
            def td_returns_t(carry, transition):
                td_return, next_value = carry
                reward, done, value = transition.reward, transition.done, transition.value
                td_return = reward + args.gamma * (1 - done) * (
                    args.td_lambda * td_return + (1 - args.td_lambda) * next_value
                )
                return (td_return, value), td_return

            # td returns loop: start the transitions from the end (reverse=True)
            _, td_returns = jax.lax.scan(
                init=(next_value, next_value), f=td_returns_t, xs=transition, reverse=True
            )
            return td_returns

        td_returns = compute_td_returns(transition=transitions, next_value=next_value)

        # ------ Update the actor and critic ------
        # Prepare training batches
        # (args.num_steps,num_envs, ***) to (args.num_steps* num_envs, ***)
        batches = (transitions, td_returns)
        batches = jax.tree.map(lambda x: jax.lax.collapse(x, 0, 2), batches)
        shuffle_key, permutation_key = jax.random.split(shuffle_key)
        permutation = jax.random.permutation(permutation_key, num_steps)
        batches = jax.tree.map(lambda x: x[permutation], batches)
        batches = jax.tree.map(lambda x: x.reshape((num_batches, args.batch_size) + x.shape[1:]), batches)

        # coma_batch: run one (mini)-batch update
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def coma_batch(carry, batch):
            batch_transition, batch_returns = batch
            actor, actor_optimizer, critic, critic_optimizer = carry

            # Update the critic
            def ppo_critic_loss(critic, b_state, b_obs, b_action, b_td_returns):
                # TODO try using the new values
                all_values = critic(b_state, b_obs, b_action)
                values = jnp.take_along_axis(
                    arr=all_values, indices=jnp.expand_dims(b_action, axis=-1), axis=-1
                ).squeeze(axis=-1)
                cr_loss = optax.l2_loss(values, b_td_returns).mean()
                return cr_loss, (all_values, values)

            (cr_loss, all_values), cr_grads = nnx.value_and_grad(ppo_critic_loss, has_aux=True)(
                critic,
                batch_transition.state,
                batch_transition.obs,
                batch_transition.action,
                batch_returns,
            )
            critic_optimizer.update(critic, cr_grads)

            # Update the actor
            def ppo_actor_loss(actor, b_obs, b_action, b_avail_actions, b_values):
                all_values, values = b_values
                pi = actor(obs=b_obs, avail_actions=b_avail_actions)
                probs = pi.probs
                log_pros = pi.log_prob(b_action)
                entropy = pi.entropy().mean()
                advantages = values - jnp.sum(probs * all_values, axis=-1)
                advantages = jax.lax.stop_gradient(advantages)
                ac_loss = -(log_pros * advantages).mean() - args.entropy_coef * entropy
                return ac_loss, entropy

            (ac_loss, entropy), ac_grads = nnx.value_and_grad(ppo_actor_loss, has_aux=True)(
                actor,
                batch_transition.obs,
                batch_transition.action,
                batch_transition.avail_actions,
                all_values,
            )
            actor_optimizer.update(actor, ac_grads)
            carry = (actor, actor_optimizer, critic, critic_optimizer)
            return carry, (ac_loss, entropy, cr_loss)

        (actor, actor_optimizer, critic, critic_optimizer), losses = coma_batch(
            (actor, actor_optimizer, critic, critic_optimizer),
            batches,
        )
        # ------ Update the target critic ------
        polyak_update(critic, target_critic)
        # ------ Prepare the next updating step ------
        update_state = (
            actor,
            actor_optimizer,
            critic,
            target_critic,
            critic_optimizer,
            rollout_state,
            shuffle_key,
        )
        losses = jax.tree.map(lambda x: x.mean(), losses)
        metrics = (
            *losses,
            episode_stats.episode_return,
            episode_stats.episode_length,
            episode_stats.battle_won,
            episode_stats.ep_done,
        )
        return update_state, metrics

    # ------ Run COMA ------
    update_state = actor, actor_optimizer, critic, target_critic, critic_optimizer, rollout_state, shuffle_key
    start_time = time.perf_counter()
    update_state, metrics = update_step(update_state, None)
    jax.block_until_ready(metrics)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Training time: {elapsed_time / 60:.2f} min ({elapsed_time:.2f} sec)")
    actor, _, critic, *_ = update_state

    # ------ Evaluate + tensorboard logging + checkpoints ------
    if args.eval:
        evaluate(args, actor, eval_key)
    if args.log:
        tb_logger(args, metrics, log_dir, num_updates)
    if args.save_model:
        _, actor_state = nnx.split(actor)
        _, critic_state = nnx.split(critic)
        network_states = {"actor": actor_state, "critic": critic_state}
        checkpoint_path = (Path(log_dir) / "networks").resolve()
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(checkpoint_path, network_states)
        print(f"Networks saved to {checkpoint_path}")
        with open(Path(log_dir) / "args.json", "w") as file:
            json.dump(vars(args), file, indent=2)
    return update_state, metrics


def evaluate(args, actor, eval_key):
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
        action = actor.get_action(obs, avail_actions=avail_actions)
        next_obs, _, next_env_state, reward, _, _, infos = eval_env.step(key_step, env_state, action)
        avail_actions = eval_env.get_avail_actions(next_env_state)
        reward = jnp.sum(reward, axis=-1) if args.reward_aggr == "sum" else jnp.mean(reward, axis=-1)
        active = ~ep_dones.astype(bool)
        eval_ep_returns = eval_ep_returns + jnp.where(active, reward, 0.0)
        eval_ep_lengths = eval_ep_lengths + active.astype(jnp.int32)
        ep_dones = ep_dones.astype(bool) | infos["__all__"].astype(bool)
        eval_ep_battle_win += (reward >= 1) & active & infos["__all__"].astype(bool)
        return (
            next_obs,
            avail_actions,
            next_env_state,
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


def tb_logger(args, metrics, log_dir, num_updates):
    """Log results to tensorboard"""
    metrics = jax.device_get(metrics)
    actor_losses, entropies, critic_losses, episode_returns, episode_lengths, battle_win, ep_dones = metrics
    ep_dones = ep_dones.squeeze().reshape(-1)
    episode_returns = jnp.mean(episode_returns, axis=-1).reshape(-1)
    episode_lengths = episode_lengths[:, :, :, 0].reshape(-1)
    if args.env_type == "smax":
        battle_win = jnp.mean(battle_win, axis=-1).reshape(-1)
    writer = SummaryWriter(log_dir)
    completed_steps = np.flatnonzero(ep_dones)
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
    for update in range(0, num_updates, args.log_every):
        step = (update + 1) * args.num_steps * args.num_envs
        writer.add_scalar("losses/actor_loss", float(actor_losses[update]), step)
        writer.add_scalar("losses/entropy", float(entropies[update]), step)
        writer.add_scalar("losses/critic_loss", float(critic_losses[update]), step)
    writer.close()


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
