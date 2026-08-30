"""A func to prepare the environment"""

from .mwrappers import RecordVecMARLEpisodeStatistics, RewardAggregatorWrapper, VecMARLWrapper
from .wrappers import (
    AutoResetWrapper,
    ClipAction,
    NormalizeVecObservation,
    NormalizeVecObservationEval,
    NormalizeVecObservationState,
    NormalizeVecReward,
    RecordVecEpisodeStatistics,
    TimeLimit,
    VecWrapper,
)


def make_env(args, eval=False, rollout_state=None):
    """
    A func to prepare the environment
    params: args: the args from training
            eval: is the env for evaluation (no normalization)
            rollout_state: re-use the normalization statistics

    """

    if args.env_type == "gymnax":
        # * Doesn't need a TimeLimit wrapper (see is_truncated in gymnax environment.py)
        # * Doens't need an Autoreset wrapper either
        from .gymnax import GymnaxInterface

        env = GymnaxInterface.make(args.env_name)
        env = VecWrapper(env)
        env = RecordVecEpisodeStatistics(env)

    elif args.env_type == "brax":
        from .brax import BraxInterface

        env = BraxInterface.make(env_name=args.env_name, backend=args.backend)
        if args.clip_actions:
            env = ClipAction(env)
        env = VecWrapper(env)
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
        env = RecordVecEpisodeStatistics(env)
        env = AutoResetWrapper(env)
    elif args.env_type == "playground":
        from .mujoco_playground import PlaygroundInterface

        env = PlaygroundInterface.make(env_name=args.env_name, impl=args.impl)
        if args.clip_actions:
            env = ClipAction(env)
        env = VecWrapper(env)
        env = TimeLimit(env, max_episode_steps=args.max_episode_steps)
        env = RecordVecEpisodeStatistics(env)
        env = AutoResetWrapper(env)
    else:
        raise ValueError(f" Env not yet supported: {args.env_type}")
    if args.normalize_obs:
        if not eval:
            env = NormalizeVecObservation(env)
        elif rollout_state is not None:
            normalization_state = get_state(rollout_state, NormalizeVecObservationState)
            env = NormalizeVecObservationEval(
                env, obs_mean=normalization_state.mean, obs_var=normalization_state.var
            )
        else:
            raise ValueError("Provide the rollout_state")
    if args.normalize_reward and not eval:
        env = NormalizeVecReward(env, gamma=args.gamma)
    return env


def make_marl_env(args):
    """
    A func to prepare marl environments
    params: args: the args from training

    """
    if args.env_type == "mpe":
        from .mpe import MPEInterface

        env = MPEInterface.make(args.env_name)
    elif args.env_type == "smax":
        from .smax import SMAXInterface

        env = SMAXInterface.make(args.env_name)
    env = VecMARLWrapper(env)
    env = RewardAggregatorWrapper(env=env, reward_aggr=args.reward_aggr)
    env = RecordVecMARLEpisodeStatistics(env)
    return env


def get_state(state, state_type):
    while True:
        if isinstance(state, state_type):
            return state

        if not hasattr(state, "env_state"):
            raise ValueError(f"{state_type.__name__} was not found")

        state = state.env_state
