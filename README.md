# FlaxRL

FlaxRL provides fast implementation of Deep RL algorithms using JAX-based environments  and [Flax NNX](https://flax.readthedocs.io/en/latest/why.html) neural network API. We offer the following:

- Single-agent RL algorithms: DQN, PPO, SAC, TD3.
- Multi-agent RL algorithms: VDN, QMIX, COMA, IPPO, MAPPO.
- Unified environment APIs: Gymnax, Brax, MuJoCo Playground, SMAX, MPE and (some) Jumanji.
- Wrappers: vectorization, normalization, action clipping, stats recording
- MLP, RNN and CNN policies


Why FlaxRL:

* Slow prototyping and hyper-paramter tuning is one of the main bottlenecks of any RL related research work. JAX allow us to have fast environments and implementations. FlaxRL provides the implementations of some core RL and MARL algorithms.

* Similar projects ([PureJaxRL](https://github.com/luchris429/purejaxrl), [JaxMARL](https://github.com/bold-lab-ai/JaxMARL), [MAVA](https://github.com/instadeepai/Mava) [Brax](https://github.com/google/brax/tree/main/brax/training)) are all built using the old Flax API (i.e. Linen API ) and didn't migrate yet to the new API. The new API allows writing cleaner code, and provide handy JAX transformations for neural networks (nnx.value_and_grad, nnx.scan).

* We start to have many envs but their API differ significantly, u may have to write separate RL implementation for each env. FlaxRL instead unifies the APIs of many environments (see above) and provide a single implementation for each algorithm. 


## Installation

Install dependencies using the requirements.txt file.
**IMPORTANT**: The requirements.txt file does not inlcude gymnax and jaxmarl to avoid dependency conflects. We use the following versions: `gymnax==1.0.0` and `jaxmarl==0.1.0`. After installing them, please update `jax`, `flax`, `orbax`, and  `jaxlibe`. 
**IMPORTANT**: FlaxRL was tested with `jax[cuda12]`

## Implementations
#### RL
|Algorithm | Implementations| 
| ---- | ---- |
| PPO | [ppo.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/ppo.py) : discrete actions <br> [ppo_continuous](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/ppo_continuous.py): continuous actions <br> [ppo_rnn](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/ppo_rnn.py): recurrent policies <br> [ppo_cnn](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/ppo_cnn.py): CNN policies (e.g., MinAtar)
|DQN | [dqn.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/dqn.py): discrete actions <br> [dqn_dnn](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/dqn_cnn.py): CNN Q-networks
|SAC | [sac.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/sac.py): continuous actions
|TD3| [TD3](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/td3.py): continuous actions

#### MARL
Inspired from [CleanMARL](https://github.com/AmineAndam04/cleanmarl/tree/main)

|Algorithm | Implementations| 
| ---- | ---- |
VDN | [vdn.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/vdn.py): discrete actions + common rewards
QMIX | [qmix.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/qmix.py): discrete actions + common rewards
COMA | [coma.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/qmix.py): discrete actions 
IPPO | [ippo.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/ippo.py): discrete actions  <br> [ippo_rnn.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/ippo_rnn.py): recurrent policies
MAPPO | [mappo.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/mappo.py): discrete actions <br> [mppo_rnn.py](https://github.com/AmineAndam04/flaxrl/blob/main/flaxrl/mappo_rnn.py): recurrent policies


## Examples

You can train from CLI using: 
```
python flaxrl/ppo_continuous.py  \
        --env_name=hopper --env_type=brax \ 
        --total_timesteps=10000000 --num_envs=256 --num_steps=128 \
         --n_epochs=4096 --save_model
```