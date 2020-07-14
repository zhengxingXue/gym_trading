# gym_trading
Trading environment for Reinforcement learning

<img src="img/render_example.png" alt="example output" width="700" />

## Quick start 

To install the Python module:
```commandline
git clone https://github.com/zhengxingXue/gym_trading.git
cd gym_trading
pip install -e .
```

``notebooks/`` directory contains example for training, to start:
 ```commandline
jupyter notebook notebooks/1_getting_started.ipynb 
```

## Env

To create the ``gym_trading`` environment:
```python
import gym
import gym_trading
env = gym.make('StockTrading-v1')  # One IBM stock setting
env.render()
```
