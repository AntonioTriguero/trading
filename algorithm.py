import numpy as np
import dataset
import gymnasium as gym
from stable_baselines3 import DQN
from typing import Optional

# 1. Definir el entorno personalizado (ya creado por ti)
class TradingEnv(gym.Env):
    def __init__(self, tickers, transaction_penalty=1.0):
        self.tickers = tickers
        self.current_ticker = np.random.choice(self.tickers)
        self.data = dataset.get_data(self.current_ticker)
        self.size = data.shape[1]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],))
        self.action_space = gym.spaces.Discrete(3)
        self.n_steps = len(data)
        self.current_step = 0
        self.position = 0  # 0=fuera del mercado, 1=comprado
        self.buy_price = 0
        self.transaction_penalty = transaction_penalty
        self.num_transactions = 0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.buy_price = 0
        self.num_transactions = 0
        return self.data.loc[self.current_step].values.astype(np.float32), {}

    def step(self, action):
        done = False
        reward = 0
        absorbing = False  # Indica si el estado es terminal (no necesario en este caso)
        info = {}

        price = self.data.loc[self.current_step, 'Close']

        # Lógica de acciones
        if action == 1 and self.position == 0:  # Comprar
            self.position = 1
            self.buy_price = price
            self.num_transactions += 1
            reward -= self.transaction_penalty
            
        elif action == 2 and self.position == 1:  # Vender
            reward = price - self.buy_price
            self.position = 0
            self.buy_price = 0
            self.num_transactions += 1
            reward -= self.transaction_penalty

        # Avanzar en el tiempo
        self.current_step += 1
        
        # Verificar fin del episodio
        if self.current_step >= self.n_steps - 1:
            done = True
            absorbing = True
            # Cerrar posición abierta al final
            if self.position == 1:
                final_price = self.data.loc[self.current_step, 'Close']
                reward += final_price - self.buy_price
                self.position = 0

        # Obtener siguiente observación
        next_state = self.data.loc[self.current_step].values
        
        return next_state, reward, done, absorbing, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step} | Position: {self.position} | Buy Price: {self.buy_price}")

# 2. Crear y configurar el entorno
data = featuring_engineering.get_data('AAPL')
env = TradingEnv(data)

# 3. Instanciar algoritmo DQN y entrenar
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
