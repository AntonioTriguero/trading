import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    """Custom Environment for Trading that follows gym interface."""
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.sort_index() # Asegurarse de que el DataFrame esté ordenado por fecha
        self.current_step = 0
        self.initial_balance = 10000 # Saldo inicial en USD
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.history = [] # Para registrar el historial de trading

        # Definir el espacio de acciones: Comprar, Vender, Mantener
        # 0: Mantener, 1: Comprar, 2: Vender
        self.action_space = spaces.Discrete(3)

        # Definir el espacio de observación (estado)
        # Incluirá el balance, acciones en posesión, precio actual y características del DataFrame
        # El tamaño del espacio de observación dependerá de las columnas de df
        # Por ahora, un ejemplo básico. Necesitaremos ajustar esto.
        num_features = self.df.shape[1] # Número de columnas en el DataFrame de entrada
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features + 3,), dtype=np.float32)

    def _get_obs(self):
        """Obtiene el estado actual del entorno."""
        # Obtener los datos del paso actual
        current_data = self.df.iloc[self.current_step].values
        # Concatenar con el balance, acciones en posesión y valor neto
        obs = np.concatenate(([self.balance, self.shares_held, self.net_worth], current_data))
        return obs

    def _get_info(self):
        """Obtiene información adicional del entorno."""
        return {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'net_worth': self.net_worth,
            'max_net_worth': self.max_net_worth,
            'current_price': self.df['Close'].iloc[self.current_step] # Asumiendo que 'Close' es una columna
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.history = []

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        current_price = self.df['Close'].iloc[self.current_step] # Precio de cierre del día

        # Calcular el valor neto antes de la acción
        prev_net_worth = self.net_worth

        if action == 1:  # Comprar
            # Comprar tantas acciones como sea posible con el balance actual
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
        elif action == 2:  # Vender
            # Vender todas las acciones en posesión
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0

        # Actualizar el valor neto
        self.net_worth = self.balance + self.shares_held * current_price

        # Recompensa: cambio en el valor neto
        reward = self.net_worth - prev_net_worth

        # Actualizar el paso actual
        self.current_step += 1

        # Determinar si el episodio ha terminado
        done = self.current_step >= len(self.df) - 1

        # Actualizar el máximo valor neto
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, False, info

    def render(self):
        # Implementar la lógica de renderizado si es necesario (por ejemplo, graficar el progreso)
        pass

    def close(self):
        # Limpiar recursos si es necesario
        pass

# Ejemplo de uso (esto iría en otro script o en el __main__ de analysis.py)
# if __name__ == '__main__':
#     # Suponiendo que 'history_filtered_by_financial_dates' es un DataFrame de pandas
#     # con columnas como 'Open', 'High', 'Low', 'Close', 'Volume' y otras características financieras
#     # y un índice de fecha.
#     # df_example = pd.DataFrame(...)
#     # env = TradingEnv(df_example)
#     # obs, info = env.reset()
#     # for _ in range(100):
#     #     action = env.action_space.sample() # Agente aleatorio
#     #     obs, reward, done, truncated, info = env.step(action)
#     #     if done:
#     #         break