import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    """Red neuronal (Q-Network) para aproximar la función Q."""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """Agente DQN para el trading."""
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = deque(maxlen=10000) # Tamaño del replay buffer
        self.batch_size = 64
        self.gamma = 0.99 # Factor de descuento
        self.epsilon = 1.0 # Para la estrategia epsilon-greedy
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.tau = 1e-3 # Para soft update de la red target

    def step(self, state, action, reward, next_state, done):
        """Guarda la experiencia en la memoria y aprende si hay suficientes muestras."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.batch_size:
            experiences = self.memory_sample()
            self.learn(experiences)

    def memory_sample(self):
        """Muestra un batch aleatorio de experiencias de la memoria."""
        return random.sample(self.memory, k=self.batch_size)

    def act(self, state):
        """Retorna la acción para el estado dado según la política actual (epsilon-greedy)."""
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Actualiza los parámetros de la red Q local usando un batch de experiencias."""
        states, actions, rewards, next_states, dones = (torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float(),
                                                      torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long(),
                                                      torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float(),
                                                      torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float(),
                                                      torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float())

        # Calcular los valores Q esperados (target Q values)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Calcular los valores Q locales (local Q values)
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Calcular la pérdida y optimizar
        loss = self.criterion(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualizar la red target
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update de los parámetros del modelo target."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def decay_epsilon(self):
        """Decae el valor de epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Para ejecutar este archivo de forma independiente para pruebas, si es necesario
# if __name__ == '__main__':
#     state_size = 10 # Ejemplo
#     action_size = 3 # Comprar, Vender, Mantener
#     agent = DQNAgent(state_size, action_size)
#     print("Agente DQN inicializado.")