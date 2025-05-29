from yahoo_fin import stock_info
import yfinance as yf
import pandas as pd
from trading_env import TradingEnv # Importar el entorno de trading
from dqn_agent import DQNAgent # Importar el agente DQN
import numpy as np

def fetch_data(ticker):
    """Fetches financial data and historical prices for a given ticker."""
    stock = yf.Ticker(ticker)
    income_stmt = stock.get_income_stmt(freq='quarterly')
    balance_sheet = stock.get_balance_sheet(freq='quarterly')
    cash_flow = stock.get_cash_flow(freq='quarterly')
    try:
        history = stock.history(period="max")
    except Exception as e:
        history = pd.DataFrame()
    return income_stmt, balance_sheet, cash_flow, history

def process_financial_data(income_stmt, balance_sheet, cash_flow):
    """Transposes and cleans financial data DataFrames."""
    income_stmt_T = income_stmt.T
    balance_sheet_T = balance_sheet.T
    cash_flow_T = cash_flow.T

    # Ensure index is datetime and sorted
    income_stmt_T.index = pd.to_datetime(income_stmt_T.index)
    income_stmt_T = income_stmt_T.sort_index()

    balance_sheet_T.index = pd.to_datetime(balance_sheet_T.index)
    balance_sheet_T = balance_sheet_T.sort_index()

    cash_flow_T.index = pd.to_datetime(cash_flow_T.index)
    cash_flow_T = cash_flow_T.sort_index()

    return income_stmt_T, balance_sheet_T, cash_flow_T

def merge_data(history, income_stmt_T, balance_sheet_T, cash_flow_T):
    """Merges historical prices with transposed financial data."""
    # Ensure history index is datetime, sorted, and timezone-naive
    history.index = pd.to_datetime(history.index)
    if history.index.tz is not None:
        history.index = history.index.tz_convert(None)
    history = history.sort_index()

    # Ensure financial data DataFrames are not empty and have datetime index
    # Return empty DataFrame if any financial data is missing
    if income_stmt_T.empty or balance_sheet_T.empty or cash_flow_T.empty:
        print("Datos financieros incompletos o vacíos. Devolviendo DataFrame vacío.")
        return pd.DataFrame()

    income_stmt_T.index = pd.to_datetime(income_stmt_T.index)
    income_stmt_T = income_stmt_T.sort_index()

    balance_sheet_T.index = pd.to_datetime(balance_sheet_T.index)
    balance_sheet_T = balance_sheet_T.sort_index()

    cash_flow_T.index = pd.to_datetime(cash_flow_T.index)
    cash_flow_T = cash_flow_T.sort_index()

    # Merge transposed financial data with history using merge_asof
    merged_history = pd.merge_asof(history, income_stmt_T, left_index=True, right_index=True, direction='backward')
    merged_history = pd.merge_asof(merged_history, balance_sheet_T, left_index=True, right_index=True, direction='backward')
    merged_history = pd.merge_asof(merged_history, cash_flow_T, left_index=True, right_index=True, direction='backward')

    return merged_history

if __name__ == "__main__":
    tickers_list = stock_info.tickers_sp500() + stock_info.tickers_dow() + stock_info.tickers_nasdaq() + stock_info.tickers_other() + stock_info.tickers_other()
    tickers_list = list(set(tickers_list))
    tickers_list = [ticker for ticker in tickers_list if ticker and not ticker.startswith('.')]

    # Configuración del entrenamiento
    num_episodes = 100 # Número de episodios de entrenamiento

    # Vamos a entrenar el agente en un solo ticker por ahora para simplificar
    # Puedes modificar esto para iterar sobre tickers_list si deseas entrenar en múltiples activos
    # For demonstration, let's pick the first valid ticker
    for ticker in tickers_list:
        income_stmt, balance_sheet, cash_flow, history = fetch_data(ticker)
        income_stmt_T, balance_sheet_T, cash_flow_T = process_financial_data(income_stmt, balance_sheet, cash_flow)
        merged_history = merge_data(history, income_stmt_T, balance_sheet_T, cash_flow_T)

        if not merged_history.empty:
            print(f"Entrenando agente DQN en {ticker}...")
            env = TradingEnv(merged_history)

            # Inicializar el agente DQN
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            agent = DQNAgent(state_size, action_size)

            # Bucle de entrenamiento
            for episode in range(1, num_episodes + 1):
                state, info = env.reset()
                state = np.array(state)
                total_reward = 0
                done = False

                while not done:
                    action = agent.act(state)
                    next_state, reward, done, truncated, info = env.step(action)
                    next_state = np.array(next_state)

                    # Guardar experiencia y aprender
                    agent.step(state, action, reward, next_state, done)

                    state = next_state
                    total_reward += reward

                # Decaer epsilon al final del episodio
                agent.decay_epsilon()

                print(f"Episodio {episode}/{num_episodes}, Recompensa Total: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Valor Neto Final: {env.net_worth:.2f}")

            print(f"\nEntrenamiento completado para {target_ticker}.")
        else:
            print(f"No se pudieron obtener datos válidos para {ticker}.")
