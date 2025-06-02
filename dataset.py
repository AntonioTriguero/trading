from yahoo_fin import stock_info
import yfinance as yf
import pandas as pd

def get_financial_data(ticker, type):
    stock = yf.Ticker(ticker)
    if type == 'income':
        get_data = stock.get_income_stmt
    elif type == 'balance':
        get_data = stock.get_balance_sheet
    elif type == 'cash':
        get_data = stock.get_cash_flow
    else:
        raise ValueError("Invalid type. Choose 'income', 'balance', or 'cash'.")

    quarterly_data = get_data(freq='quarterly')
    annual_data = get_data(freq='yearly').iloc[:, 1:]
    combined_data = pd.concat([quarterly_data, annual_data], axis=1)
    combined_data = combined_data.transpose()
    combined_data.index = pd.to_datetime(combined_data.index)
    combined_data = combined_data.sort_index()
    return combined_data

def get_prices_data(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="max")
    history.index = pd.to_datetime(history.index)
    if history.index.tz is not None:
        history.index = history.index.tz_convert(None)
    history = history.sort_index()
    return history
    
def get_history(ticker):
    """Fetches financial data and historical prices for a given ticker."""
    income_stmt = get_financial_data(ticker, 'income')
    balance_sheet = get_financial_data(ticker, 'balance')
    cash_flow = get_financial_data(ticker, 'cash')
    history = get_prices_data(ticker)

    merged_history = pd.merge_asof(history, income_stmt, left_index=True, right_index=True, direction='backward')
    merged_history = pd.merge_asof(merged_history, balance_sheet, left_index=True, right_index=True, direction='backward')
    merged_history = pd.merge_asof(merged_history, cash_flow, left_index=True, right_index=True, direction='backward')

    return merged_history.astype('float32')

def get_tickers():
    tickers_list = stock_info.tickers_sp500() + stock_info.tickers_dow() + stock_info.tickers_nasdaq() + stock_info.tickers_other() + stock_info.tickers_other()
    tickers_list = list(set(tickers_list))
    tickers_list = [ticker for ticker in tickers_list if ticker and not ticker.startswith('.')]
    return tickers_list

def get_data(ticker):
    columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'NetIncome', 'EBITDA', 'EBIT', 'GrossProfit', 'OperatingIncome', 
        'TotalRevenue', 'OperatingRevenue', 
        'TotalAssets', 'TotalLiabilitiesNetMinorityInterest', 'NetDebt', 
        'TotalEquityGrossMinorityInterest', 
        'OperatingExpense', 'ResearchAndDevelopment', 'SellingGeneralAndAdministration',
        'ShareIssued', 'OrdinarySharesNumber', 'CommonStockEquity', 
        'Dividends', 'DilutedEPS', 'BasicEPS', 'RetainedEarnings'
    ]

    history = get_history(ticker)
    history = history[columns]
    history = history.dropna()
    for column in history.columns:
        history[column] = (history[column] - history[column].min()) / (history[column].max() - history[column].min())
    return history
