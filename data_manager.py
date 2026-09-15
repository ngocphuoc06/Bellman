import pandas as pd
from scipy.stats import nbinom
from config import TOTAL_DAYS, TEST_DAYS

def generate_walmart_synthetic_data(days=TOTAL_DAYS, n_true=6.9, p_true=0.25):
    """
    Bước 1: Thu thập & Tải dữ liệu thô (Raw Data Ingestion)
    Giả lập dữ liệu chuỗi thời gian bán hàng Walmart.
    """
    dates = pd.date_range(start='2011-01-29', periods=days, freq='D') # M5 dataset starts 2011-01-29
    # Sinh dữ liệu nhu cầu từ phân phối Negative Binomial để giả lập hiện tượng siêu phân tán
    demands = nbinom.rvs(n_true, p_true, size=days)
    df = pd.DataFrame({'Date': dates, 'Demand': demands})
    return df

def split_data(df):
    """
    Bước 2: Phân tách tập dữ liệu ngay từ đầu (Immutable Data Splitting)
    Tuyệt đối không dùng random split để tránh data leakage.
    Chia theo trục thời gian (chronological split).
    """
    train_size = len(df) - TEST_DAYS
    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()
    return train_data, test_data
