import os
import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
from scipy.stats import nbinom
from config import TOTAL_DAYS, TEST_DAYS

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

def fetch_online_retail_real_data(stock_code='20675', cache_filename='real_demand_daily.csv'):
    """
    Tải & Xử lý dữ liệu THẬT (Real Data) từ tập giao dịch bán lẻ Online Retail (UCI Machine Learning Repository).
    - Tự động tải file giao dịch (540k+ đơn hàng thực tế).
    - Lọc giao dịch của sản phẩm cụ thể (Mặc định '20675' - BLUE POLKADOT BOWL).
    - Gom nhóm nhu cầu mua hàng theo ngày (Daily Demand Aggregation).
    - Lưu cache vào d:/Bellman/data/ để tiết kiệm thời gian cho các lần chạy sau.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, cache_filename)

    if os.path.exists(cache_path):
        print(f">> [Data Manager] Đã tìm thấy dữ liệu thật đã lưu cache tại: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=['Date'])
        return df

    print(">> [Data Manager] Đang tải dữ liệu giao dịch bán lẻ THẬT từ nguồn online (khoảng 25MB)...")
    url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Online_Retail/Online_Retail.csv"
    
    try:
        raw_df = pd.read_csv(url, encoding='latin1')
        raw_df = raw_df[raw_df['Quantity'] > 0] # Chỉ lấy đơn xuất bán thành công
        
        # Parse ngày giao dịch
        raw_df['InvoiceDate'] = pd.to_datetime(raw_df['InvoiceDate'], errors='coerce')
        raw_df['Date'] = raw_df['InvoiceDate'].dt.date
        
        # Lọc mặt hàng thực tế
        df_item = raw_df[raw_df['StockCode'] == stock_code].copy()
        if len(df_item) == 0:
            # Nếu không tìm thấy mã, lấy mã sản phẩm phổ biến nhất
            stock_code = raw_df.groupby('StockCode')['Quantity'].sum().idxmax()
            df_item = raw_df[raw_df['StockCode'] == stock_code].copy()
            
        desc = df_item['Description'].dropna().iloc[0] if len(df_item['Description'].dropna()) > 0 else stock_code
        print(f">> [Data Manager] Đã trích xuất dữ liệu sản phẩm THẬT: Mã {stock_code} ({desc})")

        # Gom nhóm nhu cầu theo ngày
        daily_df = df_item.groupby('Date')['Quantity'].sum().reset_index()
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df.rename(columns={'Quantity': 'Demand'}, inplace=True)
        
        # Tạo chuỗi ngày liên tục (bổ sung những ngày không có đơn bán nhu cầu = 0)
        full_dates = pd.date_range(start=daily_df['Date'].min(), end=daily_df['Date'].max(), freq='D')
        daily_df = daily_df.set_index('Date').reindex(full_dates, fill_value=0).reset_index()
        daily_df.rename(columns={'index': 'Date'}, inplace=True)
        
        # Cache dữ liệu sạch xuống ổ đĩa
        daily_df.to_csv(cache_path, index=False)
        print(f">> [Data Manager] Đã tạo & lưu cache thành công vào: {cache_path}")
        return daily_df

    except Exception as e:
        print(f"⚠️ Không thể tải dữ liệu online ({e}). Chuyển sang sinh dữ liệu giả lập...")
        return generate_walmart_synthetic_data()

def load_custom_csv_data(filepath, date_col='Date', demand_col='Demand'):
    """
    Tải dữ liệu THẬT từ một file CSV địa phương do người dùng cung cấp.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file CSV tại đường dẫn: {filepath}")
    
    df = pd.read_csv(filepath)
    if date_col not in df.columns or demand_col not in df.columns:
        raise ValueError(f"File CSV cần chứa 2 cột '{date_col}' và '{demand_col}'")
    
    df[date_col] = pd.to_datetime(df[date_col])
    df_clean = pd.DataFrame({
        'Date': df[date_col],
        'Demand': df[demand_col].astype(int)
    }).sort_values('Date').reset_index(drop=True)
    return df_clean

def generate_walmart_synthetic_data(days=TOTAL_DAYS, n_true=6.9, p_true=0.25):
    """
    Bước 1: Thu thập & Tải dữ liệu (Raw Data Ingestion)
    Giả lập dữ liệu chuỗi thời gian bán hàng Walmart.
    """
    dates = pd.date_range(start='2011-01-29', periods=days, freq='D')
    demands = nbinom.rvs(n_true, p_true, size=days)
    df = pd.DataFrame({'Date': dates, 'Demand': demands})
    return df

def split_data(df, test_days=TEST_DAYS):
    """
    Bước 2: Phân tách tập dữ liệu ngay từ đầu (Immutable Data Splitting)
    Tuyệt đối không dùng random split để tránh data leakage.
    Chia theo trục thời gian (chronological split).
    """
    if len(df) <= test_days * 2:
        train_size = int(len(df) * 0.8)
    else:
        train_size = len(df) - test_days

    train_data = df.iloc[:train_size].copy()
    test_data = df.iloc[train_size:].copy()
    return train_data, test_data
