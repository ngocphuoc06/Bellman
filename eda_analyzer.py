import matplotlib.pyplot as plt

def perform_eda(train_data):
    """
    Bước 3: Khám phá & Trực quan hóa dữ liệu (EDA trên tập Train)
    Tính toán thống kê và vẽ biểu đồ phân bố để phát hiện dị biệt.
    Tuyệt đối không dùng thông tin từ tập Test.
    """
    train_mean = train_data['Demand'].mean()
    train_var = train_data['Demand'].var()
    
    print(f"Trung bình nhu cầu (Mean): {train_mean:.2f}")
    print(f"Phương sai nhu cầu (Variance): {train_var:.2f}")
    
    if train_var > train_mean:
        print("=> Phát hiện hiện tượng Siêu phân tán (Overdispersion). Dữ liệu phù hợp với phân phối Negative Binomial.")
        
    # Trực quan hóa
    plt.hist(train_data['Demand'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Phân phối Nhu cầu (Train Data)')
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.savefig('demand_histogram.png')
    print("Đã lưu biểu đồ phân phối nhu cầu ra file 'demand_histogram.png'.")
    plt.close()
    
    return train_mean, train_var

def fit_parameters(train_mean, train_var, verbose=True):
    """
    Bước 4: Tiền xử lý & Kỹ thuật đặc trưng (Fit trên Train)
    Học tham số n và p bằng phương pháp Moment Matching từ tập Train.
    """
    # Tránh chia cho 0 hoặc giá trị âm nếu không có siêu phân tán
    if train_var <= train_mean:
        train_var = train_mean + 0.01 
        
    p_estimated = train_mean / train_var
    n_estimated = (train_mean ** 2) / (train_var - train_mean)
    
    if verbose:
        print(f"Học tham số: p_estimated = {p_estimated:.4f}, n_estimated = {n_estimated:.4f}")
    return n_estimated, p_estimated
