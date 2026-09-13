import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# 導入 TensorFlow 與 Keras 相關套件
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# ⚙️ 訓練設定區
# ==========================================
BASE_FOLDER = "./" 
WINDOW_SIZE = 250 
# ==========================================

print("🚀 啟動 CNN 深度學習訓練引擎...")
# 檢查是否有 GPU 可用 (您的 RTX 4070)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"🔥 偵測到 GPU 顯示卡: {physical_devices[0]}，準備加速運算！")
else:
    print("⚠️ 未偵測到 GPU，將使用 CPU 進行運算。")

print(f"\n🔍 開始掃描目錄: {BASE_FOLDER}")
muscle_groups_found = set()
for root, dirs, files in os.walk(BASE_FOLDER):
    current_folder = os.path.basename(root)
    parent_folder = os.path.basename(os.path.dirname(root))
    if parent_folder.endswith("_data"):
        muscle_groups_found.add(current_folder)

if not muscle_groups_found:
    print("❌ 找不到任何肌肉群組資料夾！")
    exit()

print(f"✅ 共找到 {len(muscle_groups_found)} 種肌肉部位: {list(muscle_groups_found)}\n")
os.makedirs("pkl", exist_ok=True)

for target_muscle in muscle_groups_found:
    print("=" * 50)
    print(f"🚀 開始處理肌肉群組：【 {target_muscle} 】")
    
    all_csv_files = []
    for root, dirs, files in os.walk(BASE_FOLDER):
        if os.path.basename(root) == target_muscle:
            csv_pattern = os.path.join(root, "*_emg_2ch_data.csv")
            all_csv_files.extend(glob.glob(csv_pattern))
            
    valid_csv_files = [f for f in all_csv_files if not os.path.basename(f).startswith(("test", "QQ"))]
    
    if not valid_csv_files:
        print(f"⚠️ {target_muscle} 內沒有有效檔案，跳過。")
        continue
        
    X_raw = [] 
    y_raw = [] 

    for file in valid_csv_files:
        try:
            df = pd.read_csv(file)
            if len(df) < WINDOW_SIZE: continue
                
            filename = os.path.basename(file)
            label_part = filename.split("_2026")[0]
            
            # 滑動切窗 (保留原始波形結構: 250 x 2)
            for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
                window = df[['ch1', 'ch2']].iloc[i:i+WINDOW_SIZE].values
                X_raw.append(window)
                y_raw.append(label_part)
                
        except Exception as e:
            print(f"⚠️ 錯誤: {e}")

    X = np.array(X_raw)
    y = np.array(y_raw)
    
    if len(X) == 0: continue
    
    print(f"📊 資料載入完成！共 {len(X)} 筆樣本。資料形狀: {X.shape} (樣本數, 時間點, 通道數)")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # 針對 CNN 的特殊標準化 (將 3D 壓扁成 2D 給 Scaler 縮放，再彈回 3D)
    num_samples, time_steps, num_channels = X.shape
    X_reshaped = X.reshape(-1, num_channels)
    
    scaler = StandardScaler()
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(num_samples, time_steps, num_channels)

    # 分割資料集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    print("🧠 建構 1D-CNN 神經網路...")
    model = Sequential([
        # 第一層卷積 (提取淺層波形特徵)
        Conv1D(filters=32, kernel_size=10, activation='relu', input_shape=(WINDOW_SIZE, 2)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 第二層卷積 (提取深層肌肉收縮特徵)
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # 第三層卷積
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(), # 全域平均池化，降低維度防過擬合
        
        # 全連接層
        Dense(128, activation='relu'),
        Dropout(0.5), # 丟棄 50% 神經元，強力防止小樣本過擬合 (Overfitting)
        
        # 輸出層 (Softmax 轉換為各類別的機率)
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
                  
    # 設定早停機制 (如果測試集準確率連續 10 輪沒進步，就提早結束訓練)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("🚂 開始訓練 CNN 模型...")
    # epochs=50 代表最多訓練 50 輪
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=32, 
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop],
                        verbose=1) # verbose=1 會顯示進度條

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n🏆 訓練完成！CNN 測試集準確率: {test_acc * 100:.2f}%")
    print(f"🏷️ 辨識類別: {label_encoder.classes_}")

    # 儲存神經網路模型 (Keras 專用格式 .h5) 與其配件
    model_filename = os.path.join("pkl", f"emg_cnn_model_{target_muscle}.h5")
    scaler_filename = os.path.join("pkl", f"emg_scaler_{target_muscle}.pkl")
    encoder_filename = os.path.join("pkl", f"emg_label_encoder_{target_muscle}.pkl")

    # Keras 模型必須用自己的 save 函數儲存
    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(label_encoder, encoder_filename)
    print(f"💾 CNN 大腦已儲存至: {model_filename}")

print("\n🎉 全自動 CNN 批次訓練完畢！")