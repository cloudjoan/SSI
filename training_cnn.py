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

print("🚀 啟動 CNN 單一肌肉獨立訓練引擎...")
# 檢查是否有 GPU 可用 (例如 Mac 的 MPS 或 NVIDIA)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"🔥 偵測到 GPU 顯示卡: {physical_devices[0]}，準備加速運算！")
else:
    print("⚠️ 未偵測到 GPU，將使用 CPU 進行運算 (Intel i7 沒問題的！)。")

print("\n🔍 開始掃描目錄並自動拆分肌肉通道...")
# 用來存放所有被發現的單一肌肉資料池
muscle_data_pool = {}

for root, dirs, files in os.walk(BASE_FOLDER):
    current_folder = os.path.basename(root)
    parent_folder = os.path.basename(os.path.dirname(root))
    
    if parent_folder.endswith("_data"):
        csv_pattern = os.path.join(root, "*_emg_2ch_data.csv")
        csv_files = glob.glob(csv_pattern)
        valid_files = [f for f in csv_files if not os.path.basename(f).startswith(("test", "QQ"))]
        
        if not valid_files:
            continue
            
        # 拆解資料夾名稱 (例如 'zgm_abd' -> ['zgm', 'abd'])
        muscle_tags = current_folder.lower().split('_')
        ch1_muscle = muscle_tags[0]
        ch2_muscle = muscle_tags[1] if len(muscle_tags) > 1 else None
        
        # 開闢資料池
        if ch1_muscle not in muscle_data_pool:
            muscle_data_pool[ch1_muscle] = {'X': [], 'y': []}
        if ch2_muscle and ch2_muscle not in muscle_data_pool:
            muscle_data_pool[ch2_muscle] = {'X': [], 'y': []}

        print(f"📂 處理資料夾: {current_folder} (分配 -> Ch1:{ch1_muscle}, Ch2:{ch2_muscle})")
        for file in valid_files:
            try:
                df = pd.read_csv(file)
                if len(df) < WINDOW_SIZE: continue
                    
                label_part = os.path.basename(file).split("_2026")[0]
                
                # 滑動切窗
                for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
                    window = df.iloc[i:i+WINDOW_SIZE]
                    
                    # 處理 Ch1 (CNN 吃的是原始波形序列，不再算特徵！)
                    # 我們把 250 個點變成形狀為 (250, 1) 的矩陣
                    ch1_raw = window['ch1'].values.reshape(-1, 1)
                    muscle_data_pool[ch1_muscle]['X'].append(ch1_raw)
                    muscle_data_pool[ch1_muscle]['y'].append(label_part)
                    
                    # 處理 Ch2
                    if ch2_muscle:
                        ch2_raw = window['ch2'].values.reshape(-1, 1)
                        muscle_data_pool[ch2_muscle]['X'].append(ch2_raw)
                        muscle_data_pool[ch2_muscle]['y'].append(label_part)
                        
            except Exception as e:
                print(f"⚠️ 讀取檔案錯誤: {e}")

os.makedirs("pkl", exist_ok=True)
print("\n" + "="*50)

# 開始針對每個單一肌肉進行獨立 CNN 訓練
for target_muscle, data in muscle_data_pool.items():
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    if len(X) == 0:
        continue
        
    print(f"🚀 開始訓練單一部位 (CNN): 【 {target_muscle.upper()} 】 (共 {len(X)} 筆樣本)")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # 針對 3D 結構 (樣本數, 250, 1) 的特殊標準化
    num_samples, time_steps, num_channels = X.shape
    X_reshaped = X.reshape(-1, num_channels)
    
    scaler = StandardScaler()
    X_scaled_reshaped = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled_reshaped.reshape(num_samples, time_steps, num_channels)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # 建構單通道專用的 1D-CNN (輸入維度改為 (250, 1))
    model = Sequential([
        Conv1D(filters=32, kernel_size=10, activation='relu', input_shape=(WINDOW_SIZE, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("🚂 神經網路訓練中...")
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  🏆 {target_muscle.upper()} 測試集準確率: {test_acc * 100:.2f}% | 標籤: {label_encoder.classes_}")

    model_filename = os.path.join("pkl", f"emg_cnn_model_{target_muscle}.h5")
    scaler_filename = os.path.join("pkl", f"emg_scaler_{target_muscle}.pkl")
    encoder_filename = os.path.join("pkl", f"emg_label_encoder_{target_muscle}.pkl")

    model.save(model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(label_encoder, encoder_filename)

print("="*50)
print("🎉 所有單一部位 CNN 模型皆已產出！")