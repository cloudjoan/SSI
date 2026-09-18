import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

BASE_FOLDER = "./" 
WINDOW_SIZE = 250 

def extract_single_ch_features(channel_data):
    """ 
    針對單一通道提取 4 個時域特徵 (不再合併雙通道) 
    """
    mav = np.mean(np.abs(channel_data))
    wl = np.sum(np.abs(np.diff(channel_data)))
    rms = np.sqrt(np.mean(channel_data**2))
    std = np.std(channel_data)
    return [mav, wl, rms, std]

print("🔍 開始掃描目錄並自動拆分肌肉通道...")
# 建立一個字典，用來存放所有被發現的單一肌肉資料
# 格式: {'zgm': {'X': [], 'y': []}, 'abd': {'X': [], 'y': []}}
muscle_data_pool = {}

for root, dirs, files in os.walk(BASE_FOLDER):
    current_folder = os.path.basename(root)
    parent_folder = os.path.basename(os.path.dirname(root))
    
    # 確認這層是肌肉資料夾 (上一層是 _data 結尾)
    if parent_folder.endswith("_data"):
        # 尋找 CSV 檔案
        csv_pattern = os.path.join(root, "*_emg_2ch_data.csv")
        csv_files = glob.glob(csv_pattern)
        valid_files = [f for f in csv_files if not os.path.basename(f).startswith(("test"))]
        
        if not valid_files:
            continue
            
        # 拆解資料夾名稱，例如 'zgm_abd' 會變成 ['zgm', 'abd']
        # 統一轉小寫避免 Zgm 和 zgm 被當作不同肌肉
        muscle_tags = current_folder.lower().split('_')
        ch1_muscle = muscle_tags[0]
        ch2_muscle = muscle_tags[1] if len(muscle_tags) > 1 else None
        
        # 確保字典裡有這些肌肉的空間
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
                    
                    # 處理 Ch1
                    ch1_feats = extract_single_ch_features(window['ch1'].values)
                    muscle_data_pool[ch1_muscle]['X'].append(ch1_feats)
                    muscle_data_pool[ch1_muscle]['y'].append(label_part)
                    
                    # 如果有 Ch2，也獨立處理並放入其專屬池子
                    if ch2_muscle:
                        ch2_feats = extract_single_ch_features(window['ch2'].values)
                        muscle_data_pool[ch2_muscle]['X'].append(ch2_feats)
                        muscle_data_pool[ch2_muscle]['y'].append(label_part)
                        
            except Exception as e:
                print(f"⚠️ 讀取檔案錯誤: {e}")

os.makedirs("pkl", exist_ok=True)
print("\n" + "="*50)

# 開始針對每個單一肌肉進行獨立訓練
for target_muscle, data in muscle_data_pool.items():
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    if len(X) == 0:
        continue
        
    print(f"🚀 開始訓練單一部位: 【 {target_muscle.upper()} 】 (共 {len(X)} 筆樣本)")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  🏆 準確率: {accuracy * 100:.2f}% | 標籤: {label_encoder.classes_}")

    model_filename = os.path.join("pkl", f"emg_model_{target_muscle}.pkl")
    scaler_filename = os.path.join("pkl", f"emg_scaler_{target_muscle}.pkl")
    encoder_filename = os.path.join("pkl", f"emg_label_encoder_{target_muscle}.pkl")

    joblib.dump(rf_classifier, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(label_encoder, encoder_filename)

print("="*50)
print("🎉 所有單一部位模型皆已產出，並獨立存放於 pkl/ 資料夾！")