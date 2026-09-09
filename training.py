import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==========================================
# ⚙️ 訓練設定區
# ==========================================
# 假設這支程式跟您的日期資料夾 (20260318_data 等) 放在同一個大目錄下
BASE_FOLDER = "./" 

# 切窗大小 (Samples)，500Hz 採樣率下，250 點代表 0.5 秒
WINDOW_SIZE = 250 

# 日期資料夾的特徵，用來篩選出正確的母目錄 (避免抓到系統隱藏資料夾)
DATE_FOLDER_SUFFIX = "_data"
# ==========================================

def extract_features(window_data):
    """ 從一段時間窗內的波形提取 8 個數學特徵 """
    features = []
    # 確保只取 ch1 和 ch2 兩欄
    for col in ['ch1', 'ch2']:
        if col in window_data.columns:
            channel_data = window_data[col].values
            mav = np.mean(np.abs(channel_data))
            wl = np.sum(np.abs(np.diff(channel_data)))
            rms = np.sqrt(np.mean(channel_data**2))
            std = np.std(channel_data)
            features.extend([mav, wl, rms, std])
        else:
            # 如果真的遇到單通道資料 (缺 ch2)，補上 0 特徵防呆
            features.extend([0, 0, 0, 0])
    return features

print(f"🔍 步驟一：開始掃描並盤點所有的肌肉群組 (從 {BASE_FOLDER} 開始)")

# 1. 自動盤點出所有的「肌肉部位名稱」
# 透過掃描所有以 _data 結尾的資料夾，把裡面的子資料夾名稱收集起來
muscle_groups_found = set()

for root, dirs, files in os.walk(BASE_FOLDER):
    # 只看當前資料夾名稱
    current_folder = os.path.basename(root)
    # 找出上一層目錄的名稱
    parent_folder = os.path.basename(os.path.dirname(root))
    
    # 判斷邏輯：如果上一層是日期資料夾 (例如 20260325_data)，那當前這層就是肌肉名稱
    if parent_folder.endswith(DATE_FOLDER_SUFFIX):
        muscle_groups_found.add(current_folder)

if not muscle_groups_found:
    print(f"❌ 找不到任何肌肉群組資料夾，請確認您的日期資料夾結尾是 {DATE_FOLDER_SUFFIX}")
    exit()

print(f"✅ 盤點完畢！共找到 {len(muscle_groups_found)} 種肌肉部位配置：")
print(f"   👉 {list(muscle_groups_found)}\n")
print("=" * 50)

# 2. 針對每一個肌肉部位，進行獨立收集與訓練
for target_muscle in muscle_groups_found:
    print(f"\n🚀 開始處理肌肉群組：【 {target_muscle} 】")
    
    all_csv_files = []
    # 重新遞迴掃描一次，只收集當前迴圈這個 target_muscle 裡的 CSV
    for root, dirs, files in os.walk(BASE_FOLDER):
        if os.path.basename(root) == target_muscle:
            csv_pattern = os.path.join(root, "*_emg_2ch_data.csv")
            all_csv_files.extend(glob.glob(csv_pattern))
            
    # 過濾測試檔案
    valid_csv_files = [f for f in all_csv_files if not os.path.basename(f).startswith(("test", "QQ"))]
    
    if not valid_csv_files:
        print(f"⚠️ {target_muscle} 資料夾內沒有有效的訓練檔案，跳過。")
        continue
        
    print(f"  📂 找到 {len(valid_csv_files)} 個 CSV 檔案，準備萃取特徵...")
    
    X = [] 
    y = [] 

    # 讀取資料並切窗
    for file in valid_csv_files:
        try:
            df = pd.read_csv(file)
            if len(df) < WINDOW_SIZE:
                continue
                
            # 從檔案名稱自動推斷標籤 (擷取 "_2026" 前面的字串)
            filename = os.path.basename(file)
            label_part = filename.split("_2026")[0]
            
            # 滑動切窗
            for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
                window = df.iloc[i:i+WINDOW_SIZE]
                features = extract_features(window)
                X.append(features)
                y.append(label_part)
                
        except Exception as e:
            print(f"  ⚠️ 讀取檔案 {filename} 時發生錯誤: {e}")

    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
         print(f"  ❌ {target_muscle} 沒有足夠的資料長度，跳過訓練。")
         continue

    print(f"  📊 特徵萃取完成！共產生 {len(X)} 筆訓練樣本。")

    # 機器學習標準流程
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    print("  🧠 模型訓練中...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"  🏆 訓練完成！測試集準確率: {accuracy * 100:.2f}%")
    print(f"  🏷️ 學習到的動作標籤: {label_encoder.classes_}")

    # --- 新增：確保 pkl 資料夾存在，若無則自動建立 ---
    os.makedirs("pkl", exist_ok=True)

    # 儲存此肌肉部位專屬的模型 (加上 pkl/ 路徑)
    model_filename = os.path.join("pkl", f"emg_model_{target_muscle}.pkl")
    scaler_filename = os.path.join("pkl", f"emg_scaler_{target_muscle}.pkl")
    encoder_filename = os.path.join("pkl", f"emg_label_encoder_{target_muscle}.pkl")

    joblib.dump(rf_classifier, model_filename)
    joblib.dump(scaler, scaler_filename)
    joblib.dump(label_encoder, encoder_filename)
    print(f"  💾 模型已安全存放至: {model_filename}")

print("\n🎉 全部處理完畢！所有肌肉部位的模型皆已產出並集中管理於 pkl/ 資料夾中。")