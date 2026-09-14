import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import serial
import serial.tools.list_ports
import time
import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import datetime
import joblib 

# 解決 Matplotlib 中文顯示
plt.rcParams['font.sans-serif'] = ['PingFang TC', 'Arial Unicode MS', 'Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False 

class EMGGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG 動態多模型預測工具 (決策融合版)")
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")
        
        self.ser = None
        self.baud_rate = 921600 # 🚀 關鍵修改：配合 STM32 的真實高速傳輸速率
        self.is_recording = False
        self.thread_running = True
        
        self.data_queue = queue.Queue()
        self.all_data_log = []
        self.num_channels = 2
        self.y_data = [[] for _ in range(self.num_channels)]
        self.ch_vars = [] 
        
        self.current_label = ""
        self.window_size = 250 
        self.last_predict_time = time.time() 

        # 這裡存放兩個通道獨立的大腦模型
        self.models = {
            'ch1': {'active': False, 'model': None, 'scaler': None, 'le': None, 'name': ''},
            'ch2': {'active': False, 'model': None, 'scaler': None, 'le': None, 'name': ''}
        }

        self.setup_ui()
        self.setup_plot()
        
        self.serial_thread = threading.Thread(target=self.read_serial_task)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        
        self.update_plot()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 延遲 0.5 秒後嘗試自動連線 (確保 GUI 先畫完)
        self.root.after(500, self.connect_serial)

    def scan_available_models(self):
        """ 掃描 pkl/ 資料夾中所有可用的單一肌肉模型 """
        model_files = glob.glob(os.path.join("pkl", "emg_model_*.pkl"))
        muscle_names = ["-- 停用此通道 --"] # 第一個選項是停用
        for file in model_files:
            name = os.path.basename(file).replace("emg_model_", "").replace(".pkl", "")
            muscle_names.append(name.upper()) # 轉大寫顯示比較好看
        return muscle_names

    def assign_model(self, channel_key, combobox_widget):
        """ 當使用者在下拉選單選擇部位時，載入對應的模型 """
        selected = combobox_widget.get()
        if not selected or selected == "-- 停用此通道 --":
            self.models[channel_key]['active'] = False
            self.update_predict_label()
            return

        muscle = selected.lower() # 轉回小寫讀檔
        model_path = os.path.join("pkl", f"emg_model_{muscle}.pkl")
        scaler_path = os.path.join("pkl", f"emg_scaler_{muscle}.pkl")
        encoder_path = os.path.join("pkl", f"emg_label_encoder_{muscle}.pkl")

        try:
            if os.path.exists(model_path):
                self.models[channel_key]['model'] = joblib.load(model_path)
                self.models[channel_key]['scaler'] = joblib.load(scaler_path)
                self.models[channel_key]['le'] = joblib.load(encoder_path)
                self.models[channel_key]['name'] = selected
                self.models[channel_key]['active'] = True
                print(f"🧠 通道 {channel_key.upper()} 已成功指派為: {selected}")
            else:
                self.models[channel_key]['active'] = False
                messagebox.showerror("錯誤", f"找不到 {selected} 的模型檔案")
        except Exception as e:
            self.models[channel_key]['active'] = False
            print(f"載入錯誤: {e}")
            
        self.update_predict_label()

    def update_predict_label(self):
        """ 更新畫面上方的狀態文字 """
        ch1_act = self.models['ch1']['active']
        ch2_act = self.models['ch2']['active']
        if ch1_act and ch2_act:
            text = "決策融合模式 (等待雙通道訊號...)"
        elif ch1_act or ch2_act:
            text = "單一部位預測模式 (等待訊號...)"
        else:
            text = "AI 已停用 (請指派模型)"
        self.lbl_predict.config(text=text, bg="#eeeeee", fg="gray")

    def setup_ui(self):
        control_frame = tk.Frame(self.root, pady=5, padx=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        row1_frame = tk.Frame(control_frame)
        row1_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        row2_frame = tk.Frame(control_frame)
        row2_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

        tk.Label(row1_frame, text="標籤:").pack(side=tk.LEFT)
        self.entry_label = tk.Entry(row1_frame, width=10)
        self.entry_label.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = tk.Button(row1_frame, text="▶ 錄製", command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_stop = tk.Button(row1_frame, text="■ 停止", state=tk.DISABLED, command=self.stop_recording)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        available_muscles = self.scan_available_models()
        
        # --- Ch1 指派選單 ---
        tk.Label(row2_frame, text="Ch1 (綠線) 貼片部位:", font=("Arial", 12, "bold"), fg="green").pack(side=tk.LEFT)
        self.cb_ch1 = ttk.Combobox(row2_frame, values=available_muscles, state="readonly", width=15)
        self.cb_ch1.pack(side=tk.LEFT, padx=5)
        self.cb_ch1.bind("<<ComboboxSelected>>", lambda e: self.assign_model('ch1', self.cb_ch1))
        
        # --- Ch2 指派選單 ---
        tk.Label(row2_frame, text="Ch2 (黃線) 貼片部位:", font=("Arial", 12, "bold"), fg="#b8860b").pack(side=tk.LEFT, padx=(20, 0))
        self.cb_ch2 = ttk.Combobox(row2_frame, values=available_muscles, state="readonly", width=15)
        self.cb_ch2.pack(side=tk.LEFT, padx=5)
        self.cb_ch2.bind("<<ComboboxSelected>>", lambda e: self.assign_model('ch2', self.cb_ch2))

        self.lbl_predict = tk.Label(row2_frame, text="AI 已停用 (請指派模型)", font=("Arial", 16, "bold"), fg="gray", bg="#eeeeee", padx=15, pady=2)
        self.lbl_predict.pack(side=tk.LEFT, padx=30)
        
        # 預設選取
        if len(available_muscles) > 1:
            self.cb_ch1.current(0)
            self.cb_ch2.current(0)

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        colors = ['#00ff00', '#ffff00']
        self.lines = []
        for i in range(self.num_channels):
            line, = self.ax.plot([], [], lw=1.5, color=colors[i], label=f'Ch{i+1}')
            self.lines.append(line)
        
        self.ax.set_facecolor('black') 
        self.fig.patch.set_facecolor('#222222') 
        self.ax.set_ylim(0, 4200)   
        self.ax.set_xlim(0, 500)    
        self.ax.legend(loc='upper right')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def connect_serial(self):
        try:
            ports = [p.device for p in serial.tools.list_ports.comports() if 'usbmodem' in p.device]
            if not ports:
                print("❌ 錯誤：找不到 usbmodem 裝置")
                return
            port = ports[0]
            self.ser = serial.Serial(port, self.baud_rate, timeout=1)
            # 清空緩衝區
            self.ser.reset_input_buffer()
            print(f"🔌 Serial 已連接: {self.ser.name}")
        except Exception as e:
            print(f"⚠️ 無法自動連接 Serial: {e}")

    def start_recording(self):
        # 💡 取消標籤檢查，因為現在主要是用來啟動預測
        # lbl = self.entry_label.get().strip()
        # if not lbl: return messagebox.showwarning("警告", "請輸入標籤")
        # self.current_label = lbl
        
        # 💡 在開始前清空舊的暫存資料，確保圖表從 0 開始畫
        for i in range(self.num_channels):
            self.y_data[i].clear()
            self.lines[i].set_data([], [])
        while not self.data_queue.empty():
            self.data_queue.get()
        if self.ser:
            self.ser.reset_input_buffer()

        self.is_recording = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)

    def stop_recording(self):
        self.is_recording = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        # 💡 停止時將狀態標籤歸零
        self.update_predict_label()

    def extract_features(self, channel_data):
        # 🚀 關鍵修復：把 Python 基礎的 list 轉成 NumPy 陣列，才能進行平方運算！
        channel_data = np.array(channel_data)
        
        mav = np.mean(np.abs(channel_data))
        wl = np.sum(np.abs(np.diff(channel_data)))
        rms = np.sqrt(np.mean(channel_data**2))
        std = np.std(channel_data)
        return [mav, wl, rms, std]

    def read_serial_task(self):
        while self.thread_running:
            if self.ser and self.ser.is_open and self.ser.in_waiting > 0:
                try:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # 🚀 關鍵修改：只有在 is_recording (按鈕按下) 時，才把資料放進 Queue 去畫圖！
                    if self.is_recording:
                        parts = line.split(',')
                        if len(parts) >= 2: 
                            self.data_queue.put([int(parts[0]), int(parts[1])])
                except Exception as e: 
                    pass # 忽略雜訊
            else: time.sleep(0.01)

    def update_plot(self):
        # 🚀 關鍵修改：只有在 is_recording (按鈕按下) 時，才進行圖表更新與 AI 預測！
        if self.is_recording:
            updated = False
            while not self.data_queue.empty():
                vals = self.data_queue.get()
                for i in range(2):
                    self.y_data[i].append(vals[i])
                    if len(self.y_data[i]) > 600: self.y_data[i].pop(0)
                updated = True
            
            if updated and len(self.y_data[0]) > 0:
                for i in range(2):
                    y_disp = self.y_data[i][-500:]
                    self.lines[i].set_data(range(len(y_disp)), y_disp)
                self.canvas.draw_idle()  
                
                # --- 🚀 決策融合推論邏輯 (Decision Fusion) ---
                current_time = time.time()
                if len(self.y_data[0]) >= self.window_size and (current_time - self.last_predict_time > 0.2):
                    
                    predictions = [] # 用來收集各通道的預測結果
                    
                    # 💡 加入 try-except 防護罩，避免少數特徵錯誤導致介面當機卡死
                    try:
                        # 處理 Ch1 預測
                        if self.models['ch1']['active']:
                            win = self.y_data[0][-self.window_size:]
                            feats = self.extract_features(win)
                            scaled = self.models['ch1']['scaler'].transform([feats])
                            probs = self.models['ch1']['model'].predict_proba(scaled)[0]
                            best_idx = np.argmax(probs)
                            pred_text = self.models['ch1']['le'].inverse_transform([best_idx])[0]
                            confidence = probs[best_idx] * 100
                            predictions.append({'channel': 'Ch1', 'text': pred_text, 'conf': confidence})

                        if self.models['ch2']['active']:
                            win = self.y_data[1][-self.window_size:]
                            feats = self.extract_features(win)
                            scaled = self.models['ch2']['scaler'].transform([feats])
                            probs = self.models['ch2']['model'].predict_proba(scaled)[0]
                            best_idx = np.argmax(probs)
                            pred_text = self.models['ch2']['le'].inverse_transform([best_idx])[0]
                            confidence = probs[best_idx] * 100
                            predictions.append({'channel': 'Ch2', 'text': pred_text, 'conf': confidence})

                        if predictions:
                            predictions.sort(key=lambda x: x['conf'], reverse=True)
                            best_pred = predictions[0]
                            
                            if len(predictions) == 2:
                                ui_text = f"融合判定: 【 {best_pred['text']} 】 (採信 {best_pred['channel']} 信心: {best_pred['conf']:.0f}%)"
                                self.lbl_predict.config(text=ui_text, bg="#e8f5e9", fg="#2e7d32")
                            else:
                                ui_text = f"單一判定: 【 {best_pred['text']} 】 (信心: {best_pred['conf']:.0f}%)"
                                self.lbl_predict.config(text=ui_text, bg="#e3f2fd", fg="#1565c0")

                    except ValueError as ve:
                        # 🚀 把被隱藏的錯誤顯示在介面上
                        print(f"⚠️ 預測維度錯誤: {ve}")
                        self.lbl_predict.config(text="錯誤：特徵數量不相符！請重跑訓練程式", bg="red", fg="white")
                    except Exception as e:
                        # 🚀 把其他所有被隱藏的錯誤也顯示出來
                        print(f"⚠️ 預測未知錯誤: {e}")
                        self.lbl_predict.config(text=f"AI 錯誤: {str(e)[:20]}", bg="red", fg="white")
                            
                    self.last_predict_time = current_time

        if self.thread_running:
            self.root.after(15, self.update_plot)

    def on_closing(self):
        self.thread_running = False
        if self.ser: self.ser.close()
        self.root.destroy()

if __name__ == "__main__":
    app = EMGGUIApp(tk.Tk())
    app.root.mainloop()