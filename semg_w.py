import tkinter as tk
from tkinter import messagebox
import serial
import serial.tools.list_ports
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
import datetime

class EMGGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG 示波器與訊號收集工具 (自動偵測 1/2 通道)")
        
        # --- 設定視窗為 100% 螢幕大小 ---
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        
        # --- 變數初始化 ---
        self.serial_port = None
        self.ser = None
        self.baud_rate = 115200
        self.is_recording = False
        self.thread_running = True
        
        self.data_queue = queue.Queue()
        self.all_data_log = []
        
        # 預設準備 2 個通道的畫圖陣列
        self.num_channels = 2
        self.y_data = [[] for _ in range(self.num_channels)]
        
        self.current_label = ""
        self.current_filename = ""

        # --- 建立 GUI 介面 ---
        self.setup_ui()
        self.setup_plot()
        
        # --- 啟動背景執行緒與更新迴圈 ---
        self.serial_thread = threading.Thread(target=self.read_serial_task)
        self.serial_thread.daemon = True
        self.serial_thread.start()
        
        self.update_plot()
        
        # 處理視窗關閉事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 🚀 程式開啟後，立刻嘗試自動連線並進入「示波器預覽模式」
        self.root.after(500, self.connect_serial)

    def setup_ui(self):
        """ 建立上方控制面板 """
        control_frame = tk.Frame(self.root, pady=10, padx=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(control_frame, text="發音文字 (Label):", font=("Arial", 14)).pack(side=tk.LEFT, padx=(0, 5))
        self.entry_label = tk.Entry(control_frame, font=("Arial", 14), width=15)
        self.entry_label.pack(side=tk.LEFT, padx=5)
        
        self.btn_start = tk.Button(control_frame, text="▶ 開始錄製", font=("Arial", 14, "bold"), width=12, command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(control_frame, text="■ 停止並存檔", font=("Arial", 14, "bold"), width=12, state=tk.DISABLED, command=self.stop_recording)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar()
        self.status_var.set("狀態: 尋找硬體設備中...")
        self.lbl_status = tk.Label(control_frame, textvariable=self.status_var, font=("Arial", 12))
        self.lbl_status.pack(side=tk.LEFT, padx=20)

    def setup_plot(self):
        """ 建立 Matplotlib 圖表並嵌入 Tkinter """
        plt.rcParams['figure.dpi'] = 100 
        
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        
        # 設定 2 條線的顏色 (綠色為 Ch1, 黃色為 Ch2)
        colors = ['#00ff00', '#ffff00']
        self.lines = []
        for i in range(self.num_channels):
            line, = self.ax.plot([], [], lw=1.5, color=colors[i], label=f'Ch{i+1}')
            self.lines.append(line)
        
        # 樣式設定 (黑底綠/黃線，經典示波器風格)
        self.ax.set_facecolor('black') 
        self.fig.patch.set_facecolor('#222222') 
        self.ax.set_ylim(0, 4200)   
        self.ax.set_xlim(0, 500)    
        self.ax.set_title("Real-time Oscilloscope (即時示波器預覽)", color='white')
        self.ax.set_xlabel("Time (Samples)", color='white')
        self.ax.set_ylabel("ADC Value", color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.grid(True, color='#444444', linestyle='--')
        self.ax.legend(loc='upper right', facecolor='black', labelcolor='white')

        # 將圖表嵌入 Tkinter Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def find_mac_port(self):
        """ 自動尋找 STM32 的 Port """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if 'usbmodem' in port.device.lower() or 'COM' in port.device.upper():
                return port.device
        if ports:
            return ports[0].device
        return '/dev/tty.usbmodem1103' 

    def connect_serial(self):
        """ 連接 Serial Port """
        try:
            port_name = self.find_mac_port()
            self.ser = serial.Serial(port_name, self.baud_rate, timeout=1)
            self.ser.reset_input_buffer()
            self.status_var.set(f"狀態: 🟢 已連線 ({port_name})，示波器即時預覽中...")
            return True
        except Exception as e:
            self.status_var.set("狀態: 🔴 連線失敗，請檢查 USB 傳輸線。")
            return False

    def start_recording(self):
        """ 開始錄製按鈕事件 """
        label_text = self.entry_label.get().strip()
        if not label_text:
            messagebox.showwarning("警告", "請先輸入「發音文字」！")
            return

        if not self.ser or not self.ser.is_open:
            if not self.connect_serial():
                messagebox.showerror("錯誤", "無法連線到 STM32 硬體！")
                return

        self.current_label = label_text
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        safe_label = "".join(c for c in label_text if c.isalnum() or c in (' ', '_', '-')).rstrip()
        self.current_filename = f"{safe_label}_{timestamp_str}_emg_data.csv"
        
        self.all_data_log.clear()
        
        # 清空圖表舊線條，準備記錄新動作
        for i in range(self.num_channels):
            self.y_data[i].clear()
            self.lines[i].set_data([], [])
            
        while not self.data_queue.empty():
            self.data_queue.get()
            
        self.canvas.draw_idle()

        self.ax.set_title(f"🔴 RECORDING... Label: {self.current_label}", color='#ff4444')
        self.entry_label.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set(f"狀態: 🔴 錄製中... (動作標籤: {self.current_label})")
        
        if self.ser:
            self.ser.reset_input_buffer()
        self.is_recording = True

    def stop_recording(self):
        """ 停止錄製按鈕事件 """
        self.is_recording = False
        
        if self.all_data_log:
            try:
                with open(self.current_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 判斷記錄到的是單通道還是雙通道，動態寫入標題
                    if len(self.all_data_log[0]) == 4: # [時間, v1, v2, 標籤]
                        writer.writerow(['timestamp', 'ch1', 'ch2', 'label'])
                    else: # [時間, v1, 標籤]
                        writer.writerow(['timestamp', 'ch1', 'label'])
                    writer.writerows(self.all_data_log)
                messagebox.showinfo("存檔成功", f"成功儲存 {len(self.all_data_log)} 筆數據！\n\n檔名: {self.current_filename}")
            except Exception as e:
                messagebox.showerror("存檔失敗", f"存檔時發生錯誤: {e}")
        else:
            messagebox.showwarning("警告", "沒有收集到任何數據，未產生檔案。")

        self.entry_label.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set("狀態: 🟢 已停止，恢復示波器預覽模式。")
        self.ax.set_title("Real-time Oscilloscope (即時示波器預覽)", color='white')
        self.canvas.draw_idle()

    def read_serial_task(self):
        """ 背景執行緒：負責讀取 Serial 數據 (拔掉 is_recording 限制，隨時預覽) """
        while self.thread_running:
            if self.ser and self.ser.is_open:
                try:
                    if self.ser.in_waiting > 0:
                        raw_line = self.ser.readline()
                        line = raw_line.decode('utf-8', errors='ignore').strip()
                        parts = line.split(',')
                        
                        # 💡 動態適應：過濾出真實的數字 (支援 1 個或 2 個)
                        vals = []
                        for p in parts:
                            if p.isdigit():
                                vals.append(int(p))
                        
                        # 如果有收到數據，就放進佇列畫圖
                        if len(vals) > 0:
                            timestamp = time.time()
                            self.data_queue.put(vals) # 隨時放入 Queue 供畫圖使用
                            
                            # 如果正在錄製，才儲存到 list 裡面準備存檔
                            if self.is_recording:
                                self.all_data_log.append([timestamp] + vals + [self.current_label])
                    else:
                        time.sleep(0.001)
                except Exception as e:
                    time.sleep(0.1)
            else:
                time.sleep(0.05)

    def update_plot(self):
        """ 更新圖表 (示波器動態滾動效果) """
        updated = False
        
        # 💡 解除錄製鎖定：只要 Queue 裡面有資料，就一直拿出來畫！
        while not self.data_queue.empty():
            vals = self.data_queue.get()
            
            # 將數值分配到對應的通道 (收到 1 個就畫 1 條，收到 2 個畫 2 條)
            for i in range(min(len(vals), self.num_channels)):
                self.y_data[i].append(vals[i])
                if len(self.y_data[i]) > 500: # 維持示波器畫面寬度
                    self.y_data[i].pop(0)
                    
            # 如果只收到 1 個通道，第 2 個通道補 0 (確保畫圖陣列不崩潰)
            if len(vals) == 1 and self.num_channels == 2:
                self.y_data[1].append(0)
                if len(self.y_data[1]) > 500:
                    self.y_data[1].pop(0)
                    
            updated = True
        
        if updated and len(self.y_data[0]) > 0:
            x_data = range(len(self.y_data[0]))
            for i in range(self.num_channels):
                # 如果該通道有資料，更新該條線
                if len(self.y_data[i]) == len(x_data):
                    self.lines[i].set_data(x_data, self.y_data[i])
                    
            self.ax.set_xlim(0, max(500, len(self.y_data[0])))
            self.canvas.draw_idle()  
        
        if self.thread_running:
            self.root.after(20, self.update_plot)

    def on_closing(self):
        """ 視窗關閉時的清理動作 """
        self.is_recording = False
        self.thread_running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.root.destroy()
        print("🔌 程式已安全關閉。")

if __name__ == "__main__":
    root = tk.Tk()
    app = EMGGUIApp(root)
    root.mainloop()