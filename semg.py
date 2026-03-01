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
import glob
import datetime

class EMGGUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG 訊號收集工具")
        
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
        self.y_data = []
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

    def setup_ui(self):
        """ 建立上方控制面板 """
        control_frame = tk.Frame(self.root, pady=10, padx=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # 發音文字標籤與輸入框 (放大字體)
        tk.Label(control_frame, text="發音文字 (Label):", font=("Arial", 14)).pack(side=tk.LEFT, padx=(0, 5))
        self.entry_label = tk.Entry(control_frame, font=("Arial", 14), width=15)
        self.entry_label.pack(side=tk.LEFT, padx=5)
        
        # 開始按鈕 (移除 bg/fg 避免 Mac 顯示異常，並加大字體與寬度)
        self.btn_start = tk.Button(control_frame, text="▶ 開始錄製", font=("Arial", 14, "bold"), width=12, command=self.start_recording)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        # 停止按鈕 (移除 bg/fg 避免 Mac 顯示異常，並加大字體與寬度)
        self.btn_stop = tk.Button(control_frame, text="■ 停止並存檔", font=("Arial", 14, "bold"), width=12, state=tk.DISABLED, command=self.stop_recording)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        # 狀態顯示文字 (移除強制灰色，讓其適應系統深色模式，並放大)
        self.status_var = tk.StringVar()
        self.status_var.set("狀態: 待機中，請輸入發音文字並按下開始。")
        self.lbl_status = tk.Label(control_frame, textvariable=self.status_var, font=("Arial", 12))
        self.lbl_status.pack(side=tk.LEFT, padx=20)

    def setup_plot(self):
        """ 建立 Matplotlib 圖表並嵌入 Tkinter """
        # Mac Retina 解析度設定
        plt.rcParams['figure.dpi'] = 100 
        
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.line, = self.ax.plot([], [], lw=1.5, color='#00ff00') 
        
        # 樣式設定
        self.ax.set_facecolor('black') 
        self.fig.patch.set_facecolor('#222222') 
        self.ax.set_ylim(0, 4200)   
        self.ax.set_xlim(0, 500)    
        self.ax.set_title("Real-time sEMG Signal", color='white')
        self.ax.set_xlabel("Time (Samples)", color='white')
        self.ax.set_ylabel("ADC Value", color='white')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.grid(True, color='#444444', linestyle='--')

        # 將圖表嵌入 Tkinter Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def find_mac_port(self):
        """ 嘗試自動尋找 STM32 的 Port """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # 尋找名稱中包含 usbmodem 的設備
            if 'usbmodem' in port.device:
                return port.device
        return '/dev/tty.usbmodem1103' # 預設值

    def connect_serial(self):
        """ 連接 Serial Port """
        try:
            port_name = self.find_mac_port()
            self.ser = serial.Serial(port_name, self.baud_rate, timeout=1)
            self.ser.reset_input_buffer()
            return True
        except Exception as e:
            messagebox.showerror("連線失敗", f"無法連接 Serial Port。\n錯誤訊息: {e}\n\n請檢查硬體是否插好。")
            return False

    def start_recording(self):
        """ 開始錄製按鈕事件 """
        # 1. 檢查是否有輸入文字
        label_text = self.entry_label.get().strip()
        if not label_text:
            messagebox.showwarning("警告", "請先輸入「發音文字」！")
            return

        # 2. 嘗試連線
        if not self.ser or not self.ser.is_open:
            success = self.connect_serial()
            if not success:
                return

        # 3. 初始化變數與檔名
        self.current_label = label_text
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # 產生檔名：發音文字_時間_emg_data.csv
        safe_label = "".join(c for c in label_text if c.isalnum() or c in (' ', '_', '-')).rstrip()
        self.current_filename = f"{safe_label}_{timestamp_str}_emg_data.csv"
        
        self.all_data_log.clear()
        self.y_data.clear()
        while not self.data_queue.empty():
            self.data_queue.get()
            
        # 清空圖表舊線條
        self.line.set_data([], [])
        self.canvas.draw_idle()

        # 4. 更新介面狀態
        self.ax.set_title(f"Real-time sEMG Signal - Label: {self.current_label}", color='white')
        self.entry_label.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set(f"狀態: 🔴 錄製中... (標籤: {self.current_label})")
        
        # 5. 開始收集
        if self.ser:
            self.ser.reset_input_buffer()
        self.is_recording = True

    def stop_recording(self):
        """ 停止錄製按鈕事件 """
        # 1. 停止收集
        self.is_recording = False
        
        # 2. 儲存檔案
        if self.all_data_log:
            try:
                with open(self.current_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'sensor_value', 'label'])
                    writer.writerows(self.all_data_log)
                messagebox.showinfo("存檔成功", f"成功儲存 {len(self.all_data_log)} 筆數據！\n\n檔名: {self.current_filename}")
            except Exception as e:
                messagebox.showerror("存檔失敗", f"存檔時發生錯誤: {e}")
        else:
            messagebox.showwarning("警告", "沒有收集到任何數據，未產生檔案。")

        # 3. 恢復介面狀態
        self.entry_label.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_var.set("狀態: 🟢 已停止，準備好進行下一次錄製。")
        self.ax.set_title("Real-time sEMG Signal", color='white')
        self.canvas.draw_idle()

    def read_serial_task(self):
        """ 背景執行緒：負責讀取 Serial 數據 """
        while self.thread_running:
            if self.ser and self.ser.is_open:
                try:
                    # 💡 核心修正點：
                    # 不管是否在「錄製中」，都要持續把接收到的資料讀出來，
                    # 避免 Mac 的 USB Buffer 塞滿導致 STM32 硬體端 USB 卡死當機。
                    if self.ser.in_waiting > 0:
                        raw_line = self.ser.readline()
                        
                        # 只有當按下開始錄製時，才把資料存起來並畫圖
                        if self.is_recording:
                            line = raw_line.decode('utf-8', errors='ignore').strip()
                            if line.isdigit():
                                val = int(line)
                                timestamp = time.time()
                                self.data_queue.put(val)
                                self.all_data_log.append([timestamp, val, self.current_label])
                    else:
                        # 避免 while 迴圈空轉吃光 CPU
                        time.sleep(0.001)
                except Exception as e:
                    print(f"讀取錯誤: {e}")
                    time.sleep(0.1)
            else:
                # 沒連線時稍微休息
                time.sleep(0.05)

    def update_plot(self):
        """ Tkinter 主迴圈的定時更新函式，負責重繪圖表 """
        if self.is_recording:
            updated = False
            # 將 Queue 裡的新資料取出來放到 y_data
            while not self.data_queue.empty():
                val = self.data_queue.get()
                self.y_data.append(val)
                if len(self.y_data) > 500:
                    self.y_data.pop(0)
                updated = True
            
            # 如果有新資料，更新圖表線條
            if updated:
                x_data = range(len(self.y_data))
                self.line.set_data(x_data, self.y_data)
                self.ax.set_xlim(0, max(500, len(self.y_data)))
                self.canvas.draw_idle()  # 優化重繪效能
        
        # 設定 20 毫秒後再次呼叫自己 (約 50 FPS)
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