import hmac
import hashlib
import tkinter as tk
from tkinter import messagebox



class LicenseGenerator:
    def __init__(self, master_key):
        self.master_key = master_key
        
    def generate(self, client_id, days=365):
        """生成离线激活码"""
        data = f"{client_id}|{days}".encode()
        signature = hmac.new(self.master_key, data, hashlib.sha256).hexdigest()[:16]
        return f"THOR-{client_id}-{days}D-{signature.upper()}"

class LicenseGeneratorUI:
    def __init__(self, master):
        self.master = master
        master.title("激活码生成器")

        self.master_key = hashlib.pbkdf2_hmac('sha256', b'03106666', b'salt', 100000)
        self.generator = LicenseGenerator(self.master_key)

        self.label_client_id = tk.Label(master, text="客户设备ID:")
        self.label_client_id.grid(row=0, column=0, padx=2, pady=10)
        self.entry_client_id = tk.Entry(master, width=20)
        self.entry_client_id.grid(row=0, column=1, padx=2, pady=10)

        self.label_days = tk.Label(master, text="有效期(天):")
        self.label_days.grid(row=1, column=0, padx=2, pady=10)
        self.entry_days = tk.Entry(master, width=20)
        self.entry_days.grid(row=1, column=1, padx=2, pady=10)

        # 创建生成按钮
        self.generate_button = tk.Button(master, text="生成", command=self.generate_license)
        self.generate_button.grid(row=2, column=0, columnspan=2, pady=10)

        # 显示生成的激活码
        self.label_result = tk.Label(master, text="生成的激活码:")
        self.label_result.grid(row=3, column=0, padx=2, pady=10)
        
        self.result = tk.StringVar()
        # self.result_label = tk.Label(master, textvariable=self.result)
        # self.result_label.grid(row=3, column=1, padx=10, pady=10)
        self.result_text = tk.Text(master, height=1, width=20, state='disabled')
        self.result_text.grid(row=3, column=1, padx=2, pady=10)

    def generate_license(self):
        client_id = self.entry_client_id.get()
        days = self.entry_days.get()
        if not client_id or not days.isdigit():
            messagebox.showerror("错误", "客户设备ID和有效期必须为数字！")
            return
        try:
            license_key = self.generator.generate(client_id, int(days))
            # self.result.set(license_key)
            self.result_text.config(state='normal')  
            self.result_text.delete(1.0, tk.END)  
            self.result_text.insert(tk.END, license_key)  
            self.result_text.config(state='disabled') 
        except Exception as e:
            messagebox.showerror("错误", f"生成激活码失败：{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LicenseGeneratorUI(root)
    root.mainloop()

# if __name__ == "__main__":
#     master_key = hashlib.pbkdf2_hmac('sha256', b'03106666', b'salt', 100000)
    
#     generator = LicenseGenerator(master_key)
#     client_id = input("请输入客户设备ID: ")
#     license_key = generator.generate(client_id, days=365)
    
#     print(f"生成的激活码：{license_key}")
