import os
import sys
import time
import random
import platform
import threading
from datetime import datetime
from tkinter import Tk, ttk, StringVar, messagebox, Text, END, DISABLED
from tkcalendar import DateEntry
import openpyxl
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import subprocess
import requests
import winreg
from pathlib import Path
from webdriver_manager.core.driver_cache import DriverCacheManager
from webdriver_manager.chrome import ChromeDriverManager



class ChromeAutoInstaller:
    """Chrome浏览器与驱动自动安装器"""
    
    def __init__(self):
        self.system = platform.system()
        self.arch = platform.machine()
        self.chrome_path = None
        self.driver_path = None

    def _get_windows_chrome_path(self):
        """Windows系统获取Chrome安装路径"""
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe"
            )
            path = winreg.QueryValue(key, None)
            winreg.CloseKey(key)
            return path
        except Exception:
            return None

    def _install_windows_chrome(self):
        """Windows安装Chrome"""
        print("正在自动安装Chrome浏览器...")
        url = "https://dl.google.com/tag/s/dl/chrome/install/standalone/GoogleChromeStandaloneEnterprise64.msi"
        installer_path = "chrome_installer.msi"
        
        try:
            # 下载安装包
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(installer_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # 静默安装
            subprocess.run(
                ["msiexec", "/i", installer_path, "/qn"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            os.remove(installer_path)
            return True
        except Exception as e:
            print(f"安装失败: {str(e)}")
            return False

    def _find_chrome(self):
        """查找已安装的Chrome路径"""
        if self.system == "Windows":
            paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            ]
            reg_path = self._get_windows_chrome_path()
            if reg_path and os.path.exists(reg_path):
                return reg_path
            for path in paths:
                if os.path.exists(path):
                    return path
        elif self.system == "Darwin":
            mac_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
            return mac_path if os.path.exists(mac_path) else None
        elif self.system == "Linux":
            return "/usr/bin/google-chrome" if os.path.exists("/usr/bin/google-chrome") else None
        return None

    def ensure_chrome(self):
        """确保Chrome已安装"""
        self.chrome_path = self._find_chrome()
        if self.chrome_path:
            print(f"检测到Chrome路径: {self.chrome_path}")
            return True

        print("未找到Chrome浏览器，开始自动安装...")
        if self.system == "Windows":
            success = self._install_windows_chrome()
        elif self.system == "Darwin":
            raise NotImplementedError("Mac自动安装暂未实现")
        elif self.system == "Linux":
            raise NotImplementedError("Linux自动安装暂未实现")
        else:
            raise OSError("不支持的操作系统")

        if success:
            self.chrome_path = self._find_chrome()
            return self.chrome_path is not None
        return False

    def setup_driver(self):
        """配置ChromeDriver"""
        try:
            # 处理打包路径
            if getattr(sys, 'frozen', False):
                base_dir = sys._MEIPASS
            else:
                base_dir = os.path.dirname(__file__)
            
            driver_dir = os.path.join(base_dir, "driver")
            os.makedirs(driver_dir, exist_ok=True)
            
            # 创建自定义缓存管理器
            cache_manager = DriverCacheManager(root_dir=driver_dir)
            
            # 自动下载驱动到指定目录
            self.driver_path = ChromeDriverManager(
                cache_manager=cache_manager
            ).install()
            
            return True
        except Exception as e:
            print(f"驱动配置失败: {str(e)}")
            return False

class MyChrome(Chrome):
    """自定义浏览器驱动，集成反检测功能"""
    def __init__(self):

        # 自动安装检测
        self.installer = ChromeAutoInstaller()
        if not self.installer.ensure_chrome():
            raise RuntimeError("Chrome浏览器安装失败")
        if not self.installer.setup_driver():
            raise RuntimeError("Chrome驱动配置失败")
        
        # 配置浏览器选项
        options = webdriver.ChromeOptions()
        options.binary_location = self.installer.chrome_path

        # 添加反检测选项
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        # 初始化父类
        service = Service(executable_path=self.installer.driver_path)
        super().__init__(service=service, options=options)

        self._apply_anti_detection()

        # self.time_str = time_str  # 新增时间参数存储
        self.history = {"index": 0, "handle": None}
        self.output_parameters = {}

    def _apply_anti_detection(self):
        """应用反检测措施"""

        # 反检测JavaScript代码
        js = """
        Object.defineProperties(navigator, {webdriver:{get:()=>undefined}});
        window.navigator.chrome = {runtime: {}, etcetera: 'etc'};
        """
        self.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': js})
        self.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"""
        })
    
    def random_sleep(self, min_sec=1, max_sec=3):
        """随机等待时间"""
        delay = random.randint(min_sec, max_sec)
        time.sleep(delay)
    
    def print_log(self, message):
        """日志记录"""
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}')

    def open_page(self, url, cookies=None, max_wait=10):
        """安全打开网页的完整流程"""
        try:
            if len(self.window_handles) > 1:
                self.switch_to.window(self.window_handles[-1])
                self.close()
            self.switch_to.window(self.window_handles[0])
            
            self.set_page_load_timeout(max_wait)
            self.set_script_timeout(max_wait)
            self.get(url)
            self.print_log(f'成功加载页面: {url}')
            self.click_certificate_login()
            
        except TimeoutException:
            self.print_log(f'页面加载超时: {url}')
            self.execute_script('window.stop()')
        except Exception as e:
            self.print_log(f'加载页面失败: {str(e)}')

    def click_certificate_login(self, time_str=None, timeout=15):
        """证书登录流程（含日期参数）"""
        try:
            # 证书登录按钮
            self.random_sleep()
            cert_login_btn = WebDriverWait(self, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//div[@class="login-container"]//li[contains(text(),"证书登录")]'))
            )
            cert_login_btn.click()
            self.print_log(f'点击证书登录按钮')
            
            # 密码输入
            self.random_sleep()
            password_field = WebDriverWait(self, 10).until(
                EC.visibility_of_element_located((By.ID, "keyPassword"))
            )
            password_field.send_keys("12345678")
            self.print_log(f'输入密码')
            
            # 登录按钮
            self.random_sleep()
            login_btn = WebDriverWait(self, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="cersubmit"]'))
            )
            login_btn.click()
            self.print_log(f'点击登录按钮')
            
            # 处理窗口切换
            main_window = self.current_window_handle
            self.random_sleep()
            WebDriverWait(self, 20).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="openMall"]'))
            ).click()
            self.print_log(f'进入国铁商城')
            
            # 进入新窗口
            WebDriverWait(self, 10).until(lambda d: len(d.window_handles) == 2)
            self.switch_to.window([h for h in self.window_handles if h != main_window][0])
            self.maximize_window()
            
            # 导航到目标页面
            self.random_sleep(2,3)
            WebDriverWait(self, 10).until(
                EC.element_to_be_clickable((By.XPATH, '/html/body/div[3]/div[1]/ul/p/a[3]'))
            ).click()
            self.print_log(f'进入批量采购专区')
            
            # 日期输入处理
            self.random_sleep()
            WebDriverWait(self, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="projectAnnBtn"]'))
            ).click()
            self.print_log(f'点击更多')
            
            time_t_btn = WebDriverWait(self, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="orderTimeStartAll"]'))
            )
            time_t_btn.click()
            time_t_btn.clear()
            self.print_log(f'点击日期')
            
            # 使用传入的时间参数
            final_time_str = time_str if time_str else '2025-05-01 00:00:00 ~ 2025-05-31 00:00:00'
            time_t_btn.send_keys(final_time_str)
            
            WebDriverWait(self, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="btnSearch"]'))
            ).click()
            self.print_log(f'点击搜索')
            
            self.check_and_close_popup()

        except Exception as e:
            self.print_log(f'操作失败: {str(e)}')
            raise

    def get_all_uuids_on_page(self):
        """采集当前页所有li.goProDetails的data-uuid"""
        uuid_list = []
        lis = self.find_elements(By.CSS_SELECTOR, 'li.goProDetails')
        for li in lis:
            uuid = li.get_attribute('data-uuid')
            if uuid:
                uuid_list.append(uuid)
        return uuid_list

    def get_detail_info_old(self, detail_url):
        """进入详情页采集信息"""
        self.get(detail_url)
        time.sleep(1)  # 等待详情页加载
        # 项目联系信息
        try:
            contact_info = self.find_element(By.XPATH,
                                             '//p[contains(text(),"项目联系信息")]/following-sibling::div[1]').text
        except:
            contact_info = ""
        # 收货人信息
        try:
            harvest_info = self.find_element(By.XPATH,
                                             '//p[contains(text(),"收货人信息")]/following-sibling::div[1]').text
        except:
            harvest_info = ""
        # 收货地址
        # try:
        #     harvest_address = self.find_element(By.XPATH, '//span[@id="harvestAddressId"]').text
        # except:
        #     harvest_address = ""
        return {
            "contact_info": contact_info,
            "harvest_info": harvest_info,
            # "harvest_address": harvest_address,
            "url": detail_url
        }

    def get_detail_info(self, detail_url):
        self.get(detail_url)
        time.sleep(1)
        # self.check_and_close_popup()
        # 项目联系信息
        try:
            contact_info_div = WebDriverWait(self, 5).until(
                EC.presence_of_element_located((By.XPATH,
                                                '//p[contains(text(),"项目联系信息") or contains(text(),"项目联系人")]/following-sibling::div[1]'))
            )
            contact_info = contact_info_div.text
        except Exception as e:
            contact_info = ""
            print(f"未采集到项目联系信息: {e}")
        # 收货人信息
        try:
            harvest_info_div = WebDriverWait(self, 5).until(
                EC.presence_of_element_located((By.XPATH,
                                                '//p[contains(text(),"收货人信息") or contains(text(),"收货人")]/following-sibling::div[1]'))
            )
            harvest_info = harvest_info_div.text
        except Exception as e:
            harvest_info = ""
            print(f"未采集到收货人信息: {e}")
        return {
            "contact_info": contact_info,
            "harvest_info": harvest_info,
            "url": detail_url
        }


    def go_to_next_page(self):
        """点击下一页按钮"""
        try:
            next_btn = self.find_element(By.CSS_SELECTOR, 'a.layui-laypage-next')
            # 判断是否为最后一页
            if "layui-disabled" in next_btn.get_attribute("class"):
                return False
            next_btn.click()
            time.sleep(1)  # 等待页面加载
            return True
        except Exception as e:
            print("未找到下一页按钮，或已到最后一页", e)
            return False

    def batch_collect(self, detail_url_prefix):
        """分页采集主流程"""
        all_data = []
        while True:
            uuid_list = self.get_all_uuids_on_page()
            print(f"本页采集到{len(uuid_list)}个uuid")
            for uuid in uuid_list:
                detail_url = f"{detail_url_prefix}{uuid}"  # 这里拼接你的详情页URL
                info = self.get_detail_info(detail_url)
                print(info)
                all_data.append(info)
                # 采集完详情页后返回列表页
                self.back()
                time.sleep(1)
            if not self.go_to_next_page():
                break
            # if len(uuid_list) == 24:
            #     break
        print("全部采集完毕，共采集：", len(all_data), "条")
        return all_data

    def check_and_close_popup(driver, timeout=15):
        """
        智能检测layui弹窗并关闭
        :param driver: 浏览器驱动实例
        :param timeout: 最大等待时间(秒)
        """
        popup_xpath = '//*[@id="layui-layer2"]/div/div[1]/p[1]'
        close_btn_xpath = '//*[@id="layui-layer2"]/div/div[3]/button'  
        
        try:
            # 显式等待弹窗内容可见
            WebDriverWait(driver, timeout).until(
                EC.visibility_of_element_located((By.XPATH, popup_xpath))
            )
            print("检测到系统弹窗，正在关闭...")
            
            # 点击关闭按钮（优先使用具体关闭按钮定位）
            driver.find_element(By.XPATH, close_btn_xpath).click()
            
            # 可选：等待弹窗关闭
            WebDriverWait(driver, 3).until(
                EC.invisibility_of_element_located((By.XPATH, popup_xpath))
            )
        except TimeoutException:
            print("未检测到弹窗，继续执行操作")

class ScraperApp:
    """GUI应用程序主类"""
    def __init__(self, root):
        self.root = root
        self.running = False
        self.browser = None
        self._setup_ui()
        
    def _setup_ui(self):
        """初始化界面组件"""
        self.root.title("国铁商城数据采集系统 v1.0")
        self.root.geometry("680x520")
        
        # 日期选择框架
        date_frame = ttk.LabelFrame(self.root, text="选择采集日期范围")
        date_frame.pack(pady=10, padx=15, fill="x")
        
        ttk.Label(date_frame, text="开始日期:").grid(row=0, column=0, padx=5, pady=5)
        self.start_date = DateEntry(date_frame, date_pattern="yyyy-mm-dd", width=12)
        self.start_date.grid(row=0, column=1, padx=5)
        
        ttk.Label(date_frame, text="结束日期:").grid(row=1, column=0, padx=5, pady=5)
        self.end_date = DateEntry(date_frame, date_pattern="yyyy-mm-dd", width=12)
        self.end_date.grid(row=1, column=1, padx=5)
        
        # 控制按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        self.start_btn = ttk.Button(btn_frame, text="开始采集", command=self.toggle_scraper)
        self.start_btn.pack(side="left", padx=5)
        ttk.Button(btn_frame, text="退出系统", command=self.quit_app).pack(side="left")
        
        # 日志输出
        log_frame = ttk.LabelFrame(self.root, text="运行日志")
        log_frame.pack(padx=15, pady=5, fill="both", expand=True)
        self.log_area = Text(log_frame, height=12, state=DISABLED, wrap="word")
        scroll = ttk.Scrollbar(log_frame, command=self.log_area.yview)
        self.log_area.configure(yscrollcommand=scroll.set)
        self.log_area.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        
    def toggle_scraper(self):
        """启动/停止采集"""
        if self.running:
            self.stop_scraper()
        else:
            if self._validate_dates():
                self.start_scraper()
                threading.Thread(target=self.run_scraper, daemon=True).start()
                
    def start_scraper(self):
        self.running = True
        self.start_btn.config(text="停止采集")
        self.log_area.config(state="normal")
        self.log_area.delete(1.0, END)
        self.log_area.config(state=DISABLED)
        
    def stop_scraper(self):
        self.running = False
        self.start_btn.config(text="开始采集")
        if self.browser:
            try:
                self.browser.quit()
            except:
                pass
            
    def quit_app(self):
        self.stop_scraper()
        self.root.quit()
    
    def _validate_dates(self):
        start = self.start_date.get_date()
        end = self.end_date.get_date()
        if start > end:
            messagebox.showerror("日期错误", "结束日期不能早于开始日期！")
            return False
        return True
    
    def _get_time_str(self):
        start = self.start_date.get_date().strftime("%Y-%m-%d") + " 00:00:00"
        end = self.end_date.get_date().strftime("%Y-%m-%d") + " 00:00:00"
        return f"{start} ~ {end}"
    
    def log(self, message):
        self.log_area.config(state="normal")
        self.log_area.insert(END, f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        self.log_area.see(END)
        self.log_area.config(state=DISABLED)
        print(message)  # 同时输出到控制台
    
    def run_scraper(self):
        """执行采集任务主逻辑"""
        try:
            self.log("正在初始化浏览器...")

            # 自动安装检测
            # installer = ChromeAutoInstaller()
            # if not installer.ensure_chrome():
            #     messagebox.showerror("错误", "Chrome浏览器安装失败，请手动安装")
            #     return
            # if not installer.setup_driver():
            #     messagebox.showerror("错误", "驱动安装失败，请检查网络连接")
            #     return
            
            ime_str = self._get_time_str()
            self.browser = MyChrome()
            
            self.log("启动浏览器成功，开始数据采集...")
            time_str = self._get_time_str()
            
            # 执行登录和搜索流程
            self.browser.get("https://cg.95306.cn/passport/verifyLogin")
            self.browser.click_certificate_login(time_str=time_str)
            
            # 执行数据采集
            detail_url_prefix = "https://mall.95306.cn/mall-view/negotiationArea/details-project-ann?uuid="
            all_data = self.browser.batch_collect(detail_url_prefix)
            
            # 保存结果
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"采购数据_{time_str}.xlsx"
            save_to_excel(all_data, filename)
            self.log(f"数据采集完成，已保存到：{filename}")
            messagebox.showinfo("完成", f"数据采集完成！\n保存文件：{filename}")
            
        except Exception as e:
            self.log(f"发生错误：{str(e)}")
            messagebox.showerror("错误", f"采集过程中发生错误：\n{str(e)}")
        finally:
            self.stop_scraper()


def save_to_excel(data_list, filename="result.xlsx"):
    """
    data_list: [{'contact_info':..., 'harvest_info':..., 'url':...}, ...]
    """
    from collections import defaultdict

    # 1. 按采购人单位分组
    unit_dict = defaultdict(list)
    for item in data_list:
        unit, contact, buy_num, address = parse_harvest_info(item['harvest_info'])
        if not unit:
            continue
        contact_info = item.get("contact_info", "")
        people, phone, yiyi_people, yiyi_phone = parse_contact_info(contact_info)
        unit_dict[unit].append({
            "采购人": contact,
            "联系人": people,
            "联系方式": phone,
            "收货地址": address,
            "项目联系人": item.get("contact_info", ""),
            # "详情页": item.get("url", "")
        })

    # 2. 写入Excel
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "采购信息"

    for unit, records in unit_dict.items():
        # 2.1 单独表头
        ws.append([unit])
        ws.append(["采购人", "联系人", "联系方式", "收货地址"])
        # 2.2 去重（联系人+联系方式）
        seen = set()
        for rec in records:
            key = (rec["联系人"], rec["联系方式"])
            if key in seen:
                continue
            seen.add(key)
            ws.append([rec["采购人"], rec["联系人"], rec["联系方式"], rec["收货地址"]])
            ws.append([])  # 间隔一行
        ws.append([])  # 单位之间再空一行

    wb.save(filename)
    print(f"已保存到 {filename}")

def parse_harvest_info(harvest_info):
    """
    从 harvest_info 字符串中提取采购人单位、联系人、联系方式、收货地址等
    """
    unit = ""
    contact = ""
    buy_num = ""
    address = ""
    for line in harvest_info.split('\n'):
        if "采购人单位" in line:
            unit = line.split("：", 1)[-1].strip()
        elif "采购人1" in line or "采购人" in line:
            contact = line.split("：", 1)[-1].strip()
        elif "采购数量" in line:
            buy_num = line.split("：", 1)[-1].strip()
        elif "收货地址" in line:
            address = line.split("：", 1)[-1].strip()
    return unit, contact, buy_num, address

def parse_contact_info(contact_info):
    """
    从 contact_info 字符串中提取采购人单位、联系人、联系方式、收货地址等
    """
    people = ""
    phone = ""
    yiyi_people = ""
    yiyi_phone = ""
    for line in contact_info.split('\n'):
        if "项目联系人" in line:
            people = line.split("：", 1)[-1].strip()
        elif "质疑异议联系方式" in line:
            yiyi_phone = line.split("：", 1)[-1].strip()
        elif "联系方式" in line:
            phone = line.split("：", 1)[-1].strip()
        elif "质疑异议联系人" in line:
            yiyi_people = line.split("：", 1)[-1].strip()

    return people, phone, yiyi_people, yiyi_phone


if __name__ == "__main__":
    root = Tk()
    app = ScraperApp(root)
    root.mainloop()
