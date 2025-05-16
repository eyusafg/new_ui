import os
import sys
import time
import json
import platform
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
# import requests
from urllib.parse import urlencode
import random
from selenium.webdriver import ActionChains
from write_xl import DailyTrafficWriter

# 反检测JavaScript代码
js = """
Object.defineProperties(navigator, {webdriver:{get:()=>undefined}});
window.navigator.chrome = {runtime: {}, etcetera: 'etc'};
"""

class MyChrome(webdriver.Chrome):
    """自定义浏览器驱动，集成反检测功能"""
    def __init__(self, *args, **kwargs):
        self.history = {"index": 0, "handle": None}
        self.output_parameters = {}
        
        # 初始化父类
        super().__init__(*args, **kwargs)
        
        # 执行反检测脚本
        self._apply_anti_detection()

                
    def _apply_anti_detection(self):
        """应用反检测措施"""
        self.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': js})
        self.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"""
        })
        
    def open_page(self, url, cookies=None, max_wait=10):
        """
        安全打开网页的完整流程
        :param url: 要加载的URL
        :param cookies: 需要设置的cookies列表（格式：[{"name": "x", "value": "y"}, ...]）
        :param max_wait: 最大等待时间（秒）
        """
        try:
            # 处理多窗口
            if len(self.window_handles) > 1:
                self.switch_to.window(self.window_handles[-1])
                self.close()
            self.switch_to.window(self.window_handles[0])
            
            # 设置超时时间
            self.set_page_load_timeout(max_wait)
            self.set_script_timeout(max_wait)
            
            # 加载页面
            self.get(url)
            self.print_log(f'成功加载页面: {url}')

            self.click_certificate_login()

            
        except TimeoutException:
            self.print_log(f'页面加载超时: {url}')
            self.execute_script('window.stop()')
        except Exception as e:
            self.print_log(f'加载页面失败: {str(e)}')

    def click_certificate_login(self, timeout=15):
        """使用提供的XPath点击证书登录"""

        # 1. 点击证书登录
        a = random.randint(1, 3)
        print(a)
        time.sleep(a)
        try:
            cert_login_btn = WebDriverWait(self, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//div[@class="login-container"]//li[contains(text(),"证书登录")]'))
            )
            cert_login_btn.click()
        except Exception as e:
            print("证书登录按钮未找到:", str(e))

        a = random.randint(1, 3)
        time.sleep(a)

        # 2. 输入密码（等待密码框加载）
        try:
            password_field = WebDriverWait(self, 10).until(
                EC.visibility_of_element_located((By.ID, "keyPassword"))
            )
            password_field.send_keys("12345678")
        except Exception as e:
            print("密码框操作失败:", str(e))
            raise
        a = random.randint(1, 3)
        time.sleep(a)
        try:
            # 3. 点击登录按钮
            login_btn = WebDriverWait(self, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="cersubmit"]'))
            )
            login_btn.click()
            self.execute_script("arguments[0].style.border='3px solid red';", login_btn)
            ## 在周围夹一个红色边框， 边框宽度为3像素
        except Exception as e:
            self.print_log(f'点击失败: {str(e)}')
            raise

        # self.d = {
        #     "username": "13812345678",'authorization': self.execute_script("return localStorage.getItem('token')"),
        #     'cookie': '; '.join([f"{c['name']}={c['value']}" for c in self.get_cookies()])}

        main_window = self.current_window_handle  # 记录原窗口句柄
        a = random.randint(1, 3)
        time.sleep(a)
        # 4. 等待页面加载完成
        try:
            open_mall_btn = WebDriverWait(self, 20).until(
                EC.element_to_be_clickable((By.XPATH, '//*[@id="openMall"]'))
            )
            open_mall_btn.click()
            self.print_log('进入国铁商城')
        except Exception as e:
            self.print_log(f'进入国铁商城: {str(e)}')
            raise
        a = random.randint(1, 3)
        time.sleep(a)
        WebDriverWait(self, 10).until(lambda d: len(d.window_handles) == 2)
        # 遍历句柄找到新窗口
        new_window = [h for h in self.window_handles if h != main_window][0]
        self.switch_to.window(new_window)  # 关键切换操作
        self.maximize_window()
        # a = random.randint(1, 3)
        time.sleep(2)
        try:
            mall_btn = WebDriverWait(self, 10).until(
                EC.element_to_be_clickable((By.XPATH, '/html/body/div[3]/div[1]/ul/p/a[3]'))
            )
            mall_btn.click()
            self.print_log('批量采购专区')
        except Exception as e:
            self.print_log(f'点击失败: {str(e)}')
            raise
        time.sleep(3)
        more_btn = WebDriverWait(self, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="projectAnnBtn"]'))
        )
        more_btn.click()
        self.print_log('点击更多')
        time.sleep(5)

        time_t_btn = WebDriverWait(self, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="orderTimeStartAll"]'))
        )
        time_t_btn.click()
        self.print_log('点击日期')
        time.sleep(3)

        # 清空原有内容
        time_t_btn.clear()
        # 输入日期，比如 '2024-05-01'
        time_str = '2025-05-01 00:00:00 ~ 2025-05-31 00:00:00'
        time_t_btn.send_keys(time_str)
        #
        find_t_btn = WebDriverWait(self, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="btnSearch"]'))
        )
        find_t_btn.click()
        '''
        直接通过网页 + uuid 进行采集
        '''

        self.check_and_close_popup()

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
            
    def print_log(self, message):
        """日志记录"""
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}')

def init_browser(config):
    """初始化浏览器实例"""
    options = webdriver.ChromeOptions()
    driver_path = "chromedriver.exe"

    # Windows平台特殊配置
    if sys.platform == "win32" and platform.architecture()[0] == "64bit":
        base_path = r'D:\EasySpider_Windows_x64\EasySpider\resources\app'
        options.binary_location = os.path.join(base_path, "chrome_win64/chrome.exe")
        driver_path = os.path.join(base_path, "chrome_win64/chromedriver_win64.exe")
        
        # 加载扩展程序
        extension_path = os.path.join(base_path, "XPathHelper.crx")
        if os.path.exists(extension_path):
            options.add_extension(extension_path)

    # 通用反检测配置
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-web-security")
    options.add_argument("--disable-features=CrossSiteDocumentBlockingIfIsolating")
    options.add_argument('-ignore-certificate-errors')
    # 关键配置：启用性能日志
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    options.add_argument('--enable-logging')  # 启用基础日志
    options.add_argument('--v=1')  # 设置详细日志级别

    # 无头模式
    if config.get('headless'):
        options.add_argument("--headless=new")

    # 初始化服务
    service = Service(executable_path=driver_path)
    
    return MyChrome(service=service, options=options)

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
    import openpyxl

    config = {
        "headless": False,
        "max_wait": 15,
        "cookies": [
            {"name": "test_cookie", "value": "12345"}
        ]
    }
    
    try:
        # 初始化浏览器
        browser = init_browser(config)
        
        # 打开页面示例
        browser.open_page(
            url="https://cg.95306.cn/passport/verifyLogin?returnUrl=https://cg.95306.cn/",
            cookies=config["cookies"],
            max_wait=config["max_wait"]
        )

        # 模拟滚动操作
        # browser.scroll_down(times=2)
        # https://mall.95306.cn/mall-view/negotiationArea/details-project-ann?uuid=d01713470f0b4a1cafe4e5e65b3c5f6d
        detail_url_prefix = "https://mall.95306.cn/mall-view/negotiationArea/details-project-ann?uuid="
        all_data = browser.batch_collect(detail_url_prefix)

        now = datetime.now()  # current date and time
        # year = now.strftime("%Y")
        time_str = str(now.strftime("%Y_%m_%d"))
        xlsx_name = f"{time_str}.xlsx"
        save_to_excel(all_data, xlsx_name)
        # 保持浏览器打开
        input("按回车键退出...")
        
    except Exception as e:
        print(f"操作失败: {str(e)}")
    finally:
        if 'browser' in locals():
            browser.quit()
