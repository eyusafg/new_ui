import pymysql
from pymysql.err import IntegrityError

class DeviceManagementDB:
    def __init__(self, host='localhost', user='root', password='password', database=None, charset='utf8mb4', filename='test_sql', table_name='devices'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset
        self.connection = None
        self.filename = filename
        self.table_name = table_name
    def __enter__(self):
        self.connection = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
            charset=self.charset
        )
        return self.connection.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connection:
            if exc_type is not None:
                self.connection.rollback()
            else:
                self.connection.commit()
            self.connection.close()
            self.connection = None

    def create_database(self):
        with self.__class__(host=self.host, user=self.user, password=self.password) as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.filename}")
        print("数据库创建成功")

    def create_table(self):
        with self as cursor:
            sql = f"""
            CREATE TABLE IF NOT EXISTS `{self.table_name}` (
                id INT AUTO_INCREMENT PRIMARY KEY,
                company_name VARCHAR(255) NOT NULL,
                uid VARCHAR(80) UNIQUE NOT NULL,
                software_version VARCHAR(80),
                hardware_version VARCHAR(80),
                serial_number VARCHAR(80),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(sql)
        print("数据表创建成功")

    def column_exists(self, column_name, cursor):
        # with self as cursor:
        try:
            sql = f"SHOW COLUMNS FROM `{self.table_name}` LIKE %s"
            cursor.execute(sql, (column_name,))
            result = cursor.fetchone()
            return result is not None
        except Exception as e:
            print(f"检查列时发生错误: {e}")
            return False
            
    def add_serial_number_column(self,cursor):
        if not self.column_exists('serial_number', cursor):
            try:
                sql = f"ALTER TABLE `{self.table_name}` ADD COLUMN serial_number VARCHAR(80) AFTER hardware_version"
                cursor.execute(sql)
                print("serial_number 列新增成功")
            except Exception as e:
                print(f"修改表结构时发生错误: {e}")

    def add_column_if_not_exists(self, column_name, data_type, after_column, cursor):
        """通用方法：如果列不存在则添加列"""
        if not self.column_exists(column_name, cursor):
            try:
                # 使用反引号包裹列名和位置，防止SQL保留字冲突
                sql = f"""
                    ALTER TABLE `{self.table_name}` 
                    ADD COLUMN `{column_name}` {data_type} 
                    AFTER `{after_column}`
                """
                cursor.execute(sql)
                print(f"列 `{column_name}` 添加成功")
            except Exception as e:
                print(f"添加列 `{column_name}` 时发生错误: {e}")
                
    def insert_device(self, company, machine_code, software_version, hardware_version,serial_number, HMI_code):
        with self as cursor:
            try:
                # self.add_serial_number_column(cursor)
                self.add_column_if_not_exists(
                    column_name='serial_number',
                    data_type='VARCHAR(80)',
                    after_column='hardware_version',
                    cursor=cursor
                )
                self.add_column_if_not_exists(
                    column_name='HMI_code',
                    data_type='VARCHAR(80)', 
                    after_column='serial_number',
                    cursor=cursor
                )
                sql = f"INSERT INTO {self.table_name} (company_name, uid,software_version, hardware_version, serial_number, HMI_code) VALUES (%s, %s, %s, %s, %s, %s)"
                cursor.execute(sql, (company, machine_code, software_version, hardware_version,serial_number, HMI_code))
                print("数据插入成功")
                return 1
            except IntegrityError:
                print("错误：机器码已存在（唯一约束冲突）")
                return -1
            except Exception as e:
                print(f"数据库错误: {e}")
                return f"数据库错误: {e}"
            
    ################################# 增加备份功能 ##########################

    ################################# 增加备份功能 ##########################
    
    ######### 还要增加查询功能 可以显示信息在界面 ##################### 
    def update_device(self, company, machine_code, software_version, hardware_version,serial_number, HMI_code):
        with self as cursor:
            try:
                sql = f"UPDATE {self.table_name} SET company_name = %s, software_version = %s, hardware_version = %s, serial_number = %s, HMI_code = %s WHERE uid = %s"
                cursor.execute(sql, (company, software_version, hardware_version, serial_number, HMI_code, machine_code))
                print("数据更新成功")
                return 1
            except Exception as e:
                print(f"数据库错误: {e}")
                return 0

    def query_devices(self):
        with self as cursor:
            sql = f"SELECT id, company_name, uid, software_version, hardware_version, serial_number, HMI_code, created_at FROM {self.table_name}"
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                # print(f"ID: {row[0]}, 公司: {row[1]}, 机器码: {row[2]}")
                print(f"ID: {row[0]}, 公司: {row[1]}, 机器码: {row[2]},  soft_version: {row[3]},  hard_version: {row[4]},  serial_number: {row[5]}, HMI_code: {row[6]}, 创建时间: {row[7]}")
            return results
        
    def query_devices_by_name(self, column_name,name):
        if self.database is None:
            raise Exception("数据库未指定，请先创建或指定数据库")

        with self as cursor:
            sql = f"SELECT id, company_name, uid, software_version, hardware_version, serial_number, HMI_code, created_at FROM {self.table_name} WHERE {column_name}=%s"
            cursor.execute(sql, (name,))
            results = cursor.fetchall()
            for row in results:
                # print(f"ID: {row[0]}, 公司: {row[1]}, 机器码: {row[2]}")
                print(f"ID: {row[0]}, 公司: {row[1]}, 机器码: {row[2]},  soft_version: {row[3]},  hard_version: {row[4]},  serial_number: {row[5]}, HMI_code: {row[6]}, 创建时间: {row[7]}")
            return results

    def delete_table(self, name):
        with self as cursor:
            sql = f"DROP TABLE IF EXISTS {name}"
            cursor.execute(sql)
            self.connection.commit()  # 提交事务
        print("数据表删除成功")


    def delete_data(self, name):
        with self as cursor:
            sql = f"DROP TABLE IF EXISTS {name}"
            cursor.execute(sql)
            self.connection.commit()  # 提交事务
        print("数据库删除成功")

if __name__ == '__main__':
    filename = 'syt_machine_data'
    table_name = '圆领'
    # db = DeviceManagementDB(host='localhost', user='syt', password='03106666', filename=filename)
    db = DeviceManagementDB(host='192.168.6.239', user='syt', password='03106666', filename=filename, table_name=table_name)
    db.create_database()
    db.database = filename  # 指定数据库
    db.create_table()
    # db.insert_device('syt', '123','2','3', '4')
    # db.backup_table()
    # db.query_devices()  
    db.query_devices_by_name('uid', '192116c9')
    # db.query_devices_by_company('syt')  # 按照指定公司名进行查询
    # db.delete_table(table_name)
    # db.delete_data(filename)
