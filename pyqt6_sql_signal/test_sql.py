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
        if exc_type is not None:
            self.connection.rollback()
        else:
            self.connection.commit()
        self.connection.close()

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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(sql)
        print("数据表创建成功")

    def insert_device(self, company, machine_code, software_version, hardware_version):
        with self as cursor:
            try:
                sql = f"INSERT INTO {self.table_name} (company_name, uid,software_version, hardware_version) VALUES (%s, %s, %s, %s)"
                cursor.execute(sql, (company, machine_code, software_version, hardware_version))
                print("数据插入成功")
                return 1
            except IntegrityError:
                print("错误：机器码已存在（唯一约束冲突）")
                return -1
            except Exception as e:
                print(f"数据库错误: {e}")
                return f"数据库错误: {e}"
            
    def update_device(self, company, machine_code, software_version, hardware_version):
        with self as cursor:
            try:
                sql = f"UPDATE {self.table_name} SET company_name = %s, software_version = %s, hardware_version = %s WHERE uid = %s"
                cursor.execute(sql, (company, software_version, hardware_version, machine_code))
                print("数据更新成功")
                return 1
            except Exception as e:
                print(f"数据库错误: {e}")
                return 0

    def query_devices(self):
        with self as cursor:
            sql = f"SELECT id, company_name, uid, software_version, hardware_version, created_at FROM {self.table_name}"
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                # print(f"ID: {row[0]}, 公司: {row[1]}, 机器码: {row[2]}")
                print(f"ID: {row[0]}, 公司: {row[1]}, 机器码: {row[2]},  soft_version: {row[3]},  hard_version: {row[4]},  创建时间: {row[5]}")

    def query_devices_by_company(self, company_name):
        if self.database is None:
            raise Exception("数据库未指定，请先创建或指定数据库")

        with self as cursor:
            sql = f"SELECT id, company_name, uid, software_version, hardware_version, created_at FROM {self.table_name} WHERE company_name=%s"
            cursor.execute(sql, (company_name,))
            results = cursor.fetchall()
            for row in results:
                # print(f"ID: {row[0]}, 公司: {row[1]}, 机器码: {row[2]}")
                print(f"ID: {row[0]}, 公司: {row[1]}, 机器码: {row[2]},  soft_version: {row[3]},  hard_version: {row[4]},  创建时间: {row[5]}")

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
    filename = 'syt_round_neck_data'
    db = DeviceManagementDB(host='localhost', user='root', password='03106666', filename=filename)
    db.create_database()
    db.database = filename  # 指定数据库
    db.create_table()
    # db.insert_device('syt', '123')
    db.query_devices()  
    # db.query_devices_by_company('syt')  # 按照指定公司名进行查询
    db.delete_table('devices')
    db.delete_data(filename)
