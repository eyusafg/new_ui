from distutils.core import setup
from Cython.Build import cythonize
import os 

file_list = [file for file in os.listdir() if file.endswith('.py') and 'main' not in file and 'compile' not in file and '1' in file ]
if file_list:
    for file in file_list:
        new_file_name = file.replace('1', '')
        os.rename(file, new_file_name)
file_list = [file for file in os.listdir() if file.endswith('.py') and 'main' not in file and 'compile' not in file]

setup(
    name='cython_compile_test',
    ext_modules=cythonize(
        file_list,
        language_level=3
    ),
)

pyd_name = '.cp38-win_amd64.pyd'
# 编译成功后删除源.py文件和生成的.c文件
for file in file_list:
    # 删除.py文件
    # os.remove(file)
    # os.rename(file, file.replace('.py', '1.py'))
    # print(f"Deleted: {file}")
    
    # 删除对应的.c文件
    c_file = file.replace('.py', '.c')
    if os.path.exists(c_file):
        os.remove(c_file)
        print(f"Deleted: {c_file}")

    pyd_file = file.replace('.py', '.pyd')
    if os.path.exists(pyd_file):
        os.remove(pyd_file)
        print(f"Deleted: {pyd_file}")

    pyd_name_ = c_file.replace('.c', pyd_name)
    print(f"pyd_name_ : {pyd_name_}")
    if os.path.exists(pyd_name_):
        name_ = c_file.replace('.c', '.pyd')
        os.rename(pyd_name_, name_)
        
# // 编译
# python compile.py build_ext --inplace


