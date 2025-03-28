####
可以借鉴写法pyqt6, 有个很多没有必要的函数没有引进， 没有与下位机通信的方法
### 将python程序打包为exe文件
  - 使用流程：
    - 第一步
      - python compile.py build_ext --inplace
      - 上述执行完之后，会生成与.py文件同名的.pyd文件
    - 第二步
      - pyinstaller .\main.spec
      - 即可在目录下生成build以及dist文件夹， exe文件在dist文件夹下

### Change Log 
  v1.4.0.0:
    - 修复的已知bug
    - 将单纯pyinstaller打包 改为更为安全的方法：
      - 将py文件通过cython转换为二进制文件， 然后在的进行pyinstaller， 这样可以防止反编译
    - 修改涉及Qt函数的传参方法， 使其更加鲁棒
    - 修改main.py文件， 使其不包含任何源代码，提高打包之后的安全性
