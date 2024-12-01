---
title: python_debug
date: 2023-07-10 20:00:06
tags: debug
categories: debug
diytitle:
  enable: true
  leaveTitle: w(ﾟДﾟ)w 不要走！再看看嘛！
  backTitle: ♪(^∇^*)欢迎肥来！
---
1. **报错AttributeError: ‘NoneType‘ object has no attribute‘shape’**  
出现的原因 摄像头接线断了  
解决办法 氪金或者把线焊好

2. **关于Python函数中变量报错UnboundLocalError: local variable referenced before assignment**  
出现的原因 定义的同名变量在函数外和里都进行了赋值操作  
解决办法 改变量名  

3. **Python: TypeError: cannot unpack non-iterable NoneType object**  
出现的原因 TypeError: cannot unpack non-iterable NoneType object这个错误的意思是；
对非元组（迭代）类型NoneType，不能做解构操作；  
比如 x,y,z = func1() ; 这里有一个隐式的解构操作；如果func1没有返回值，这里就可能会有一个NoneType类型的返回值。就会出现标题里的错误。  
解决办法 明确函数各种情况下返回什么值，不能返回空值  

4. **未完待续**