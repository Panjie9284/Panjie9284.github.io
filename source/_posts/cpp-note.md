---
title: cpp_note
date: 2023-09-20 21:17:16
categories: note
diytitle:
  enable: true
  leaveTitle: w(ﾟДﾟ)w 不要走！再看看嘛！
  backTitle: ♪(^∇^*)欢迎肥来！
cover: https://pic3.zhimg.com/80/v2-e3dab84260c5618458cbc0b9abf4e612_720w.webp
---

# Hello World

## ~~写个hello world就算会cpp了，成为语言大师~~ 
## 会写helllo world 是算刚入门
``` cpp
#include <iostream>

using namespace std;

int main()
{
    system("pause");

    cout << "Hello world!" << endl;

    return 0;

}

```

## 注释
```cpp
//我是单行的注释

/*
我
是
多
行
注
释
*/

```

## 变量
变量存在的意义：方便管理内存空间

数据类型 变量名 = 初始值;
```cpp
int a = 10;
```

## 常量
**作用**：不可改变的数据
```cpp
//第一种
#define PI 3.14
//第二种
const a = 10;
```

## 关键字

**作用**：关键字是cpp中预先保留的单词（标识符）

* 在定义变量或常量时不要使用关键字

## 标识符命名规则

* 不能是关键字  
* 只能由字母、数字、下划线组成  
* 第一个字符必须为字母或者下划线  
* 标识符中区分大小写 
* 使用英文规范化命名  

# 数据类型

## 整型
**作用**：表示整数类型的数据  

short(2字节)、int(4字节)、long(4字节)、long long(8字节)

## sizeof关键字
**作用**：可以统计数据类型所占内存大小  

**语法**：sizeof(a);

##  实型（浮点型）

**作用**：用于表示小数


**语法**：  
1. 单精度float  4字节  7位有效数字   

2. 双精度double  8字节  15~16位有效数字

##  字符型

**作用**：用于显示单个字符

**语法** char c = 'a';

##  字符型

**作用**：用于显示单个字符

**语法** char c = 'a';

## 转义字符

**作用**：用于表示一些不能显示出来的ASCLL字符

**语法**：入门需掌握 \n换行 \\输出反斜杠 \t制表 

## 字符串型

**作用**：用于表示一串字符  
**语法**：两种风格

1. c风格：
```cpp
char str1[] = "hello world!";
cout << str1 << endl;

```

2. cpp风格：
```cpp
string str2 = "hello world!";
cout << str2 << endl;
// 需要包含头文件#include<string>
```

## bool 类型

**作用**：布尔数据类型代表真或假  

所占空间为1  

## 数据的输入

**作用**：用于键盘获取数据

**关键字**：cin

**语法**：cin >> 变量;

```cpp
cin >> a;
scanf(%d,&a);
//cin虽然比较简洁，但是效率不如scanf
```

## 运算符

**作用**：用于执行代码的运算






















