---
title: splitlnt
date: 2023-07-15 21:05:02
tags: boom boom fly
categories: 项目
diytitle:
  enable: true
  leaveTitle: w(ﾟДﾟ)w 不要走！再看看嘛！
  backTitle: ♪(^∇^*)欢迎肥来！
cover: https://pic1.zhimg.com/80/v2-fa7677b3604d0de67ad6f172f18da15c_1440w.webp
---

# 说明
获取到数据高八位第八位  
特别感谢张哥的代码支持，**写的肥肠好**  

# 代码

```python
def get_high_low_data(data):
    if data == 0:
        return (0, 0)
    temp_h = data >> 8
    temp_l = data - (temp_h << 8)
    return (temp_h, temp_l)
```