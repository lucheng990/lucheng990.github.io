# 从函数开始

## 逻辑判断
> True&False
- python 中对逻辑函数的判断比c++更加智能
##### 参考如下代码
```
1<2<3 #True
42!='42' #True
'M' in 'Magic' #true
{
    number=12
    number is 12   #true
}
```
##### 成员运算符和身份运算符
- in和 not in:测试前者是否存在与in后面的集合中

> **列表**是一种集合类型
    
> 可以用如下代码创建一个列表（list)
```
album=[]
album=['Black Star','David Bowie',25,True]
```
> 当列表创建完成以后，想要再次往里面添加内容，可以使用列表的append方法，并且使用这种方法添加的元素会自动排列到列表的尾部：
```
album.append('new song')
```
> 打印列表
```
print(album[0],album[-1]) //打印第一个和最后一个元素
```
- is和 is not：表示身份鉴别
> python中任何一个对象都要满足身份、类型、值这三个点

##### 条件语句
```
if condition:
    do something
elif conditino:
    do something
else:
    do something
```
