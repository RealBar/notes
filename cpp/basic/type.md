# 变量的声明和定义的区别
**声明让名字为程序所知，定义负责创建与名字关联的实体**
声明机制的作用是为了让程序可以使用定义在别的文件的变量
```c++
extern int a; // 仅声明
int a; // 声明和定义
extern int a= 1; // 定义，抵消了extern的作用
```
- 声明不会申请存储空间，定义会
- 函数体内部不能出
# 变量的定义和初始化
- 变量的定义包含：基本数据类型和申明符
- 同一条语句中，数据类型只有一个，但是声明符的形式可以各不相同，所以一条语句可以定义出不同类型的变量
```c++
int a = 1, *p = &a, &r = a;
```
- 注意，**初始化不等于赋值**
- 内置变量必须初始化
```
int a = 1;
int a(1);
int a{1}; // c++11
int a; // 如果在函数体外，默认初始化为0；函数体内部不会被初始化，访问或拷贝会报错
```
- 复合类型有默认构造函数，可以不用显示初始化
```c++
std::string name;
```
# 引用
- 引用是别名，不是对象，因此不能创建引用的引用
- 引用必须初始化
- 引用一旦创建，不能换邦
```c++
int a = 1;
int &b = a; // right
int &d; // 编译错误，引用必须初始化
int &c = 1; // 编译错误，引用只能用变量初始化
int &e = b; // 编译错误，不能创建引用的引用
b = 12
int f = b; // f==12 
```
# 指针
- 指针是一个对象
- 指针不用初始化，但是c++ primer建议所有指针都要初始化
- 指针可以赋值和拷贝
```c++
int a = 1;
int *b = &a;
*b = 3; // a==3
```
## 关于类型修饰符
类型修饰符&,\*，有两种写法
```c++
int* a;
int* b;

int *a,*b; 
int* a,b;// 合法，但是容易引起歧义。这里a是指针，b是基本类型
```
将修饰符和类型写在一起一般每条语句都只定义一个变量，将修饰符和变量写在一起每条语句可以定义多个便量。c++ primer采用后者
> 在声明语句中，&和\*组成复合类型；在表达式中，\*和&是运算符。在不同场景中，他们的含义截然不同

## 空指针
```c++
int *a = 0;
int *a = nullptr; // c++11,两种等效
#include<cstdlib>
int *c = NULL; // 不推荐
```
## void*
- void* 用于存放任意类型的指针
- void* 的用途
  - 和另一个指针比较
  - 赋值给另一个void*
  - 作为函数的输入和输出
  - 强转
- void* 不能做的
  - 解引用

## 指针和引用
- 引用不是对象，因此没有指向引用的指针
- 指针是对象，因此有指针的引用
```c++
int a = 1;
int *p = &a;
int *&r = p; // r是指针p的引用
r = nullptr; // 使用r和p是一样的，p==nullptr
```
阅读变量类型的技巧： 从变量名称最近的地方开始看，比如这里，r最近的是&，因此r首先是一个引用；然后往左看，\*说明r是一个指针的引用