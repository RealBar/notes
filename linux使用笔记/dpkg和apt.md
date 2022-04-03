# linux的软件安装
## 开门见山：apt系和dpkg常用命令
### apt-get 

- apt-get install:不解释
  - -y：默认安装
- apt-get update/upgrade:不解释
- apt-get autoremove/autoclean/remove/purge: 删除为了没有必要的依赖软件/删除本地没用的软件包文件/删除软件/删除软件并删除配置文件

### apt-cache
- show: 查看所有版本的基础信息
- showpkg: 
- depends
- rdepends
### apt添加源
先添加key  
 ```
 ascii加固过的公钥
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg  
 or
没有加固过的
 wget -O /usr/share/keyrings/<myrepository-archive-keyring.gpg> <https://example.com/key/repo-key.gpg>
 ```
 然后添加源
 ```
 echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
注意：
1. wget 用大O，curl用小o
2. gpg公钥分为ascii加固（armor）过的和没有加固过，如果是加固过的，需要用gpg -deamror 解加固
3. 现在的apt公钥不再建议使用apt-key命令管理，这个命令会在22.04版本后移除。建议的是直接将解加固后的公钥放置到/usr/share/keyring/目录下，需要注意：**密钥文件的名称要和/etc/apt/source.list.d/目录下的名称保持一致**，例如docker仓库：  
> 公钥：/usr/share/keyrings/docker-archive-keyring.gpg  
source:/etc/apt/sources.list.d/docker.list  

然后，source的内容要加上 signed-by=key_path，例如  
```
deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" 
```


### dpkg
- -l:已安装的全部包
- -L package_name: 显示包所包含的全部文件安装路径
- -S file: 显示某个文件属于哪个包
- --print-architecture: 显示当前cpu架构，64位为amd64（注意区别于uname -m的结果x86_64）
- -P/-r purge/remove 区别是purge删除配置文件