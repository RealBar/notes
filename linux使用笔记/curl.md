# curl 一次说清楚
## case1
```shell
 curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```
- -f 如果发生错误，不输出错误页面（例如404页面）
- -s 不展示进度条或者错误信息
- -S 展示作错误信息
- -L 自动重定向

## case2
```shell
curl -X POST\
-d '{"a":1}'
-H "Content-Type:application/json"
```
- -d 带数据
- -H header，注意键值之间用冒号

