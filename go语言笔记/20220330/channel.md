# Channel
## 常用姿势
创建：
```go
c := make(chan int,10)
c := make(chan int)
```
没有缓存的叫做无缓存channel，写入时如果没有接收会block；

写入
```go
c <-10
```
接收
```go
<-c

a := <-c

a, ok := <-c

for e := range c{

}
```
select
```go
// select 如果没有default且全部的channel都没数据，会阻塞；如果有default，则如果没有channel可读，则会进入default。
select{
    case a := <-c:
    ...
    case <-ctx.Done():
    ...
    default:
}
// 空的select用来永久阻塞
select{}
```
作为参数，可以规定channel是只读还是只写
```go
func Stream(ctx context.Context, out chan<- Value) error {...

Done() <-chan struct{}
```
两个panic：关闭已经关闭的channel会panic，向已关闭的channel写入会panic。
## 先看文档(effectuve go)


## 读取的奇技淫巧
### range
range读取channel，会重复读取，直到close掉。而且range读取只能有一个返回值，因为range不需要判断是否关闭。对于关闭的channel，range会跳出循环。

### 读取并判断是否接收到数据
```go
func main(){
    c := make(chan int)
    go func(){
        time.Sleep(time.Second * 10)
        c <- 10
        close(c)
    }()

    for {
        select{
        case _,ok := <-c:

            fmt.Printf("not ")
        default:
        }
    }
}
```
我们都知道`data, ok := <- chan`第一个变量表示读出的数据，第二个变量表示**是否成功读取了数据**，有意思的是，第二个变量并不用于指示管道的关闭的状态。第二个变量常常被误以为关闭状态是因为它确实和管道状态有关，确切的来说，是和管道缓冲区是否有数据有关。



所以，如果channel中有数据，此时close，读取端采用二元读取，并不会立马收到false的，而是在全部数据读取完成后，才会收到false

## 检查channel是否关闭
没有准确的方法用来判断，除非使用unsafe。推荐使用context来控制channel的关闭。