# singleflight 详解
## 概述
go的internal包下有一个singleflight包，但是不可以直接使用。因此开发者使用一般是通过golang.org/x/sync库下的internal包。这两个包的实现是大致相似的。我们着重说golang.org/x/sync。这个包的作用是保证对于一个传入的参数key，如果有多个协程发起调用，同时只有一个实际的调用发生。使用的场景一般是缓存防止雪崩。

## 用法
singleflight包只有一个可导出的类型：Group，它主要只有两个方法：Do, DoChan。另一个Forget是用来取消的。

Do是同步调用，所有调用Do并且传入了相同的key的协程，在同一时间只会发起一次传入的fn函数调用，其他的协程会被阻塞。

DoChan返回一个可读channel，调用方可以更灵活地处理，例如加超时读取
```go
func testSingleFlight(){
    s := singleflight.Group{}
    w := sync.Group{}
    for i :=0;i <5;i++{
        w.Add(1)
        go func(num int){
            fmt.Printf("No.%v is calling\n",i)
            res := s.DoChan("test_key", func()(interface{},error){
                fmt.Printf("No.%v doing real function\n",num)
                time.Sleep(time.Second * 5)
                return 100,nil
            })
            select{
                case r := <-res:
                    fmt.Printf("get result:%v\n",r.Val)
                case <- time.After(time.Second * 5):
                    fmt.Printf("timeout\n")
            }
        }(i)
    }
}
```
## 阻塞版原理
Group只有两个成员，一个Mutex锁和一个以传入key为key，call为value的map。call里有一个WaitGroup，每个key只会生成一个call。最基本的原理就是使用mutex同步读取map，如果没有值就new一个call，放到map里，并且执行传入的函数，并且将函数的返回值写入call的结构；如果有，就在call的WaitGroup上Wait，直到传入的函数Done了以后，从call里读取函数的结果。

如果要自己手动实现，可以考虑同步的版本，但是有两个注意点：
1. 是否可以使用读写所优化？不可以，因为这里上锁后并不是全部读，而是有一个协程会写，其他的读。
2. 注意，写携程lock在UnLock必须放在将call写入map，以及waitGroup.Add之后，尤其是Add，因为如果add没有同步，读写成会wait不住。
```go
type Result{
    val interface{}
    err error
    shared bool
}
type call struct{
    dups int
    w sync.WaitGroup
    val interface{}
    err error
    chans []chan<- Result
    forget bool
}
type Group struct{
    lock sync.Mutex
    m map[string]*call
}
```
## Chan版需要注意的坑
首先，要注意返回的chan类型是可读chan，而官方文档写了返回的chan是不会被关闭的，因此一定不能range。

其次，如果执行函数发生了panic，异步的方式是没有做recover的，因此一定会造成进程crash。原因见下：
```go
		if e, ok := c.err.(*panicError); ok {
			// In order to prevent the waiting channels from being blocked forever,
			// needs to ensure that this panic cannot be recovered.
			if len(c.chans) > 0 {
				go panic(e)
				select {} // Keep this goroutine around so that it will appear in the crash dump.
			} else {
				panic(e)
			}
		} else if c.err == errGoexit {
			// Already in the process of goexit, no need to call again
		} else {
			// Normal return
			for _, ch := range c.chans {
				ch <- Result{c.val, c.err, c.dups > 0}
			}
		}
```