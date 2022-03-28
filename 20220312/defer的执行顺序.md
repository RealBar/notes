# defer 和return究竟是谁先执行？
```
func main() {
	fmt.Println(testDeferSeq())
}

func testDeferSeq() int {
	a := map[string]int{"a": 1}
	defer func() { a["a"] = 2 }()
	return a["a"]
}

```
执行结果：
```
1
```
结论： return语句先执行但是并没有返回，执行defer，最后返回。但是有一种情况要特别小心：返回值命名的函数，如果在**defer中修改了返回值，最终会生效**
```go
func testDeferSeq2() (res int) {
	res = 10
	defer func() {
		res++
	}()
	return 5
}
```
执行结果：
```
6
```
另外，defer之后的代码如果发生panic，defer能**确保被执行**，但是如果panic发生在defer之前，则不能被保证。
```go
func testDeferPanic(){
	defer func(){
		
	}()
	panic("aaa")
}
```