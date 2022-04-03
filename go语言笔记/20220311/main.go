package main

import (
	"reflect"
	"strings"
	"unsafe"

	"github.com/davecgh/go-spew/spew"
)

type Person struct {
	Name string
	Age  uint8
}

type People struct {
	Hand string
	Leg  uint
}

func main() {
	// pointer的转换作用
	a := &Person{Name: "wangyalou", Age: 23}
	b := (*People)(unsafe.Pointer(a))
	spew.Dump(a, b)

	// 偷梁换柱1
	c := strings.Repeat("x", 5)
	d := (*reflect.StringHeader)(unsafe.Pointer(&c))

	f := strings.Repeat("a", 5)
	g := (*reflect.StringHeader)(unsafe.Pointer(&f))
	d.Data = g.Data
	d.Len = g.Len

	spew.Dump(c, f)

	// 修改底层数据后，两个字符串都发生了改变
	w := &reflect.SliceHeader{
		Data: d.Data,
		Len:  d.Len,
	}
	cc := *(*[]byte)(unsafe.Pointer(w))
	cc[0] = 'x'
	spew.Dump(c, f)

	// 偷梁换柱2
	data := []byte{65, 66, 67}
	change := "this is raw data"
	changeHeader := (*reflect.StringHeader)(unsafe.Pointer(&change))
	changeHeader.Data = uintptr(unsafe.Pointer(&data[0]))
	changeHeader.Len = len(data)
	spew.Dump(change)
}
