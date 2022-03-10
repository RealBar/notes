package main

import (
	"fmt"
	"reflect"
	"strings"
	"unsafe"

	"github.com/davecgh/go-spew/spew"
)

func main() {
	testUnchangable()
}

func test1() {
	a := strings.Repeat("x", 10)
	p1 := *(*reflect.StringHeader)(unsafe.Pointer(&a))

	a2 := a
	p2 := *(*reflect.StringHeader)(unsafe.Pointer(&a2))
	spew.Dump(p1)
	spew.Dump(p2)
}

func test2() {
	b1 := "wangyalou"
	b2 := "wangyalou"
	spew.Dump(*(*reflect.StringHeader)(unsafe.Pointer(&b1)))
	spew.Dump(*(*reflect.StringHeader)(unsafe.Pointer(&b2)))
}

func testChangable() {
	a := strings.Repeat("x", 10)

	pa := *(*reflect.StringHeader)(unsafe.Pointer(&a))
	sa := reflect.SliceHeader{
		Data: pa.Data,
		Len:  pa.Len,
		Cap:  pa.Len,
	}
	ca := *(*[]byte)(unsafe.Pointer(&sa))
	ca[0] = 'a'
	fmt.Printf("%v", a)
}

func testUnchangable() {
	a := "xxxxxxxxxx"

	pa := *(*reflect.StringHeader)(unsafe.Pointer(&a))
	sa := reflect.SliceHeader{
		Data: pa.Data,
		Len:  pa.Len,
		Cap:  pa.Len,
	}
	ca := *(*[]byte)(unsafe.Pointer(&sa))
	ca[0] = 'a'
	fmt.Printf("%v", a)
}
