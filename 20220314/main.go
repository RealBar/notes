package main

import (
	"fmt"
	"unsafe"
)

type Person struct {
	Data [256]byte
	Name string
}

func main() {
	var a [1024]byte

	for i := 0; i < 1024; i++ {
		a[i] = 'a'
	}

	fmt.Println(a)

	p := &Person{}
	p.Data[111] = 'a'
	fmt.Println(unsafe.Sizeof(*p))
}
