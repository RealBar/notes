package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"

	"golang.org/x/sync/singleflight"
)

func main() {
	// defer 和return的顺序
	fmt.Println(testDeferSeq())
	fmt.Println(testDeferSeq2())
	testDeferPanic()
	// test singleflight
	testSingleFlight()

	// test singleflight chan
	testSingleFlightChan()
}

func testDeferSeq() int {
	a := map[string]int{"a": 1}
	defer func() { a["a"] = 2 }()
	return a["a"]
}

func testDeferSeq2() (res int) {
	res = 10
	defer func() {
		res++
	}()
	return 5
}

func testDeferPanic() {
	defer func() {
		fmt.Printf("defer get panic!\n")
		if err := recover(); err != nil {
			fmt.Println("recoverd!")
		}
	}()
	fmt.Println("panicing")
	panic("aaa")
}

func testSingleFlight() {
	g := singleflight.Group{}
	w := sync.WaitGroup{}
	for i := 0; i < 5; i++ {
		w.Add(1)
		go func(num int) {
			defer w.Done()
			fmt.Printf("No.%v calling Do\n", num)
			res, _, _ := g.Do("test_key", func() (interface{}, error) {
				fmt.Println("Doing long query...")
				time.Sleep(time.Second * 10)
				return 100, nil
			})
			fmt.Printf("No.%v res:%v\n", num, res)
		}(i)
	}
	w.Wait()
}

func testSingleFlightChan() {
	w := sync.WaitGroup{}
	s := singleflight.Group{}
	for i := 0; i < 5; i++ {
		w.Add(1)
		go func(num int) {
			defer w.Done()
			fmt.Printf("No.%v starting DoChan\n", num)
			res := s.DoChan("test_key", func() (interface{}, error) {
				fmt.Printf("Doing real query by No.%v\n", num)
				time.Sleep(time.Second * 3)
				return 100, nil
			})
			select {
			case r := <-res:
				fmt.Printf("No.%v get res:%v\n", num, r.Val)
			case <-time.After(time.Second * time.Duration(rand.Intn(5)+1)):
				fmt.Printf("No.%v timeout\n", num)

			}
		}(i)
	}
	w.Wait()
}
