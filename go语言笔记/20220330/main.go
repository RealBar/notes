package main

import (
	"fmt"
	"time"
)

type Person struct{}

func main() {
	c := make(chan int, 10)

	go func() {
		c <- 1
		c <- 1
		close(c)
	}()
	time.Sleep(time.Second)

	// 打印false
	fmt.Println(checkChannelIsClosed(c))
}

func checkChannelIsClosed(c chan int) bool {
	select {
	case _, ok := <-c:
		return !ok
	default:
	}
	return false
}

func testCheckClose() {
	c := make(chan *Person)

	go func() {
		time.Sleep(time.Second * 4)
		c <- &Person{}
		close(c)
	}()

	for {
		a, b := <-c
		if !b {
			time.Sleep(time.Second)
			fmt.Println("closed")
			break
		}
		fmt.Printf("%v,%v\n", a, b)
	}
}
