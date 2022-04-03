package main

import (
	"sync"
	"time"
)

type Cache struct {
	data []byte
	lock sync.RWMutex
	ttl  time.Duration
}

func NewCache(size int, ttl time.Duration) *Cache {
	return &Cache{data: make([]byte, 0, size), ttl: ttl}
}

func (c *Cache) Set() {

}
