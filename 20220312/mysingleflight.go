package main

import "sync"

type Group struct {
	lock sync.Mutex
	m    map[string]*call
}

type call struct {
	w    sync.WaitGroup
	dups int
	val  interface{}
	err  error
}

// 关键点：通过map来判断当前是否已经有协程在执行方法了
func (g *Group) Do(key string, fn func() (interface{}, error)) (interface{}, bool, error) {
	g.lock.Lock()
	if g.m == nil {
		g.m = map[string]*call{}
	}
	c, ok := g.m[key]
	if ok {
		c.dups++
		g.lock.Unlock()
		c.w.Wait()
		return c.val, c.dups > 0, c.err
	}
	c = new(call)
	g.m[key] = c
	c.w.Add(1)
	g.lock.Unlock()

	c.val, c.err = fn()
	c.w.Done()
	return c.val, c.dups > 0, c.err
}
