# NPTL、进程和线程
## linux的两个线程库
- linuxthread：不符合POSIX规范
- NPTL：pthread库
- NGPT（已废弃）  

linux可以使用下边的命令检查thread库是哪种
```shell
$ getconf GNU_LIBPTHREAD_VERSION

NPTL 2.19
```
## 什么是pid，tgid？
看[stackoverflow:如果线程都用同一个pid，那么内核如何区分调度他们？](https://stackoverflow.com/questions/9305992/if-threads-share-the-same-pid-how-can-they-be-identified)  

The four threads will have the same PID but only when viewed from above. What you (as a user) calls a PID is not what the kernel (looking from below) calls a PID.  
In the kernel, each thread has its own ID, called a PID, although it would possibly make more sense to call this a TID, or thread ID, and they also have a TGID (thread group ID) which is the PID of the first thread that was created when the process was created.  
When a new process is created, it appears as a thread where both the PID and TGID are the same (currently unused) number.  
When a thread starts another thread, that new thread gets its own PID (so the scheduler can schedule it independently) but it inherits the TGID from the original thread.  
That way, the kernel can happily schedule threads independent of what process they belong to, while processes (thread group IDs) are reported to you.  
The following hierarchy of threads may help(a):
```
                         USER VIEW
                         vvvv vvvv
              |          
<-- PID 43 -->|<----------------- PID 42 ----------------->
              |                           |
              |      +---------+          |
              |      | process |          |
              |     _| pid=42  |_         |
         __(fork) _/ | tgid=42 | \_ (new thread) _
        /     |      +---------+          |       \
+---------+   |                           |    +---------+
| process |   |                           |    | process |
| pid=43  |   |                           |    | pid=44  |
| tgid=43 |   |                           |    | tgid=42 |
+---------+   |                           |    +---------+
              |                           |
<-- PID 43 -->|<--------- PID 42 -------->|<--- PID 44 --->
              |                           |
                        ^^^^^^ ^^^^
                        KERNEL VIEW
```
You can see that starting a new process (on the left) gives you a new PID and a new TGID (both set to the same value). Starting a new thread (on the right) gives you a new PID while maintaining the same TGID as the thread that started it.


FYI, getpid() returns tgid:
```c
 asmlinkage long sys_getpid(void) { return current->tgid;}, as shown in www.makelinux.com/ 
 ```
总结：
1. 每个线程都有不同的pid，内核根据pid进行调度
2. NPTL创建的线程和原线程有相同的tgid；fork创建的进程pid、tgid都不同
3. 对于进程的第一个线程，pid=tgid
4. getpid返回的实际上是tgid
5. gettid返回的实际上是真正的内核pid



## 进程组 pgid，session
> A process group is a collection of related processes which can all be signalled at once.  
A session is a collection of process groups, which are either attached to a single terminal device (known as the controlling terminal) or not attached to any terminal.  
Sessions are used for job control: one of the process groups in the session is the foreground process group, and can be sent signals by terminal control characters. You can think of a session with a controlling terminal as corresponding to a "login" on that terminal. (Daemons normally disassociate themselves from any controlling terminal by creating a new session without one.)  
e.g. if you run some_app from the shell, the shell creates a new process group for it, and makes that the foreground process group of the session. (some_app might create some child processes; by default they will be part of the same process group.) If you then press ^Z, some_app's process group is signalled to stop it; and the shell's process group is switched to be the foreground process group again. Then e.g.bg %1 would start some_app's process group again, but keep it running in the background.

POSIX标准：
> The POSIX standard says:  
If pid is greater than 0, sig shall be sent to the process whose process ID is equal to pid.  
If pid is 0, sig shall be sent to all processes (excluding an unspecified set of system processes) whose **process group ID is equal to the process group ID of the sender**, and for which the process has permission to send a signal.  
If pid is -1, sig shall be sent to all processes (excluding an unspecified set of system processes) for which the process has permission to send that signal.  
If pid is negative, but not -1, sig shall be sent to all processes (excluding an unspecified set of system processes) whose process group ID is equal to the absolute value of pid, and for which the process has permission to send a signal.

## fork详解