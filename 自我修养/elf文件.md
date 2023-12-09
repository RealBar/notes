# elf 文件
elf文件可以分为4类
- relocatable: 目标文件和静态代码库都属于这类(.a, .o)，这类文件是携带重定位符号的
- executable
- shared object
- core dump
# elf文件解析常用工具
以下所有的工具都在binutils包下
- objdump
- readelf
- size

## objdump

- -h 查看目标文件的各个段
- -s 将所有段以16进制输出
- -d 将包含指令的段反汇编
- -t 查看符号表（-ht一般连用）
- -f 查看文件头
- -x 查看段表、符号表、重定位表（集大成者）


```shell
(base) ➜  test git:(master) ✗ objdump -h elf.o                                                                                │
                                                                                                                              │
elf.o:     file format elf64-x86-64                                                                                           │
                                                                                                                              │
Sections:                                                                                                                     │
Idx Name          Size      VMA               LMA               File off  Algn                                                │
  0 .text         00000062  0000000000000000  0000000000000000  00000040  2**0                                                │
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE                                                                │
  1 .data         00000008  0000000000000000  0000000000000000  000000a4  2**2                                                │
                  CONTENTS, ALLOC, LOAD, DATA                                                                                 │
  2 .bss          00000008  0000000000000000  0000000000000000  000000ac  2**2                                                │
                  ALLOC                                                                                                       │
  3 .rodata       00000004  0000000000000000  0000000000000000  000000ac  2**0                                                │
                  CONTENTS, ALLOC, LOAD, READONLY, DATA                                                                       │
  4 .comment      0000002c  0000000000000000  0000000000000000  000000b0  2**0                                                │
                  CONTENTS, READONLY                                                                                          │
  5 .note.GNU-stack 00000000  0000000000000000  0000000000000000  000000dc  2**0                                              │
                  CONTENTS, READONLY                                                                                          │
  6 .note.gnu.property 00000020  0000000000000000  0000000000000000  000000e0  2**3                                           │
                  CONTENTS, ALLOC, LOAD, READONLY, DATA                                                                       │
  7 .eh_frame     00000058  0000000000000000  0000000000000000  00000100  2**3                                                │
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, DATA 
```
size 命令可以查看主要的三个段的长度
```shell
(base) ➜  自我修养 git:(master) ✗ size elf.o 
   text    data     bss     dec     hex filename
    222       8       8     238      ee elf.o 
```
## readelf
### -l：读取segment
section和segment：elf文件在编译完成后，是以section的方式组织的，section的排列在文件的section table中；segment是将相同类型（可读、可写、可执行）的section组织在一起的一个概念，存放在program header中。

```
~ ➤ readelf -l /bin/bash

Elf file type is EXEC (Executable file)
Entry point 0x422270
There are 9 program headers, starting at offset 64

Program Headers:
  Type           Offset             VirtAddr           PhysAddr
                 FileSiz            MemSiz              Flags  Align
  PHDR           0x0000000000000040 0x0000000000400040 0x0000000000400040
                 0x00000000000001f8 0x00000000000001f8  R E    0x8
  INTERP         0x0000000000000238 0x0000000000400238 0x0000000000400238
                 0x000000000000001c 0x000000000000001c  R      0x1
      [Requesting program interpreter: /lib64/ld-linux-x86-64.so.2]
  LOAD           0x0000000000000000 0x0000000000400000 0x0000000000400000
                 0x00000000000ffd0c 0x00000000000ffd0c  R E    0x200000
  LOAD           0x0000000000100548 0x0000000000700548 0x0000000000700548
                 0x000000000000b6fc 0x0000000000015440  RW     0x200000
  DYNAMIC        0x0000000000102df0 0x0000000000702df0 0x0000000000702df0
                 0x00000000000001f0 0x00000000000001f0  RW     0x8
  NOTE           0x0000000000000254 0x0000000000400254 0x0000000000400254
                 0x0000000000000044 0x0000000000000044  R      0x4
  GNU_EH_FRAME   0x00000000000e3750 0x00000000004e3750 0x00000000004e3750
                 0x0000000000004324 0x0000000000004324  R      0x4
  GNU_STACK      0x0000000000000000 0x0000000000000000 0x0000000000000000
                 0x0000000000000000 0x0000000000000000  RW     0x10
  GNU_RELRO      0x0000000000100548 0x0000000000700548 0x0000000000700548
                 0x0000000000002ab8 0x0000000000002ab8  R      0x1

 Section to Segment mapping:
  Segment Sections...
   00
   01     .interp
   02     .interp .note.ABI-tag .note.gnu.build-id .gnu.hash .dynsym .dynstr .gnu.version .gnu.version_r .rela.dyn .rela.plt .init .plt .plt.got .text .fini .rodata .eh_frame_hdr .eh_frame
   03     .init_array .fini_array .jcr .data.rel.ro .dynamic .got .got.plt .data .bss
   04     .dynamic
   05     .note.ABI-tag .note.gnu.build-id
   06     .eh_frame_hdr
   07
   08     .init_array .fini_array .jcr .data.rel.ro .dynamic .got
```
注意这里重点关注type=LOAD的Segment，一般的只有两个，一个RE的代表可读可执行，一般是.text, .init；另一个是RW代表可读写，一般是.data, .bss。