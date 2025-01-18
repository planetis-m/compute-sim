# Copyright (c) 2024 Antonis Geralis
import std/[assertions, typetraits]

proc copyInto*[N, T](dst: var array[N, T], src: array[N, T])
proc copyInto*[T](dst: var ref T, src: ref T)
proc copyInto*(dst: var string, src: string)
proc copyInto*[T: distinct](dst: var T, src: T)
proc copyInto*[T](dst: var T, src: T)
proc copyInto*[T: object | tuple](dst: var T, src: T)

proc copyInto*[T](dst: var seq[T], src: seq[T]) =
  dst.setLen(src.len)
  when supportsCopyMem(T):
    if src.len > 0:
      copyMem(addr dst[0], addr src[0], src.len * sizeof(T))
  else:
    for i in 0..<src.len:
      copyInto(dst[i], src[i])

proc copyInto*[N, T](dst: var array[N, T], src: array[N, T]) =
  when supportsCopyMem(T):
    copyMem(addr dst, addr src, sizeof(src))
  else:
    for i in low(src)..high(src):
      copyInto(dst[i], src[i])

proc copyInto*[T](dst: var ref T, src: ref T) =
  if dst.isNil:
    dst = new(typeof(src[]))
  copyInto(dst[], src[])

proc copyInto*(dst: var string, src: string) =
  dst.setLen(src.len)
  copyMem(addr dst[0], addr src[0], src.len)

proc copyInto*[T: distinct](dst: var T, src: T) =
  copyInto(dst.distinctBase, src.distinctBase)

proc copyInto*[T](dst: var T, src: T) =
  static: assert supportsCopyMem(T)
  copyMem(addr dst, addr src, sizeof(T))

proc copyInto*[T: object | tuple](dst: var T, src: T) =
  when supportsCopyMem(T):
    copyMem(addr dst, addr src, sizeof(T))
  else:
    for dstv, srcv in fields(dst, src):
      copyInto(dstv, srcv)
