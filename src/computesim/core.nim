# (c) 2024 Antonis Geralis
import threading/barrier

const
  SubgroupSize* {.intdefine.} = 8

type
  ValueType* = enum
    Bool, Int, Uint, Float, Double # Scalar types

  RawValue* = object
    data: uint64

  SubgroupOp* = enum
    invalid
    reconverge
    subgroupBroadcast
    subgroupBroadcastFirst
    subgroupAdd
    subgroupMin
    subgroupMax
    subgroupInclusiveAdd
    subgroupExclusiveAdd
    subgroupShuffle
    subgroupShuffleXor
    subgroupBallot
    subgroupElect
    subgroupAll
    subgroupAny
    subgroupBarrier
    barrier

  SubgroupCommand* = object
    id*: uint32
    case kind*: SubgroupOp
    of subgroupBroadcastFirst, subgroupAdd, subgroupMin, subgroupMax,
        subgroupInclusiveAdd, subgroupExclusiveAdd,
        subgroupBroadcast, subgroupShuffle, subgroupShuffleXor:
      dirty*: uint32
      t*: ValueType
      val*: RawValue
    of subgroupBallot, subgroupAll, subgroupAny:
      bVal*: bool
    of subgroupElect, subgroupBarrier, barrier, reconverge, invalid:
      discard

  SubgroupResult* = object
    id*: uint32
    case kind*: SubgroupOp
    of subgroupBroadcastFirst, subgroupAdd, subgroupMin, subgroupMax,
        subgroupInclusiveAdd, subgroupExclusiveAdd, subgroupBallot,
        subgroupBroadcast, subgroupShuffle, subgroupShuffleXor:
      t*: ValueType
      res*: RawValue
    of subgroupElect, subgroupAll, subgroupAny:
      bRes*: bool
    of subgroupBarrier, barrier, reconverge, invalid:
      discard

  BarrierHandle* = object
    x: ptr Barrier

proc toValue*[T](val: T): RawValue =
  cast[ptr T](addr result.data)[] = val

proc getValue*[T](v: RawValue): T =
  cast[ptr T](addr v.data)[]

proc getValueType*[T](x: T): ValueType =
  # Use when clauses to match against types
  when T is bool:
    ValueType.Bool
  elif T is int32:
    ValueType.Int
  elif T is uint32:
    ValueType.Uint
  elif T is float32:
    ValueType.Float
  elif T is float64:
    ValueType.Double
  else:
    {.error: "Unsupported type for getValueType".}

proc getHandle*(b: var Barrier): BarrierHandle {.inline.} =
  result = BarrierHandle(x: addr(b))

proc wait*(m: BarrierHandle) {.inline.} =
  wait(m.x[])

type
  ThreadClosure* = iterator (iterArg: SubgroupResult): SubgroupCommand
  SubgroupResults* = array[SubgroupSize, SubgroupResult]
  SubgroupCommands* = array[SubgroupSize, SubgroupCommand]
  SubgroupThreadIDs* = array[SubgroupSize, uint32]
  SubgroupThreads* = array[SubgroupSize, ThreadClosure]

const
  InvalidId* = high(uint32) # Sentinel value for empty/invalid

iterator threadsInGroup*(group: SubgroupThreadIDs): uint32 =
  for member in group.items:
    if member == InvalidId: break
    yield member
