# (c) 2024 Antonis Geralis
import threading/barrier, vectors

const
  SubgroupSize* {.intdefine.} = 8
  SubgroupFullMask* = (1 shl SubgroupSize) - 1
  # Precalculated masks for each possible invocation ID
  SubgroupMasks* = block:
    var masks: array[SubgroupSize, tuple[eq, ge, gt, le, lt: uint32]]
    for invocationId in 0 ..< SubgroupSize:
      let invocationMask = 1'u32 shl invocationId
      let ltMask = (invocationMask - 1) and SubgroupFullMask
      let leMask = ltMask or invocationMask
      let gtMask = not leMask and SubgroupFullMask
      let geMask = gtMask or invocationMask

      masks[invocationId] = (
        eq: invocationMask,
        ge: geMask,
        gt: gtMask,
        le: leMask,
        lt: ltMask
      )
    masks

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
    subgroupShuffleDown
    subgroupShuffleUp
    subgroupBallot
    subgroupElect
    subgroupAll
    subgroupAny
    subgroupAllEqual
    subgroupBarrier
    subgroupMemoryBarrier
    barrier
    memoryBarrier
    groupMemoryBarrier

  SubgroupCommand* = object
    id*: uint32
    case kind*: SubgroupOp
    of subgroupBroadcastFirst, subgroupAdd, subgroupMin, subgroupMax,
        subgroupInclusiveAdd, subgroupExclusiveAdd,
        subgroupBroadcast, subgroupShuffle, subgroupShuffleXor,
        subgroupShuffleDown, subgroupShuffleUp, subgroupAllEqual:
      dirty*: uint32
      t*: ValueType
      val*: RawValue
    of subgroupBallot, subgroupAll, subgroupAny:
      bVal*: bool
    of subgroupElect, subgroupBarrier, subgroupMemoryBarrier, barrier,
        memoryBarrier, groupMemoryBarrier, reconverge, invalid:
      discard

  SubgroupResult* = object
    id*: uint32
    case kind*: SubgroupOp
    of subgroupBroadcastFirst, subgroupAdd, subgroupMin, subgroupMax,
        subgroupInclusiveAdd, subgroupExclusiveAdd, subgroupBallot,
        subgroupBroadcast, subgroupShuffle, subgroupShuffleXor,
        subgroupShuffleDown, subgroupShuffleUp:
      t*: ValueType
      res*: RawValue
    of subgroupElect, subgroupAll, subgroupAny, subgroupAllEqual:
      bRes*: bool
    of subgroupBarrier, subgroupMemoryBarrier, barrier, memoryBarrier,
        groupMemoryBarrier, reconverge, invalid:
      discard

  BarrierHandle* = object
    x: ptr Barrier

  WorkGroupContext* = object
    gl_WorkGroupID*: UVec3           ## ID of the current workgroup [0..gl_NumWorkGroups)
    gl_WorkGroupSize*: UVec3         ## Size of the workgroup (x, y, z)
    gl_NumWorkGroups*: UVec3         ## Total number of workgroups (x, y, z)
    gl_NumSubgroups*: uint32         ## Number of subgroups in the workgroup
    gl_SubgroupID*: uint32           ## ID of the current subgroup [0..gl_NumSubgroups)

  ThreadContext* = object
    gl_GlobalInvocationID*: UVec3    ## Global ID of the current invocation [0..gl_NumWorkGroups*gl_WorkGroupSize)
    gl_LocalInvocationID*: UVec3     ## Local ID within the workgroup [0..gl_WorkGroupSize)
    gl_SubgroupInvocationID*: uint32 ## ID of the invocation within the subgroup [0..gl_SubgroupSize)

proc toValue*[T](val: T): RawValue {.noinit.} =
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
  ThreadClosure* = iterator (iterArg: SubgroupResult,
                             wg: WorkGroupContext, thread: ThreadContext): SubgroupCommand
  SubgroupResults* = array[SubgroupSize, SubgroupResult]
  SubgroupCommands* = array[SubgroupSize, SubgroupCommand]
  SubgroupThreadIDs* = array[SubgroupSize, uint32]
  SubgroupThreads* = array[SubgroupSize, ThreadClosure]
  ThreadContexts* = array[SubgroupSize, ThreadContext]

const
  InvalidId* = high(uint32) # Sentinel value for empty/invalid

iterator threadsInGroup*(group: SubgroupThreadIDs): uint32 =
  for member in group.items:
    if member == InvalidId: break
    yield member
