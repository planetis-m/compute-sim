import std/bitops, core, vectors

const
  SubgroupOpError = "This function can only be used inside a proc marked with {.computeShader.}"

template gl_SubgroupEqMask*(): UVec4 =
  ## Returns a mask where only the bit at the current invocation's index is set
  uvec4(SubgroupMasks[gl_SubgroupInvocationID].eq, 0, 0, 0)

template gl_SubgroupGeMask*(): UVec4 =
  ## Returns a mask where bits at and above the current invocation's index are set
  uvec4(SubgroupMasks[gl_SubgroupInvocationID].ge, 0, 0, 0)

template gl_SubgroupGtMask*(): UVec4 =
  ## Returns a mask where bits above the current invocation's index are set
  uvec4(SubgroupMasks[gl_SubgroupInvocationID].gt, 0, 0, 0)

template gl_SubgroupLeMask*(): UVec4 =
  ## Returns a mask where bits at and below the current invocation's index are set
  uvec4(SubgroupMasks[gl_SubgroupInvocationID].le, 0, 0, 0)

template gl_SubgroupLtMask*(): UVec4 =
  ## Returns a mask where bits below the current invocation's index are set
  uvec4(SubgroupMasks[gl_SubgroupInvocationID].lt, 0, 0, 0)

template subgroupBroadcast*[T](value: T; id: uint32): T =
  ## Broadcasts value from thread with specified id to all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupBroadcastFirst*[T](value: T): T =
  ## Broadcasts value from first active thread to all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupAdd*[T](value: T): T =
  ## Returns sum of value across all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupMin*[T](value: T): T =
  ## Returns minimum of value across all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupMax*[T](value: T): T =
  ## Returns maximum of value across all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupInclusiveAdd*[T](value: T): T =
  ## Returns inclusive prefix sum of value for current thread
  {.error: SubgroupOpError.}

template subgroupExclusiveAdd*[T](value: T): T =
  ## Returns exclusive prefix sum of value for current thread
  {.error: SubgroupOpError.}

template subgroupShuffle*[T](value: T; id: uint32): T =
  ## Returns value from thread with specified id
  {.error: SubgroupOpError.}

template subgroupShuffleXor*[T](value: T; mask: uint32): T =
  ## Returns value from thread with id equal to current_id XOR mask
  {.error: SubgroupOpError.}

template subgroupShuffleDown*[T](value: T; delta: uint32): T =
  ## Returns value from thread with index current_id + delta
  {.error: SubgroupOpError.}

template subgroupShuffleUp*[T](value: T; delta: uint32): T =
  ## Returns value from thread with index current_id - delta
  {.error: SubgroupOpError.}

template subgroupBallot*(condition: bool): UVec4 =
  ## Returns bitmap of which threads have condition true
  {.error: SubgroupOpError.}

func subgroupBallotBitCount*(ballot: UVec4): uint32 =
  ## Returns the number of set bits in a ballot value, only counting
  ## bits up to gl_SubgroupSize
  uint32(countSetBits(masked(ballot.x, SubgroupFullMask)))

func subgroupBallotBitExtract*(value: UVec4, index: uint32): bool =
  ## Returns true if the bit at position index is set in value
  ## Only valid for indices less than gl_SubgroupSize
  testBit(value.x, masked(index, SubgroupSize - 1))

template subgroupInverseBallot*(value: UVec4): bool =
  ## Returns true if the bit at this invocation's index in value is set
  subgroupBallotBitExtract(value, gl_SubgroupInvocationID)

template subgroupBallotInclusiveBitCount*(value: UVec4): uint32 =
  ## Returns the number of bits set in value up to the current invocation's index (inclusive)
  ## Only counts bits up to gl_SubgroupSize
  uint32(countSetBits(masked(value.x, SubgroupMasks[gl_SubgroupInvocationID].le)))

template subgroupBallotExclusiveBitCount*(value: UVec4): uint32 =
  ## Returns the number of bits set in value up to the current invocation's index (exclusive)
  ## Only counts bits up to gl_SubgroupSize
  uint32(countSetBits(masked(value.x, SubgroupMasks[gl_SubgroupInvocationID].lt)))

func subgroupBallotFindLSB*(value: UVec4): uint32 =
  ## Returns the index of the least significant 1 bit in value
  ## Only considers the bottom gl_SubgroupSize bits
  let mask = masked(value.x, SubgroupFullMask)
  if mask == 0: high(uint32) else: uint32(countTrailingZeroBits(mask))

func subgroupBallotFindMSB*(value: UVec4): uint32 =
  ## Returns the index of the most significant 1 bit in value
  ## Only considers the bottom gl_SubgroupSize bits
  let mask = masked(value.x, SubgroupFullMask)
  if mask == 0: high(uint32) else: uint32(fastLog2(mask))

template subgroupElect*(): bool =
  ## Returns true for exactly one active thread in subgroup
  {.error: SubgroupOpError.}

template subgroupAll*(condition: bool): bool =
  ## Returns true if condition is true for all active threads
  {.error: SubgroupOpError.}

template subgroupAny*(condition: bool): bool =
  ## Returns true if condition is true for any active thread
  {.error: SubgroupOpError.}

template subgroupAllEqual*[T](value: T): bool =
  ## Returns true if value is equal across all active threads in subgroup
  {.error: SubgroupOpError.}

template subgroupBarrier*() =
  ## Synchronizes all threads within the current subgroup
  {.error: SubgroupOpError.}

template barrier*() =
  ## Synchronizes all threads within the current workgroup
  {.error: SubgroupOpError.}

template subgroupMemoryBarrier*() =
  ## Performs a memory barrier that ensures memory operations within the subgroup
  ## are properly ordered as seen by other invocations.
  {.error: SubgroupOpError.}

# GLSL-style atomic operations implementation using sysatomics
# Memory model is sequentially consistent as per GLSL spec.

type
  AtomicInt* = int32|uint32 ## Type class for atomic integer operations

{.push inline, discardable.}

proc atomicAdd*[T: AtomicInt](mem: var T, data: T): T =
  ## Performs atomic addition on mem with data.
  ## Returns the original value stored in mem.
  atomicFetchAdd(addr mem, data, ATOMIC_SEQ_CST)

proc atomicAnd*[T: AtomicInt](mem: var T, data: T): T =
  ## Performs atomic AND on mem with data.
  ## Returns the original value stored in mem.
  atomicFetchAnd(addr mem, data, ATOMIC_SEQ_CST)

proc atomicOr*[T: AtomicInt](mem: var T, data: T): T =
  ## Performs atomic OR on mem with data.
  ## Returns the original value stored in mem.
  atomicFetchOr(addr mem, data, ATOMIC_SEQ_CST)

proc atomicXor*[T: AtomicInt](mem: var T, data: T): T =
  ## Performs atomic XOR on mem with data.
  ## Returns the original value stored in mem.
  atomicFetchXor(addr mem, data, ATOMIC_SEQ_CST)

proc atomicExchange*[T: AtomicInt](mem: var T, data: T): T =
  ## Atomically stores data into mem and returns the original value.
  atomicExchangeN(addr mem, data, ATOMIC_SEQ_CST)

{.pop.}

proc atomicCompSwap*[T: AtomicInt](mem: var T, compare, data: T): T {.discardable.} =
  ## Performs an atomic comparison of compare with the contents of mem.
  ## If the content of mem is equal to compare, then the content of data
  ## is written into mem, otherwise the content of mem is unmodified.
  ## Returns the original content of mem regardless of the outcome of the comparison.
  var compareVar = compare
  discard atomicCompareExchangeN(addr mem, addr compareVar, data,
    weak = false, ATOMIC_SEQ_CST, ATOMIC_SEQ_CST)
  compareVar
