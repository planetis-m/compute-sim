# Compile with at least `-d:ThreadPoolSize=MaxConcurrentWorkGroups*(ceilDiv(workgroupSize, SubgroupSize)+1)` and
# `-d:danger --opt:none --panics:on --threads:on --mm:arc -d:useMalloc -g`
# ...and debug with nim-gdb or lldb

import std/math, computesim

type
  Buffers = object
    input, output: seq[int32]
    retirementCount: uint32

  Shared = tuple
    buffer: seq[int32]
    isLastWorkGroup: uint32

  Args = tuple
    n: uint32
    coarseFactor: uint32

proc reductionShader(b: ptr Buffers, smem: ptr Shared, a: Args) {.computeShader.} =
  let localIdx = gl_LocalInvocationID.x
  let gridSize = gl_NumWorkGroups.x
  let localSize = gl_WorkGroupSize.x
  let globalIdx = gl_WorkGroupID.x * localSize * 2 * a.coarseFactor + localIdx

  # Memory coalescing occurs when threads in the same subgroup access adjacent memory
  # locations simultaneously - not when a single thread accesses different locations
  # sequentially. Here, each thread reads two values with a fixed stride between them.
  var sum: int32 = 0
  var baseIdx = globalIdx
  for tile in 0 ..< a.coarseFactor:
    # echo "ThreadId ", localIdx, " indices: ", baseIdx, " + ", baseIdx + localSize
    sum += (if baseIdx < a.n: b.input[baseIdx] else: 0) +
        (if baseIdx + localSize < a.n: b.input[baseIdx + localSize] else: 0)
    baseIdx += 2 * localSize
  smem.buffer[localIdx] = sum

  memoryBarrier() # shared
  barrier()
  var stride = localSize div 2
  while stride > 0:
    if localIdx < stride:
      # echo "Final reduction ", localIdx, " + ", localIdx + stride
      smem.buffer[localIdx] += smem.buffer[localIdx + stride]
    memoryBarrier() # shared
    barrier()
    stride = stride div 2

  if localIdx == 0:
    b.output[gl_WorkGroupID.x] = smem.buffer[0]

  if gridSize > 1:
    memoryBarrier() # buffer
    if localIdx == 0:
      let ticket = atomicAdd(b.retirementCount, 1)
      smem.isLastWorkGroup = uint32(ticket == gridSize - 1)
    memoryBarrier() # shared
    barrier()
    # The last block sums the results of all other blocks
    if smem.isLastWorkGroup != 0:
      var sum: int32 = 0
      for i in countup(localIdx, gridSize, localSize):
        sum += b.output[i]
      smem.buffer[localIdx] = sum

      memoryBarrier() # shared
      barrier()
      var stride = localSize div 2
      while stride > 0:
        if localIdx < stride:
          smem.buffer[localIdx] += smem.buffer[localIdx + stride]
        memoryBarrier() # shared
        barrier()
        stride = stride div 2

      if localIdx == 0:
        b.output[0] = smem.buffer[0]
        # reset retirement count so that next run succeeds
        b.retirementCount = 0

# Main
const
  NumElements = 1024'u32
  CoarseFactor = 4'u32
  WorkGroupSize = 16'u32 # must be a power of two!
  Segment = WorkGroupSize * 2 * CoarseFactor

proc main =
  # Set the number of work groups and the size of each work group
  let numWorkGroups = uvec3(ceilDiv(NumElements, Segment), 1, 1)
  let workGroupSize = uvec3(WorkGroupSize, 1, 1)

  # Fill the input buffer
  var inputData = newSeq[int32](NumElements)
  for i in 0..<NumElements:
    inputData[i] = int32(i)

  var buffers = Buffers(
    input: ensureMove(inputData),
    output: newSeq[int32](numWorkGroups.x + 1),
    retirementCount: 0
  )

  # Run the compute shader on CPU, pass buffers as parameters.
  runComputeOnCpu(numWorkGroups, workGroupSize, reductionShader,
    ssbo = addr buffers,
    smem = (buffer: newSeq[int32](workGroupSize.x), isLastWorkGroup: 0'u32),
    args = (NumElements, CoarseFactor)
  )

  let result = buffers.output[0]
  let expected = (NumElements - 1)*NumElements div 2
  echo "Reduction result: ", result, ", expected: ", expected

main()
