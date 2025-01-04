# Compile with at least `-d:ThreadPoolSize=MaxConcurrentWorkGroups*(ceilDiv(workgroupSize, SubgroupSize)+1)` and
# `-d:danger --opt:none --panics:on --threads:on --mm:arc -d:useMalloc -g`
# ...and debug with nim-gdb or lldb

import std/[atomics, math], computesim

type
  Buffers = object
    input, output: seq[int32]
    retirementCount: Atomic[uint32]

  Shared = object
    buffer: seq[int32]
    isLastWorkGroup: uint32

  Args = tuple
    n: uint32
    coarseFactor: uint32

proc reductionShader(b: ptr Buffers, smem: ptr Shared, args: Args) {.computeShader.} =
  let (n, coarseFactor) = args

  let localIdx = gl_LocalInvocationID.x
  let gridSize = gl_NumWorkGroups.x
  let localSize = gl_WorkGroupSize.x
  var globalIdx = gl_WorkGroupID.x * localSize * 2 * coarseFactor + localIdx

  var sum: int32 = 0
  for tile in 0 ..< coarseFactor:
    # echo "ThreadId ", localIdx, " indices: ", globalIdx, " + ", globalIdx + localSize
    # todo: use arithmetic to mask out invalid accesses instead
    sum += (if globalIdx < n: b.input[globalIdx] else: 0) +
        (if globalIdx + localSize < n: b.input[globalIdx + localSize] else: 0)
    globalIdx += 2 * localSize
  smem.buffer[localIdx] = sum

  barrier() # was memoryBarrierShared(); barrier();
  var stride = localSize div 2
  while stride > 0:
    if localIdx < stride:
      # echo "Final reduction ", localIdx, " + ", localIdx + stride
      smem.buffer[localIdx] += smem.buffer[localIdx + stride]
    barrier() # was memoryBarrierShared(); barrier();
    stride = stride div 2

  if localIdx == 0:
    b.output[gl_WorkGroupID.x] = smem.buffer[0]

  if gridSize > 1:
    subgroupBarrier() # was memoryBarrierBuffer();
    if localIdx == 0:
      let ticket = fetchAdd(b.retirementCount, 1)
      smem.isLastWorkGroup = uint32(ticket == gridSize - 1)
    barrier() # was memoryBarrierShared(); barrier();
    # The last block sums the results of all other blocks
    if smem.isLastWorkGroup != 0:
      var sum: int32 = 0
      for i in countup(localIdx, gridSize, localSize):
        sum += b.output[i]
      smem.buffer[localIdx] = sum

      barrier() # was memoryBarrierShared(); barrier();
      var stride = localSize div 2
      while stride > 0:
        if localIdx < stride:
          smem.buffer[localIdx] += smem.buffer[localIdx + stride]
        barrier() # was memoryBarrierShared(); barrier();
        stride = stride div 2

      if localIdx == 0:
        b.output[0] = smem.buffer[0]
        # reset retirement count so that next run succeeds
        b.retirementCount.store(0)

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
    retirementCount: default(Atomic[uint32])
  )

  # Run the compute shader on CPU, pass buffers as parameters.
  runComputeOnCpu(numWorkGroups, workGroupSize, reductionShader,
    ssbo = addr buffers,
    smem = Shared(buffer: newSeq[int32](workGroupSize.x), isLastWorkGroup: 0'u32),
    args = (NumElements, CoarseFactor)
  )

  let result = buffers.output[0]
  let expected = (NumElements - 1)*NumElements div 2
  echo "Reduction result: ", result, ", expected: ", expected

main()
