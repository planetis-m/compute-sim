# Compile with at least `-d:ThreadPoolSize=MaxConcurrentWorkGroups*(ceilDiv(workgroupSize, SubgroupSize)+1)` and
# `-d:danger --opt:none --panics:on --threads:on --mm:arc -d:useMalloc -g`
# ...and debug with nim-gdb or lldb

import std/math, computesim

type
  Buffers = object
    input, output: seq[int32]

  Shared = seq[int32]
  Args = tuple
    n: uint32
    coarseFactor: uint32

proc reductionShader(b: ptr Buffers, smem: ptr Shared, args: Args) {.computeShader.} =
  let (n, coarseFactor) = args

  let localIdx = gl_LocalInvocationID.x
  let localSize = gl_WorkGroupSize.x
  var globalIdx = gl_WorkGroupID.x * localSize * 2 * coarseFactor + localIdx

  # Memory coalescing occurs when threads in the same subgroup access adjacent memory
  # locations simultaneously - not when a single thread accesses different locations
  # sequentially. Here, each thread reads two values with a fixed stride between them.
  var sum: int32 = 0
  for tile in 0 ..< coarseFactor:
    sum += int32(globalIdx < n) * b.input[globalIdx] +
        int32(globalIdx + localSize < n) * b.input[globalIdx + localSize]
    globalIdx += 2 * localSize
  smem[localIdx] = sum

  barrier() # was memoryBarrierShared(); barrier();
  var stride = localSize div 2
  while stride > 8:
    if localIdx < stride:
      # echo "Final reduction ", localIdx, " + ", localIdx + stride
      smem[localIdx] += smem[localIdx + stride]
    barrier() # was memoryBarrierShared(); barrier();
    stride = stride div 2

  # Final reduction within each subgroup
  if localIdx < 8:
    smem[localIdx] += smem[localIdx + 8]
    subgroupBarrier() # on GPU threads within a subgroup execute in lock-step
    smem[localIdx] += smem[localIdx + 4]
    subgroupBarrier() # was subgroupMemoryBarrierShared();
    smem[localIdx] += smem[localIdx + 2]
    subgroupBarrier() # subgroupMemoryBarrierShared();
    smem[localIdx] += smem[localIdx + 1]
    subgroupBarrier() # subgroupMemoryBarrierShared();

  if localIdx == 0:
    b.output[gl_WorkGroupID.x] = smem[0]

# Main
const
  NumElements = 1024'u32
  CoarseFactor = 4'u32
  WorkGroupSize = 16'u32
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
    output: newSeq[int32](numWorkGroups.x)
  )

  # Run the compute shader on CPU, pass buffers as parameters.
  runComputeOnCpu(numWorkGroups, workGroupSize, reductionShader,
    ssbo = addr buffers,
    smem = newSeq[int32](workGroupSize.x),
    args = (NumElements, CoarseFactor)
  )

  let result = sum(buffers.output)
  let expected = (NumElements - 1)*NumElements div 2
  echo "Reduction result: ", result, ", expected: ", expected

main()
