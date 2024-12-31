# Compile with at least `-d:ThreadPoolSize=workgroupSize+1` and
# `-d:danger --opt:none --panics:on --threads:on --tlsEmulation:off --mm:arc -d:useMalloc -g`
# ...and debug with nim-gdb or lldb

import computesim, std/[atomics, math]

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

proc reductionShader(env: GlEnvironment, b: ptr Buffers, smem: ptr Shared, args: Args) {.computeShader.} =
  let (n, coarseFactor) = args

  let localIdx = env.gl_LocalInvocationID.x
  let gridSize = env.gl_NumWorkGroups.x
  let localSize = env.gl_WorkGroupSize.x
  var globalIdx = env.gl_WorkGroupID.x * localSize * 2 * coarseFactor + localIdx

  var sum: int32 = 0
  for tile in 0 ..< coarseFactor:
    # echo "ThreadId ", localIdx, " indices: ", globalIdx, " + ", globalIdx + localSize
    sum += b.input[globalIdx] +
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
    b.output[env.gl_WorkGroupID.x] = smem.buffer[0]

  if gridSize > 1:
    barrier() # was memoryBarrierBuffer();
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

      barrier()
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
  numElements = 256'u32
  coarseFactor = 1'u32
  localSize = 16'u32 # workgroupSize
  segment = localSize * 2 * coarseFactor

proc main =
  # Set the number of work groups and the size of each work group
  let numWorkGroups = uvec3(ceilDiv(numElements, segment), 1, 1)
  let workGroupSize = uvec3(localSize, 1, 1)

  # Fill the input buffer
  var inputData = newSeq[int32](numElements)
  for i in 0..<numElements:
    inputData[i] = int32(i)

  var buffers = Buffers(
    input: ensureMove(inputData),
    output: newSeq[int32](numWorkGroups.x + 1),
    retirementCount: default(Atomic[uint32])
  )

  # Run the compute shader on CPU, pass buffers as parameters.
  runComputeOnCpu(numWorkGroups, workGroupSize, reductionShader, addr buffers,
    Shared(buffer: newSeq[int32](workGroupSize.x), isLastWorkGroup: 0'u32),
    (numElements, coarseFactor))

  let result = buffers.output[0]
  let expected = (numElements - 1)*numElements div 2
  echo "Reduction result: ", result, ", expected: ", expected

main()
