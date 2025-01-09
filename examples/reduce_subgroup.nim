# Compile with at least `-d:ThreadPoolSize=MaxConcurrentWorkGroups*(ceilDiv(workgroupSize, SubgroupSize)+1)` and
# `-d:danger --opt:none --panics:on --threads:on --mm:arc -d:useMalloc -g`
# ...and debug with nim-gdb or lldb

import std/math, computesim

proc reduce(input: seq[int32], atomicSum: var int32; numElements: uint32) {.computeShader.} =
  let gid = gl_GlobalInvocationID.x
  let value = if gid < numElements: input[gid] else: 0

  # First reduce within subgroup using efficient subgroup operation
  let sum = subgroupAdd(value)

  # Only one thread per subgroup needs to add to global sum
  if gl_SubgroupInvocationID == 0:
    atomicAdd atomicSum, sum

const
  NumElements = 1024'u32
  WorkGroupSize = 256'u32

proc main() =
  # Set up compute dimensions
  let numWorkGroups = uvec3(ceilDiv(NumElements, WorkGroupSize), 1, 1)
  let workGroupSize = uvec3(WorkGroupSize, 1, 1)

  # Initialize buffers
  var buffers = (
    input: newSeq[int32](NumElements),
    atomicSum: 0'i32
  )
  for i in 0..<NumElements:
    buffers.input[i] = int32(i)

  # Run reduction on CPU
  runComputeOnCpu(numWorkGroups, workGroupSize, reduce,
    ssbo = addr buffers,
    args = NumElements
  )

  let result = buffers.atomicSum
  let expected = int32(NumElements * (NumElements - 1)) div 2
  echo "Reduction result: ", result, ", expected: ", expected

main()
