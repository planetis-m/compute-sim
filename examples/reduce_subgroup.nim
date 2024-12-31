import std/[atomics, math], computesim

type
  Buffers = object
    input: seq[int32]
    sum: Atomic[int32]

proc reduce(env: GlEnvironment; buffers: ptr Buffers; smem: ptr int32; numElements: uint32) {.computeShader.} =
  let gid = env.gl_GlobalInvocationID.x
  let value = if gid < numElements: buffers.input[gid] else: 0

  # First reduce within subgroup using efficient subgroup operation
  let sum = subgroupAdd(value)

  # Only one thread per subgroup needs to add to global sum
  if env.gl_SubgroupInvocationID == 0:
    atomicInc buffers.sum, sum

const
  NumElements = 1024'u32
  WorkGroupSize = 256'u32

proc main() =
  # Set up compute dimensions
  let numWorkGroups = uvec3(ceilDiv(NumElements, WorkGroupSize), 1, 1)
  let workGroupSize = uvec3(WorkGroupSize, 1, 1)

  # Initialize buffers
  var buffers = Buffers(
    input: newSeq[int32](NumElements),
    sum: default(Atomic[int32])
  )
  for i in 0..<NumElements:
    buffers.input[i] = int32(i)

  # Run reduction on CPU
  runComputeOnCpu(
    numWorkGroups = numWorkGroups,
    workGroupSize = workGroupSize,
    compute = reduce,
    ssbo = addr buffers,
    smem = 0'i32, # unused
    args = NumElements
  )

  let result = buffers.sum.load(moRelaxed)
  let expected = int32(NumElements * (NumElements - 1)) div 2
  echo "Reduction result: ", result, " expected: ", expected

main()
