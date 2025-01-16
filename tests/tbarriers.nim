import std/math, computesim

proc calculate(output: ptr seq[int32]; numElements: uint32) {.computeShader.} =
  let tid = gl_GlobalInvocationID.x
  let localId = gl_LocalInvocationID.x
  let workgroupSize = gl_WorkGroupSize.x
  # Only proceed if within bounds
  # if tid < numElements:
  var value = tid.int32
  # First phase - each thread stores its ID
  output[tid] = value
  # Barrier to ensure all threads in workgroup have written
  barrier()
  # Second phase - read value from next position within same workgroup
  # But only if we're not at the workgroup boundary
  if localId + 1 < workgroupSize:
    value = output[tid + 1]
    value = subgroupBroadcastFirst(value)
  # Barrier before final write
  barrier()
  # Final phase - store the result
  output[tid] = value

const
  NumElements = 64'u32
  WorkGroupSize = 32'u32 # Force underutilization of hardware subgroups

proc main() =
  # Set up compute dimensions
  let numWorkGroups = uvec3(ceilDiv(NumElements, WorkGroupSize), 1, 1)
  let workGroupSize = uvec3(WorkGroupSize, 1, 1)

  # Initialize buffer
  let output = newSeq[int32](NumElements)

  # Run reduction on CPU
  runComputeOnCpu(
    numWorkGroups = numWorkGroups,
    workGroupSize = workGroupSize,
    compute = calculate,
    ssbo = addr output,
    args = NumElements
  )
  assert output == @[1'i32, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9,
      17, 17, 17, 17, 17, 17, 17, 17, 25, 25, 25, 25, 25, 25, 25, 31,
      33, 33, 33, 33, 33, 33, 33, 33, 41, 41, 41, 41, 41, 41, 41, 41,
      49, 49, 49, 49, 49, 49, 49, 49, 57, 57, 57, 57, 57, 57, 57, 63]

# Debug Output:
# - SubgroupID 0
# [Barrier #0] inputs {t0, t1, t2, t3, t4, t5, t6, t7} | workgroup sync
# [BroadcastFirst #1] inputs {t0: 1, t1: 2, t2: 3, t3: 4, t4: 5, t5: 6, t6: 7, t7: 8} | broadcast: 1
# [Barrier #3] inputs {t0, t1, t2, t3, t4, t5, t6, t7} | workgroup sync
# - SubgroupID 1
# [Barrier #0] inputs {t0, t1, t2, t3, t4, t5, t6, t7} | workgroup sync
# [BroadcastFirst #1] inputs {t0: 9, t1: 10, t2: 11, t3: 12, t4: 13, t5: 14, t6: 15, t7: 16} | broadcast: 9
# [Barrier #3] inputs {t0, t1, t2, t3, t4, t5, t6, t7} | workgroup sync
# - SubgroupID 2
# [Barrier #0] inputs {t0, t1, t2, t3, t4, t5, t6, t7} | workgroup sync
# [BroadcastFirst #1] inputs {t0: 17, t1: 18, t2: 19, t3: 20, t4: 21, t5: 22, t6: 23, t7: 24} | broadcast: 17
# [Barrier #3] inputs {t0, t1, t2, t3, t4, t5, t6, t7} | workgroup sync
# Output Buffer:
# [1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 17, 17, 17, 17, 17, 17, 17, 17, ...
# - Matches GLSL shader output.
main()
