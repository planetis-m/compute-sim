import std/math, computesim

proc calculate(output: ptr seq[int32]; numElements: uint32) {.computeShader.} =
  let tid = gl_GlobalInvocationID.x
  var value: int32 = 0
  # Loop from 1 to 2
  for i in 1'i32..2:
    if (tid.int32 + i) mod 3 == 0: # skip iteration
      continue
    # Get the first value in the subgroup
    value = subgroupBroadcastFirst(tid.int32 + i)
  # Store the result in the SSBO
  output[tid] = value

const
  NumElements = 64'u32
  WorkGroupSize = 16'u32 # Force underutilization of hardware subgroups

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

  assert output == @[2'i32, 1, 2, 2, 1, 2, 2, 1, 10, 10, 10, 10, 10, 10, 10, 10,
    17, 19, 19, 17, 19, 19, 17, 19, 26, 25, 26, 26, 25, 26, 26, 25,
    34, 34, 34, 34, 34, 34, 34, 34, 41, 43, 43, 41, 43, 43, 41, 43,
    50, 49, 50, 50, 49, 50, 50, 49, 58, 58, 58, 58, 58, 58, 58, 58]

# Debug Output:
# - SubgroupID 0
# [BroadcastFirst #1] inputs {t0: 1, t1: 2, t3: 4, t4: 5, t6: 7, t7: 8} | broadcast: 1
# [BroadcastFirst #1] inputs {t0: 2, t2: 4, t3: 5, t5: 7, t6: 8} | broadcast: 2
# - SubgroupID 2
# [BroadcastFirst #1] inputs {t1: 10, t2: 11, t4: 13, t5: 14, t7: 16} | broadcast: 10
# [BroadcastFirst #1] inputs {t0: 10, t1: 11, t3: 13, t4: 14, t6: 16, t7: 17} | broadcast: 10
# - SubgroupID 3
# [BroadcastFirst #1] inputs {t0: 17, t2: 19, t3: 20, t5: 22, t6: 23} | broadcast: 17
# [BroadcastFirst #1] inputs {t1: 19, t2: 20, t4: 22, t5: 23, t7: 25} | broadcast: 19
# Output Buffer:
# [2, 1, 2, 2, 1, 2, 2, 1, 10, 10, 10, 10, 10, 10, 10, 10, 17, 19, 19, 17, 19, 19, 17, 19, ...
# - Matches GLSL shader output.
main()
