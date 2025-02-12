import std/math, computesim

proc calculate(output: ptr seq[int32]; numElements: uint32) {.computeShader.} =
  let tid = gl_GlobalInvocationID.x
  var value: int32 = 0
  # Loop from 1 to 2
  for i in 1'i32..2:
    if (tid.int32 + i) mod 3 == 0: # skip iteration
      continue
    value = subgroupShuffle(tid.int32 + 1, tid.uint32 + i.uint32)
  # Store the result in the SSBO
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

  assert output == @[3'i32, 0, 0, 6, 0, 0, 1, 1, 0, 12, 0, 0, 15, 0, 9, 10,
                     0, 0, 21, 0, 0, 24, 0, 18, 27, 0, 0, 30, 0, 0, 25, 25,
                     0, 36, 0, 0, 39, 0, 33, 34, 0, 0, 45, 0, 0, 48, 0, 42,
                     51, 0, 0, 54, 0, 0, 49, 49, 0, 60, 0, 0, 63, 0, 57, 58]

# Debug Output:
# - SubgroupID 0
# [Shuffle #1] inputs {t0: 1, t1: 2, t3: 4, t4: 5, t6: 7, t7: 8} | shuffled: [2, 0, 5, 0, 8, 1]
# [Shuffle #1] inputs {t0: 1, t2: 3, t3: 4, t5: 6, t6: 7} | shuffled: [3, 0, 6, 0, 1]
# - SubgroupID 1
# [Shuffle #1] inputs {t1: 10, t2: 11, t4: 13, t5: 14, t7: 16} | shuffled: [11, 0, 14, 0, 0]
# [Shuffle #1] inputs {t0: 9, t1: 10, t3: 12, t4: 13, t6: 15, t7: 16} | shuffled: [0, 12, 0, 15, 9, 10]
# - SubgroupID 2
# [Shuffle #1] inputs {t0: 17, t2: 19, t3: 20, t5: 22, t6: 23} | shuffled: [0, 20, 0, 23, 0]
# [Shuffle #1] inputs {t1: 18, t2: 19, t4: 21, t5: 22, t7: 24} | shuffled: [0, 21, 0, 24, 18]
# Output Buffer:
# [3, 0, 0, 6, 0, 0, 1, 1, 0, 12, 0, 0, 15, 0, 9, 10, 0, 0, 21, 0, 0, 24, 0, 18, 27, 0, 0, ...
# - Matches GLSL shader output.
main()
