import std/math, computesim

proc deadlock(dummy1, dummy2: uint32) {.computeShader.} =
  # Case 1: Unsafe - shows silent subgroup deadlock
  when false:
    if gl_SubgroupID == 1: # Second subgroup
      barrier() # Will deadlock since subgroup 0 has already finished

  # Case 2: Unsafe because barrier in divergent thread path
  when false:
    if gl_LocalInvocationID.x == 1: # Only thread 1 hits barrier
      barrier() # Will deadlock since other threads in same subgroup skip it

  # Case 3: Unsafe because barriers in different branches
  when false:
    if gl_LocalInvocationID.x mod 2 == 0:
      barrier() # Some threads hit this barrier
    else:
      barrier() # While others hit this one

  # Case 4: Unsafe some threads complete before barrier
  when false:
    if gl_SubgroupInvocationID >= 4:
      return # These threads try to exit early
    else:
      barrier() # While others hit a barrier

const
  WorkGroupSize = 16'u32

proc main() =
  # Run reduction on CPU
  runComputeOnCpu(
    numWorkGroups = uvec3(1, 1, 1),
    workGroupSize = uvec3(WorkGroupSize, 1, 1),
    compute = deadlock,
    ssbo = 1'u32,
    args = 2'u32
  )

main()
