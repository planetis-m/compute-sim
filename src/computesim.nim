## ## Description
##
## `runComputeOnCpu` is a function that simulates a GPU-like compute environment on the CPU.
## It organizes work into workgroups and invocations, similar to how compute shaders operate
## on GPUs.
##
## .. warning:: The thread pool size must be at least *MaxConcurrentWorkGroups *
##    (ceilDiv(workgroupSizeX * workgroupSizeY * workgroupSizeZ, SubgroupSize) + 1)*.
##    Compile with: `-d:ThreadPoolSize=N` where N meets this requirement.
##
## .. warning:: Using `barrier()` within conditional branches may lead to undefined
##    behavior. The emulator is modeled using a single barrier that must be accessible
##    from all threads within a workgroup.
##
## ## Parameters
##
## - `numWorkGroups: UVec3` The number of workgroups in each dimension (x, y, z).
## - `workGroupSize: UVec3` The size of each workgroup in each dimension (x, y, z).
## - `compute: ThreadGenerator[A, B, C]` The compute shader procedure to execute.
## - `ssbo: A` Storage buffer object(s) containing the data to process.
## - `smem: B` Shared memory for each workgroup.
## - `args: C` Additional arguments passed to the compute shader.
##
## ## Compute Function Signature
##
## The compute shader procedure can be written in two ways:
##
## 1. With shared memory:
##
## ```nim
## proc computeFunction[A, B, C](
##   buffers: A,     # Storage buffer (typically ptr T)
##   shared: ptr B,  # Shared memory for workgroup-local data
##   args: C         # Additional arguments
## ) {.computeShader.}
## ```
##
## 2. Without shared memory:
##
## ```nim
## proc computeFunction[A, C](
##   buffers: A,     # Storage buffer (typically ptr T)
##   args: C         # Additional arguments
## ) {.computeShader.}
## ```
##
## ## Example
##
## ```nim
## type
##   Buffers = object
##     input, output: seq[float32]
##   Shared = seq[float32]
##   Args = object
##     factor: int32
##
## proc myComputeShader(
##     buffers: ptr Buffers,
##     shared: ptr Shared,
##     args: Args) {.computeShader.} =
##   # Computation logic here
##
## let numWorkGroups = uvec3(4, 1, 1)
## let workGroupSize = uvec3(256, 1, 1)
## var buffers: Buffers
## let coarseFactor = 4'i32
##
## runComputeOnCpu(
##   numWorkGroups, workGroupSize,
##   myComputeShader,
##   addr buffers,
##   newSeq[float32](workGroupSize.x),
##   Args(factor: coarseFactor)
## )
## ```
##
## ## CUDA to GLSL Translation Table
##
## | CUDA Concept | GLSL Equivalent | Description |
## |--------------|-----------------|-------------|
## | `blockDim` | `gl_WorkGroupSize` | The size of a thread block (CUDA) or work group (GLSL) |
## | `gridDim` | `gl_NumWorkGroups` | The size of the grid (CUDA) or the number of work groups (GLSL) |
## | `blockIdx` | `gl_WorkGroupID` | The index of the current block (CUDA) or work group (GLSL) |
## | `threadIdx` | `gl_LocalInvocationID` | The index of the current thread within its block (CUDA) or work group (GLSL) |
## | `blockIdx * blockDim + threadIdx` | `gl_GlobalInvocationID` | The global index of the current thread (CUDA) or invocation (GLSL) |

# (c) 2024 Antonis Geralis
import std/[isolation, math], threading/barrier, malebolgia

import computesim/[core, vectors, transform, typecopy, lockstep, api]
export vectors, transform, api, SubgroupSize

type
  GlEnvironment* = object
    gl_GlobalInvocationID*: UVec3    ## Global ID of the current invocation [0..gl_NumWorkGroups*gl_WorkGroupSize)
    gl_LocalInvocationID*: UVec3     ## Local ID within the workgroup [0..gl_WorkGroupSize)
    gl_WorkGroupID*: UVec3           ## ID of the current workgroup [0..gl_NumWorkGroups)
    gl_WorkGroupSize*: UVec3         ## Size of the workgroup (x, y, z)
    gl_NumWorkGroups*: UVec3         ## Total number of workgroups (x, y, z)
    gl_NumSubgroups*: uint32         ## Number of subgroups in the workgroup
    gl_SubgroupID*: uint32           ## ID of the current subgroup [0..gl_NumSubgroups)
    gl_SubgroupInvocationID*: uint32 ## ID of the invocation within the subgroup [0..gl_SubgroupSize)

  ThreadGenerator*[A, B, C] = proc (
    env: GlEnvironment,
    buffers: A,
    shared: ptr B,
    args: C
  ): ThreadClosure {.nimcall.}

const
  MaxConcurrentWorkGroups {.intdefine.} = 2

proc subgroupProc[A, B, C](env: GlEnvironment; numActiveThreads: uint32; barrier: BarrierHandle,
    compute: ThreadGenerator[A, B, C]; buffers: A; shared: ptr B; args: C) =
  var threads = default(SubgroupThreads)
  let startIdx = env.gl_SubgroupID * SubgroupSize
  var threadId: uint32 = 0
  # Initialize coordinates from startIdx
  var x = startIdx mod env.gl_WorkGroupSize.x
  var y = (startIdx div env.gl_WorkGroupSize.x) mod env.gl_WorkGroupSize.y
  var z = startIdx div (env.gl_WorkGroupSize.x * env.gl_WorkGroupSize.y)
  while threadId < numActiveThreads:
    var env = env # Shadow for modification
    env.gl_LocalInvocationID = uvec3(x, y, z)
    env.gl_GlobalInvocationID = uvec3(
      env.gl_WorkGroupID.x * env.gl_WorkGroupSize.x + x,
      env.gl_WorkGroupID.y * env.gl_WorkGroupSize.y + y,
      env.gl_WorkGroupID.z * env.gl_WorkGroupSize.z + z
    )
    env.gl_SubgroupInvocationID = threadId
    threads[threadId] = compute(env, buffers, shared, args)
    # Update coordinates
    inc x
    if x >= env.gl_WorkGroupSize.x:
      x = 0
      inc y
      if y >= env.gl_WorkGroupSize.y:
        y = 0
        inc z
    inc threadId
  # Run threads in lockstep
  runThreads(threads, numActiveThreads, env.gl_WorkGroupID, env.gl_SubgroupID, barrier)

proc workGroupProc[A, B, C](
    workgroupID: UVec3,
    env: GlEnvironment,
    compute: ThreadGenerator[A, B, C],
    ssbo: A, smem: ptr B, args: C) {.nimcall.} =
  # Auxiliary proc for work group management
  var env = env # Shadow for modification
  env.gl_WorkGroupID = workgroupID
  let threadsInWorkgroup = env.gl_WorkGroupSize.x * env.gl_WorkGroupSize.y * env.gl_WorkGroupSize.z
  let numSubgroups = ceilDiv(threadsInWorkgroup, SubgroupSize)
  env.gl_NumSubgroups = numSubgroups
  # Initialize local shared memory
  var barrier = createBarrier(numSubgroups)
  # Create master for managing threads
  var master = createMaster(activeProducer = true)
  var remainingThreads = threadsInWorkgroup
  master.awaitAll:
    for subgroupId in 0..<numSubgroups:
      env.gl_SubgroupID = subgroupId
      # Calculate number of active threads in this subgroup
      let threadsInSubgroup = min(remainingThreads, SubgroupSize)
      master.spawn subgroupProc(env, threadsInSubgroup, barrier.getHandle(), compute, ssbo, smem, args)
      dec remainingThreads, threadsInSubgroup

proc runCompute[A, B, C](
    numWorkGroups, workGroupSize: UVec3,
    compute: ThreadGenerator[A, B, C],
    ssbo: A, smem: B, args: C) =
  let env = GlEnvironment(
    gl_NumWorkGroups: numWorkGroups,
    gl_WorkGroupSize: workGroupSize
  )
  let totalGroups = numWorkGroups.x * numWorkGroups.y * numWorkGroups.z
  let numBatches = ceilDiv(totalGroups, MaxConcurrentWorkGroups)
  var currentGroup = 0
  # Initialize workgroup coordinates
  var wgX, wgY, wgZ: uint32 = 0
  # Create array of shared memory for concurrent workgroups
  var smemArr = arrayWith(default(B), MaxConcurrentWorkGroups)
  # Process workgroups in batches to limit concurrent execution
  for batch in 0 ..< numBatches:
    let endGroup = min(currentGroup + MaxConcurrentWorkGroups, totalGroups.int)
    # Create master for managing work groups
    var master = createMaster(activeProducer = false) # not synchronized
    # Initialize shared memory for this batch
    for i in 0 ..< min(MaxConcurrentWorkGroups, endGroup - currentGroup):
      copyInto(smemArr[i], smem)
    master.awaitAll:
      var groupIdx = 0
      while currentGroup < endGroup:
        master.spawn workGroupProc(uvec3(wgX, wgY, wgZ), env, compute, ssbo, addr smemArr[groupIdx], args)
        # Increment coordinates, wrapping when needed
        inc wgX
        if wgX >= numWorkGroups.x:
          wgX = 0
          inc wgY
          if wgY >= numWorkGroups.y:
            wgY = 0
            inc wgZ
        inc groupIdx
        inc currentGroup

template runComputeOnCpu*(
    numWorkGroups, workGroupSize: UVec3,
    compute, ssbo, smem, args: typed) =
  bind isolate, extract
  runCompute(numWorkGroups, workGroupSize, compute, ssbo, smem, args)

template runComputeOnCpu*(
    numWorkGroups, workGroupSize: UVec3,
    compute, ssbo, args: typed) =
  bind isolate, extract
  proc wrapCompute(env: GlEnvironment,
      buffers: typeof(ssbo), shared: ptr int32, argsInner: typeof(args)): ThreadClosure {.nimcall.} =
    compute(env, buffers, argsInner)
  runCompute(numWorkGroups, workGroupSize, wrapCompute, ssbo, 0, args)
