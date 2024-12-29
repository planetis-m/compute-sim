# Compile with at least `-d:ThreadPoolSize=MaxConcurrentWorkGroups*
# (workgroupSizeX*workgroupSizeY*workgroupSizeZ+1)`

## ## Description
##
## `runComputeOnCpu` is a function that simulates a GPU-like compute environment on the CPU.
## It organizes work into workgroups and invocations, similar to how compute shaders operate
## on GPUs.
##
## ## Warning
## Using `barrier()` within conditional branches leads to undefined behavior. The emulator is
## modeled using a single barrier that must be accessible from all threads within a workgroup.
##
## ## Parameters
##
## - `numWorkGroups: UVec3` The number of workgroups in each dimension (x, y, z).
## - `workGroupSize: UVec3` The size of each workgroup in each dimension (x, y, z).
## - `compute: ComputeProc[A, B, C]` The compute shader procedure to execute.
## - `ssbo: A` Storage buffer object(s) containing the data to process.
## - `smem: B` Shared memory for each workgroup.
## - `args: C` Additional arguments passed to the compute shader.
##
## ## Compute Function Signature
##
## The compute shader procedure should have the following signature:
##
## ```nim
## proc computeFunction[A, B, C](
##   env: GlEnvironment,
##   buffers: A,     # Storage buffer (typically ptr T or Locker[T])
##   shared: ptr B,  # Shared memory
##   args: C         # Additional arguments
## ) {.nimcall.}
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
##     env: GlEnvironment,
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

import std/math, threading/barrier, malebolgia

import computesim/[core, vectors, transform, lockstep]
export vectors, transform

type
  GlEnvironment* = object
    gl_GlobalInvocationID*: UVec3    # Global ID of the current invocation
    gl_LocalInvocationID*: UVec3     # Local ID within the workgroup
    gl_WorkGroupID*: UVec3           # ID of the current workgroup
    gl_WorkGroupSize*: UVec3         # Size of the workgroup (x, y, z)
    gl_NumWorkGroups*: UVec3         # Total number of workgroups (x, y, z)
    gl_NumSubgroups*: uint32         # Number of subgroups in the workgroup
    gl_SubgroupSize*: uint32         # Size of each subgroup
    gl_SubgroupID*: uint32           # ID of the current subgroup [0..gl_NumSubgroups)
    gl_SubgroupInvocationID*: uint32 # ID of the invocation within the subgroup [0..gl_SubgroupSize)

  ComputeProc*[A, B, C] = proc (
    env: GlEnvironment,
    buffers: A,
    shared: ptr B,
    args: C
  ) {.nimcall.}


const
  MaxConcurrentWorkGroups {.intdefine.} = 2

proc wrapCompute[A, B, C](env: GlEnvironment, barrier: BarrierHandle,
    compute: ComputeProc[A, B, C], buffers: A, shared: ptr B, args: C) {.gcsafe.} =
  compute(env, barrier, buffers, shared, args)

proc workGroupProc[A, B, C](
    workgroupID: UVec3,
    env: GlEnvironment,
    compute: ComputeProc[A, B, C],
    ssbo: A, smem: ptr B, args: C) {.nimcall.} =
  # Auxiliary proc for work group management
  var env = env # Shadow for modification
  env.gl_WorkGroupID = workgroupID
  var smem = smem[] # Allocated per work group
  var barrier = createBarrier(
    env.gl_WorkGroupSize.x * env.gl_WorkGroupSize.y * env.gl_WorkGroupSize.z)
  # Create master for managing threads
  var master = createMaster(activeProducer = true)
  master.awaitAll:
    for z in 0 ..< env.gl_WorkGroupSize.z:
      for y in 0 ..< env.gl_WorkGroupSize.y:
        for x in 0 ..< env.gl_WorkGroupSize.x:
          env.gl_LocalInvocationID = uvec3(x, y, z)
          env.gl_GlobalInvocationID = uvec3(
            workGroupID.x * env.gl_WorkGroupSize.x + x,
            workGroupID.y * env.gl_WorkGroupSize.y + y,
            workGroupID.z * env.gl_WorkGroupSize.z + z
          )
          master.spawn wrapCompute(env, barrier.getHandle(), compute, ssbo, addr smem, args)

proc runComputeOnCpu*[A, B, C](
    numWorkGroups, workGroupSize: UVec3,
    compute: ComputeProc[A, B, C],
    ssbo: A, smem: B, args: C) =
  let env = GlEnvironment(
    gl_NumWorkGroups: numWorkGroups,
    gl_WorkGroupSize: workGroupSize
  )
  let totalGroups = numWorkGroups.x * numWorkGroups.y * numWorkGroups.z
  let numBatches = ceilDiv(totalGroups, MaxConcurrentWorkGroups)
  var currentGroup = 0
  # Initialize workgroup coordinates
  var wgX, wgY, wgZ: uint = 0
  # Process workgroups in batches to limit concurrent execution
  for batch in 0 ..< numBatches:
    let endGroup = min(currentGroup + MaxConcurrentWorkGroups, totalGroups.int)
    # Create master for managing work groups
    var master = createMaster(activeProducer = false) # not synchronized
    master.awaitAll:
      while currentGroup < endGroup:
        master.spawn workGroupProc(uvec3(wgX, wgY, wgZ), env, compute, ssbo, addr smem, args)
        # Increment coordinates, wrapping when needed
        inc wgX
        if wgX >= numWorkGroups.x:
          wgX = 0
          inc wgY
          if wgY >= numWorkGroups.y:
            wgY = 0
            inc wgZ
        inc currentGroup
