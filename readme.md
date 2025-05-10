# computesim

A compute shader simulator for learning and debugging GPU compute shaders.

## Features
- Emulates GPU compute shader execution on CPU
- Simulates workgroups and subgroups with lockstep execution
- Supports GLSL subgroup operations
- Thread state visualization and debugging
- Works with any Nim code that follows compute shader patterns

## Example

```nim
# Compile with appropriate thread pool size and optimization settings
# -d:ThreadPoolSize=MaxConcurrentWorkGroups*(ceilDiv(workgroupSize, SubgroupSize)+1)
# -d:danger --threads:on --mm:arc

import std/math, computesim

type
  Buffers = object
    input: seq[int32]
    atomicSum: int32

proc reduce(b: ptr Buffers; numElements: uint32) {.computeShader.} =
  let gid = gl_GlobalInvocationID.x
  let value = if gid < numElements: b.input[gid] else: 0

  # First reduce within subgroup using efficient subgroup operation
  let sum = subgroupAdd(value)

  # Only one thread per subgroup needs to add to global sum
  if gl_SubgroupInvocationID == 0:
    atomicAdd b.atomicSum, sum

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
    atomicSum: 0
  )
  for i in 0..<NumElements:
    buffers.input[i] = int32(i)

  # Run reduction on CPU
  runComputeOnCpu(
    numWorkGroups = numWorkGroups,
    workGroupSize = workGroupSize,
    compute = reduce,
    ssbo = addr buffers,
    args = NumElements
  )

  let result = buffers.atomicSum
  let expected = int32(NumElements * (NumElements - 1)) div 2
  echo "Reduction result: ", result, ", expected: ", expected

main()
```

The example demonstrates:
- Using subgroup operations for efficient reduction
- Automatic handling of divergent control flow
- Atomic operations for cross-workgroup communication
- Proper synchronization between threads

## Installation
```
nimble install computesim
```

## Usage

1. Write your shader using the `computeShader` macro which:
   - Transforms control flow for lockstep execution
   - Converts subgroup operations into commands
   - Handles thread synchronization

2. Configure execution:
   - Set up workgroup dimensions
   - Prepare data buffers and shared memory
   - Call `runComputeOnCpu` with your shader

See the examples directory for more patterns and use cases.

## Limitations
- Single wavefront/subgroup size
- Limited subset of GLSL/compute operations
- Performance is not representative of real GPU execution

> [!WARNING]
> ### Workgroup Scheduling
> While this simulator runs workgroups using CPU threads, real GPU compute shaders have no fairness guarantees between workgroups. This means your code might work correctly in this CPU simulator but fail on real GPU hardware where workgroups can execute in any order and with varying levels of parallelism. Do not rely on any assumptions about workgroup execution order or scheduling that might be true in this CPU simulator but not guaranteed on actual GPUs.

> [!WARNING]
> ### Maximal Reconvergence
> In Vulkan, the point at which diverging threads in a shader reconverge is not strictly guaranteed by default. To ensure that threads rejoin execution as early as possible after a conditional branch (this is "maximal reconvergence"), the application must enable the `VK_KHR_shader_maximal_reconvergence` Vulkan device extension, and the shader must declare the `MaximalReconvergenceKHR` SPIR-V capability (from the `SPV_KHR_maximal_reconvergence` SPIR-V extension). Without these, shaders might exhibit less optimal reconvergence, potentially leading to performance differences or unexpected behavior across hardware or drivers, as threads might not regroup where your shader logic implicitly expects.

## Compile-time Defines

### Thread Management
- `ThreadPoolSize` - Required. Must be at least `MaxConcurrentWorkGroups*(ceilDiv(workgroupSize, SubgroupSize)+1)`
- `SubgroupSize` - Size of each subgroup/wavefront (default: 8)
- `MaxConcurrentWorkGroups` - Maximum concurrent workgroups (default: 2)

### Debug Options
With `-d:debugSubgroup`, these control which workgroup/subgroup to debug:
- `debugWorkgroupX/Y/Z` - Workgroup coordinates to debug (default: 0)
- `debugSubgroupID` - Subgroup ID to debug (default: 0)

```nim
# Example: Configure thread pool and groups
nim c -d:ThreadPoolSize=8 -d:SubgroupSize=4 myshader.nim

# Example: Enable debugging for specific group
nim c -d:debugSubgroup -d:debugWorkgroupX=1 myshader.nim
```

## License
MIT
