# computesim

A compute shader emulator for learning and debugging GPU compute shaders.

## Features
- Emulates GPU compute shader execution on CPU
- Simulates workgroups and subgroups with lockstep execution
- Supports GLSL subgroup operations
- Thread state visualization and debugging
- Works with any Nim code that follows compute shader patterns

## Example

```nim
# Compile with appropriate thread pool size and optimization settings
# -d:ThreadPoolSize=ceilDiv(workgroupSize, SubgroupSize)+1 -d:danger --threads:on --mm:arc

import std/[atomics, math], computesim

type
  Buffers = object
    input: seq[int32]
    sum: Atomic[int32]

proc reduce(buffers: ptr Buffers; numElements: uint32) {.computeShader.} =
  let gid = gl_GlobalInvocationID.x
  let value = if gid < numElements: buffers.input[gid] else: 0

  # First reduce within subgroup using efficient subgroup operation
  let sum = subgroupAdd(value)

  # Only one thread per subgroup needs to add to global sum
  if gl_SubgroupInvocationID == 0:
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
    args = NumElements
  )

  let result = buffers.sum.load(moRelaxed)
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

## License
MIT
