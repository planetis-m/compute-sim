import vectors

const
  SubgroupOpError = "This function can only be used inside a proc marked with {.computeShader.}"

template subgroupBroadcast*[T](value: T; id: uint32): T =
  ## Broadcasts value from thread with specified id to all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupBroadcastFirst*[T](value: T): T =
  ## Broadcasts value from first active thread to all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupAdd*[T](value: T): T =
  ## Returns sum of value across all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupMin*[T](value: T): T =
  ## Returns minimum of value across all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupMax*[T](value: T): T =
  ## Returns maximum of value across all threads in subgroup
  {.error: SubgroupOpError.}

template subgroupInclusiveAdd*[T](value: T): T =
  ## Returns inclusive prefix sum of value for current thread
  {.error: SubgroupOpError.}

template subgroupExclusiveAdd*[T](value: T): T =
  ## Returns exclusive prefix sum of value for current thread
  {.error: SubgroupOpError.}

template subgroupShuffle*[T](value: T; id: uint32): T =
  ## Returns value from thread with specified id
  {.error: SubgroupOpError.}

template subgroupShuffleXor*[T](value: T; mask: uint32): T =
  ## Returns value from thread with id equal to current_id XOR mask
  {.error: SubgroupOpError.}

template subgroupShuffleDown*[T](value: T; delta: uint32): T =
  ## Returns value from thread with index current_id + delta
  {.error: SubgroupOpError.}

template subgroupShuffleUp*[T](value: T; delta: uint32): T =
  ## Returns value from thread with index current_id - delta
  {.error: SubgroupOpError.}

template subgroupBallot*(condition: bool): UVec4 =
  ## Returns bitmap of which threads have condition true
  {.error: SubgroupOpError.}

template subgroupElect*(): bool =
  ## Returns true for exactly one active thread in subgroup
  {.error: SubgroupOpError.}

template subgroupAll*(condition: bool): bool =
  ## Returns true if condition is true for all active threads
  {.error: SubgroupOpError.}

template subgroupAny*(condition: bool): bool =
  ## Returns true if condition is true for any active thread
  {.error: SubgroupOpError.}

template subgroupBarrier*() =
  ## Synchronizes all threads within the current subgroup
  {.error: SubgroupOpError.}

template barrier*() =
  ## Synchronizes all threads within the current workgroup
  {.error: SubgroupOpError.}
