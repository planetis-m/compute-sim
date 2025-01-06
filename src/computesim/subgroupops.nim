# T subgroupBroadcast(T value, uint id)
# T subgroupBroadcastFirst(T value);

# T subgroupAdd(T value);
# T subgroupMin(T value);
# T subgroupMax(T value);

# T subgroupInclusiveAdd(T value);
# T subgroupExclusiveAdd(T value);

# T subgroupShuffle(T value, uint id);
# T subgroupShuffleXor(T value, uint mask);

# T subgroupShuffleDown(T value, uint delta);
# T subgroupShuffleUp(T value, uint delta);

# uvec4 subgroupBallot(bool condition);
# bool subgroupElect();
# bool subgroupAll(bool condition);
# bool subgroupAny(bool condition);

# (c) 2024 Antonis Geralis
import std/[fenv, strutils], core, debug, vectors

template defineSubgroupOp(op, body: untyped) {.dirty.} =
  proc op*(results: var SubgroupResults, commands: SubgroupCommands,
           group: SubgroupThreadIDs, firstThreadId, opId: uint32, showDebugOutput: bool) =
    body

defineSubgroupOp(execBroadcast):
  var broadcastVal = default(RawValue)
  # First find the source value
  let srcThreadId = commands[firstThreadId].dirty
  var found = false
  for threadId in threadsInGroup(group):
    if threadId == srcThreadId:
      broadcastVal = commands[threadId].val
      found = true
      break
  # If source thread not found, use first thread's value
  if not found:
    broadcastVal = commands[firstThreadId].val

  let valueType = commands[firstThreadId].t
  # Then broadcast to all threads
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupBroadcast,
      t: valueType,
      res: broadcastVal
    )

  if showDebugOutput:
    debugSubgroupOp("Broadcast", opId, group, commands,
      "broadcast " & formatValue(commands[firstThreadId].t, commands[firstThreadId].val) &
      " from thread: " & $srcThreadId)

defineSubgroupOp(execBroadcastFirst):
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupBroadcastFirst,
      t: commands[firstThreadId].t,
      res: commands[firstThreadId].val
    )

  if showDebugOutput:
    debugSubgroupOp("BroadcastFirst", opId, group, commands,
      "broadcast: " & formatValue(commands[firstThreadId].t, commands[firstThreadId].val))

defineSubgroupOp(execAdd):
  template sumValues(initVal: typed) =
    var total = initVal
    for threadId in threadsInGroup(group):
      total += getValue[typeof(initVal)](commands[threadId].val)
    sum = toValue(total)

  var sum = default(RawValue)
  let valueType = commands[firstThreadId].t
  case valueType:
  of Int:
    sumValues(0'i32)
  of Uint:
    sumValues(0'u32)
  of Float:
    sumValues(0'f32)
  of Double:
    sumValues(0'f64)
  else:
    discard # Support for vector types not implemented

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupAdd,
      t: valueType,
      res: sum
    )

  if showDebugOutput:
    debugSubgroupOp("Add", opId, group, commands, "sum: " & formatValue(valueType, sum))

defineSubgroupOp(execMax):
  template maxValues(initVal: typed) =
    var maxVal = initVal
    for threadId in threadsInGroup(group):
      maxVal = max(maxVal, getValue[typeof(initVal)](commands[threadId].val))
    maximum = toValue(maxVal)

  var maximum = default(RawValue)
  let valueType = commands[firstThreadId].t
  case valueType:
  of Int:
    maxValues(low(int32))
  of Uint:
    maxValues(0'u32)
  of Float:
    maxValues(-maximumPositiveValue(float32))
  of Double:
    maxValues(-maximumPositiveValue(float64))
  else:
    discard

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupMax,
      t: valueType,
      res: maximum
    )

  if showDebugOutput:
    debugSubgroupOp("Max", opId, group, commands, "max: " & formatValue(valueType, maximum))

defineSubgroupOp(execMin):
  template minValues(initVal: typed) =
    var minVal = initVal
    for threadId in threadsInGroup(group):
      minVal = min(minVal, getValue[typeof(initVal)](commands[threadId].val))
    minimum = toValue(minVal)

  var minimum = default(RawValue)
  let valueType = commands[firstThreadId].t
  case valueType:
  of Int:
    minValues(high(int32))
  of Uint:
    minValues(high(uint32))
  of Float:
    minValues(maximumPositiveValue(float32))
  of Double:
    minValues(maximumPositiveValue(float64))
  else:
    discard

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupMin,
      t: valueType,
      res: minimum
    )

  if showDebugOutput:
    debugSubgroupOp("Min", opId, group, commands, "min: " & formatValue(valueType, minimum))

defineSubgroupOp(execInclusiveAdd):
  var inclusiveSums {.noinit.}: array[SubgroupSize, RawValue]

  template inclusiveSum(initVal: typed) =
    var total = initVal
    for threadId in threadsInGroup(group):
      total += getValue[typeof(initVal)](commands[threadId].val)
      inclusiveSums[threadId] = toValue(total)

  let valueType = commands[firstThreadId].t
  case valueType:
  of Int:
    inclusiveSum(0'i32)
  of Uint:
    inclusiveSum(0'u32)
  of Float:
    inclusiveSum(0'f32)
  of Double:
    inclusiveSum(0'f64)
  else:
    discard

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupInclusiveAdd,
      t: valueType,
      res: inclusiveSums[threadId]
    )

  if showDebugOutput:
    debugSubgroupOp("InclusiveAdd", opId, group, commands,
      "prefix sums: " & formatValues(group, valueType, inclusiveSums))

defineSubgroupOp(execExclusiveAdd):
  var exclusiveSums {.noinit.}: array[SubgroupSize, RawValue]

  template exclusiveSum(initVal: typed) =
    var total = initVal
    for threadId in threadsInGroup(group):
      exclusiveSums[threadId] = toValue(total)
      total += getValue[typeof(initVal)](commands[threadId].val)

  let valueType = commands[firstThreadId].t
  case valueType:
  of Int:
    exclusiveSum(0'i32)
  of Uint:
    exclusiveSum(0'u32)
  of Float:
    exclusiveSum(0'f32)
  of Double:
    exclusiveSum(0'f64)
  else:
    discard

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupExclusiveAdd,
      t: valueType,
      res: exclusiveSums[threadId]
    )

  if showDebugOutput:
    debugSubgroupOp("ExclusiveAdd", opId, group, commands,
      "prefix sums: " & formatValues(group, valueType, exclusiveSums))

defineSubgroupOp(execShuffle):
  var shuffledVals {.noinit.}: array[SubgroupSize, RawValue]
  # First gather all shuffled values into array
  for threadId in threadsInGroup(group):
    let srcThreadId = commands[threadId].dirty
    # Check if source thread is valid within the group
    var found = false
    for validId in threadsInGroup(group):
      if validId == srcThreadId:
        found = true
        break
    # If source thread is valid, take its value
    # Otherwise use this thread's own value
    shuffledVals[threadId] = commands[if found: srcThreadId else: threadId].val

  let valueType = commands[firstThreadId].t
  # Then construct results
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupShuffle,
      t: valueType,
      res: shuffledVals[threadId]
    )

  if showDebugOutput:
    debugSubgroupOp("Shuffle", opId, group, commands,
      "shuffled: " & formatValues(group, valueType, shuffledVals))

defineSubgroupOp(execShuffleXor):
  var shuffledVals {.noinit.}: array[SubgroupSize, RawValue]
  # First gather all shuffled values into array
  for threadId in threadsInGroup(group):
    let srcThreadId = threadId xor commands[threadId].dirty
    # Check if source thread is valid within the group
    var found = false
    for validId in threadsInGroup(group):
      if validId == srcThreadId:
        found = true
        break
    # If source thread is valid, take its value
    # Otherwise use this thread's own value
    shuffledVals[threadId] = commands[if found: srcThreadId else: threadId].val

  let valueType = commands[firstThreadId].t
  # Then construct results
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupShuffleXor,
      t: valueType,
      res: shuffledVals[threadId]
    )

  if showDebugOutput:
    debugSubgroupOp("ShuffleXor", opId, group, commands,
      "shuffled: " & formatValues(group, valueType, shuffledVals))

defineSubgroupOp(execShuffleDown):
  var shuffledVals {.noinit.}: array[SubgroupSize, RawValue]
  for threadId in threadsInGroup(group):
    let srcThreadId = threadId + commands[threadId].dirty
    var found = false
    for validId in threadsInGroup(group):
      if validId == srcThreadId:
        found = true
        break
    shuffledVals[threadId] = commands[if found: srcThreadId else: threadId].val

  let valueType = commands[firstThreadId].t
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupShuffleDown,
      t: valueType,
      res: shuffledVals[threadId]
    )

  if showDebugOutput:
    debugSubgroupOp("ShuffleDown", opId, group, commands,
      "shuffled: " & formatValues(group, valueType, shuffledVals))

defineSubgroupOp(execShuffleUp):
  var shuffledVals {.noinit.}: array[SubgroupSize, RawValue]
  for threadId in threadsInGroup(group):
    # Convert to signed for safe subtraction
    let srcThreadId = if threadId >= commands[threadId].dirty:
      threadId - commands[threadId].dirty
    else:
      threadId
    var found = false
    for validId in threadsInGroup(group):
      if validId == srcThreadId:
        found = true
        break
    shuffledVals[threadId] = commands[if found: srcThreadId else: threadId].val

  let valueType = commands[firstThreadId].t
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupShuffleUp,
      t: valueType,
      res: shuffledVals[threadId]
    )

  if showDebugOutput:
    debugSubgroupOp("ShuffleUp", opId, group, commands,
      "shuffled: " & formatValues(group, valueType, shuffledVals))

defineSubgroupOp(execBallot):
  var ballot: uint32 = 0 # Use uint32 as the ballot mask
  # Set bits in ballot mask based on each thread's boolean value
  for threadId in threadsInGroup(group):
    if commands[threadId].bVal:
      ballot = ballot or (1'u32 shl threadId)

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupBallot,
      t: Uint,
      res: toValue(ballot)
    )

  if showDebugOutput:
    debugSubgroupOp("Ballot", opId, group, commands, "ballot: " & toBin(ballot.int, SubgroupSize))

defineSubgroupOp(execElect):
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupElect,
      bRes: threadId == firstThreadId
    )

  if showDebugOutput:
    debugSubgroupOp("Elect", opId, group, commands, "elected: t" & $firstThreadId)

defineSubgroupOp(execAll):
  var allTrue = true
  for threadId in threadsInGroup(group):
    if not commands[threadId].bVal:
      allTrue = false
      break

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupAll,
      bRes: allTrue
    )

  if showDebugOutput:
    debugSubgroupOp("All", opId, group, commands, "all: " & $allTrue)

defineSubgroupOp(execAny):
  var anyTrue = false
  for threadId in threadsInGroup(group):
    if commands[threadId].bVal:
      anyTrue = true
      break

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupAny,
      bRes: anyTrue
    )

  if showDebugOutput:
    debugSubgroupOp("Any", opId, group, commands, "any: " & $anyTrue)

defineSubgroupOp(execAllEqual):
  var allEqual = true
  template compareValues(t: untyped) =
    let firstVal = getValue[t](commands[firstThreadId].val)
    for threadId in threadsInGroup(group):
      if getValue[t](commands[threadId].val) != firstVal:
        allEqual = false
        break

  let valueType = commands[firstThreadId].t
  case valueType:
  of Bool:
    compareValues(bool)
  of Int:
    compareValues(int32)
  of Uint:
    compareValues(uint32)
  of Float:
    compareValues(float32)
  of Double:
    compareValues(float64)

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupAllEqual,
      bRes: allEqual
    )

  if showDebugOutput:
    debugSubgroupOp("AllEqual", opId, group, commands, "allEqual: " & $allEqual)

defineSubgroupOp(execSubBarrier):
  # For barrier, just mark that each thread participated
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupBarrier
    )

  if showDebugOutput:
    debugSubgroupOp("SubBarrier", opId, group, commands, "subgroup sync")

defineSubgroupOp(execBarrier):
  # For barrier, just mark that each thread participated
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: barrier
    )

  if showDebugOutput:
    debugSubgroupOp("Barrier", opId, group, commands, "workgroup sync")
