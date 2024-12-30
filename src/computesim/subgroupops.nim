# T subgroupBroadcast(T value, uint id)
# T subgroupBroadcastFirst(T value);

# T subgroupAdd(T value);
# T subgroupMin(T value);
# T subgroupMax(T value);

# T subgroupInclusiveAdd(T value);
# T subgroupExclusiveAdd(T value);

# T subgroupShuffle(T value, uint id);
# T subgroupShuffleXor(T value, uint mask);

# uvec4 subgroupBallot(bool condition);
# bool subgroupElect();
# bool subgroupAll(bool condition);
# bool subgroupAny(bool condition);

import std/[fenv, strutils], core, debug

proc execBroadcast*(results: var SubgroupResults, commands: SubgroupCommands,
                    group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  var broadcastVal: RawValue
  # First find the source value
  let srcThreadId = commands[firstThreadId].dirty
  var found = false
  for threadId in threadsInGroup(group):
    if threadId == srcThreadId:
      broadcastVal = commands[threadId].value
      found = true
      break
  # If source thread not found, use first thread's value
  if not found:
    broadcastVal = commands[firstThreadId].value

  let valueType = commands[firstThreadId].t
  # Then broadcast to all threads
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupBroadcast,
      t: valueType,
      res: broadcastVal
    )

  when defined(debugSubgroup):
    debugSubgroupOp("Broadcast", opId, group, commands,
      "broadcast " & formatValue(commands[firstThreadId].t, commands[firstThreadId].value) &
      " from thread: " & $srcThreadId)

proc execBroadcastFirst*(results: var SubgroupResults, commands: SubgroupCommands,
                         group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupBroadcastFirst,
      t: commands[firstThreadId].t,
      res: commands[firstThreadId].value
    )

  when defined(debugSubgroup):
    debugSubgroupOp("BroadcastFirst", opId, group, commands,
      "broadcast: " & formatValue(commands[firstThreadId].t, commands[firstThreadId].value))

proc execAdd*(results: var SubgroupResults, commands: SubgroupCommands,
              group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  template sumValues(initVal: typed) =
    var total = initVal
    for threadId in threadsInGroup(group):
      total += getValue[typeof(initVal)](commands[threadId].value)
    sum = toValue(total)

  var sum: RawValue
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

  when defined(debugSubgroup):
    debugSubgroupOp("Add", opId, group, commands, "sum: " & formatValue(valueType, sum))

proc execMax*(results: var SubgroupResults, commands: SubgroupCommands,
              group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  template maxValues(initVal: typed) =
    var maxVal = initVal
    for threadId in threadsInGroup(group):
      maxVal = max(maxVal, getValue[typeof(initVal)](commands[threadId].value))
    maximum = toValue(maxVal)

  var maximum: RawValue
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

  when defined(debugSubgroup):
    debugSubgroupOp("Max", opId, group, commands, "max: " & formatValue(valueType, maximum))

proc execMin*(results: var SubgroupResults, commands: SubgroupCommands,
              group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  template minValues(initVal: typed) =
    var minVal = initVal
    for threadId in threadsInGroup(group):
      minVal = min(minVal, getValue[typeof(initVal)](commands[threadId].value))
    minimum = toValue(minVal)

  var minimum: RawValue
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

  when defined(debugSubgroup):
    debugSubgroupOp("Min", opId, group, commands, "min: " & formatValue(valueType, minimum))

proc execInclusiveAdd*(results: var SubgroupResults, commands: SubgroupCommands,
                       group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  var inclusiveSums: array[SubgroupSize, RawValue]

  template inclusiveSum(initVal: typed) =
    var total = initVal
    for threadId in threadsInGroup(group):
      total += getValue[typeof(initVal)](commands[threadId].value)
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

  when defined(debugSubgroup):
    debugSubgroupOp("InclusiveAdd", opId, group, commands,
      "prefix sums: " & formatValues(group, valueType, inclusiveSums))

proc execExclusiveAdd*(results: var SubgroupResults, commands: SubgroupCommands,
                       group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  var exclusiveSums: array[SubgroupSize, RawValue]

  template exclusiveSum(initVal: typed) =
    var total = initVal
    for threadId in threadsInGroup(group):
      exclusiveSums[threadId] = toValue(total)
      total += getValue[typeof(initVal)](commands[threadId].value)

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

  when defined(debugSubgroup):
    debugSubgroupOp("ExclusiveAdd", opId, group, commands,
      "prefix sums: " & formatValues(group, valueType, exclusiveSums))

proc execShuffle*(results: var SubgroupResults, commands: SubgroupCommands,
                  group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  var shuffledVals: array[SubgroupSize, RawValue]
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
    shuffledVals[threadId] = commands[if found: srcThreadId else: threadId].value

  let valueType = commands[firstThreadId].t
  # Then construct results
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupShuffle,
      t: valueType,
      res: shuffledVals[threadId]
    )

  when defined(debugSubgroup):
    debugSubgroupOp("Shuffle", opId, group, commands,
      "shuffled: " & formatValues(group, valueType, shuffledVals))

proc execShuffleXor*(results: var SubgroupResults, commands: SubgroupCommands,
                  group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  var shuffledVals: array[SubgroupSize, RawValue]
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
    shuffledVals[threadId] = commands[if found: srcThreadId else: threadId].value

  let valueType = commands[firstThreadId].t
  # Then construct results
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupShuffleXor,
      t: valueType,
      res: shuffledVals[threadId]
    )

  when defined(debugSubgroup):
    debugSubgroupOp("Shuffle", opId, group, commands,
      "shuffled: " & formatValues(group, valueType, shuffledVals))

proc execBallot*(results: var SubgroupResults, commands: SubgroupCommands,
                 group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  var ballot: uint32 = 0 # Use uint32 as the ballot mask
  # Set bits in ballot mask based on each thread's boolean value
  for threadId in threadsInGroup(group):
    if commands[threadId].bValue:
      ballot = ballot or (1'u32 shl threadId)

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupBallot,
      t: Uint,
      res: toValue(ballot)
    )

  when defined(debugSubgroup):
    debugSubgroupOp("Ballot", opId, group, commands, "ballot: " & toBin(ballot.int, SubgroupSize))

proc execElect*(results: var SubgroupResults, commands: SubgroupCommands,
                group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupElect,
      bRes: threadId == firstThreadId
    )

  when defined(debugSubgroup):
    debugSubgroupOp("Elect", opId, group, commands, "elected: t" & $firstThreadId)

proc execAll*(results: var SubgroupResults, commands: SubgroupCommands,
              group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  var allTrue = true
  for threadId in threadsInGroup(group):
    if commands[threadId].bValue == false:
      allTrue = false
      break

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupAll,
      bRes: allTrue
    )

  when defined(debugSubgroup):
    debugSubgroupOp("All", opId, group, commands, "all: " & $allTrue)

proc execAny*(results: var SubgroupResults, commands: SubgroupCommands,
              group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  var anyTrue = false
  for threadId in threadsInGroup(group):
    if commands[threadId].bValue == true:
      anyTrue = true
      break

  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupAny,
      bRes: anyTrue
    )

  when defined(debugSubgroup):
    debugSubgroupOp("Any", opId, group, commands, "any: " & $anyTrue)
