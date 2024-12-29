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

import core, debug

proc execBroadcastFirst*(results: var SubgroupResults, commands: SubgroupCommands,
                         group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupBroadcastFirst,
      t: commands[firstThreadId].t,
      res: commands[firstThreadId].val
    )
  when defined(debugSubgroup):
    debugSubgroupOp("BroadcastFirst", opId, group, commands,
      "broadcast: " & formatValue(commands[firstThreadId].t, commands[firstThreadId].val))

proc execAdd*(results: var SubgroupResults, commands: SubgroupCommands,
              group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  template sumValues(initVal: typed) =
    var total = initVal
    for threadId in threadsInGroup(group):
      total += getValue[typeof(initVal)](commands[threadId].val)
    sum = toValue(total)

  var sum: RawValue
  let valueType = commands[firstThreadId].t
  case valueType:
  of Int:
    sumValues(0i32)
  of Uint:
    sumValues(0u32)
  of Float:
    sumValues(0f32)
  of Double:
    sumValues(0f64)
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

proc execInclusiveAdd*(results: var SubgroupResults, commands: SubgroupCommands,
                       group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  var inclusiveSums: array[SubgroupSize, RawValue]

  template inclusiveSum(initVal: typed) =
    var total = initVal
    for threadId in threadsInGroup(group):
      total += getValue[typeof(initVal)](commands[threadId].val)
      inclusiveSums[threadId] = toValue(total)

  let valueType = commands[firstThreadId].t
  case valueType:
  of Int:
    inclusiveSum(0i32)
  of Uint:
    inclusiveSum(0u32)
  of Float:
    inclusiveSum(0f32)
  of Double:
    inclusiveSum(0f64)
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

proc execElect*(results: var SubgroupResults, commands: SubgroupCommands,
                group: SubgroupThreadIDs, firstThreadId, opId: uint32) =
  for threadId in threadsInGroup(group):
    results[threadId] = SubgroupResult(
      id: opId,
      kind: subgroupElect,
      bRes: threadId == firstThreadId
    )
  when defined(debugSubgroup):
    debugSubgroupOp("Elect", opId, group, commands, "elected: t" & firstThreadId)
