import std/strutils, core

proc formatValue*(t: ValueType, val: RawValue): string =
  case t
  of Bool: $getValue[bool](val)
  of Int: $getValue[int32](val)
  of Uint: $getValue[uint32](val)
  of Float: $getValue[float32](val)
  of Double: $getValue[float64](val)

proc formatValues*(group: SubgroupThreadIDs, t: ValueType, values: openArray[RawValue]): string =
  result = "["
  for threadId in threadsInGroup(group):
    result.addSep(", ", startLen = 1)
    result.add(formatValue(t, values[threadId]))
  result.add("]")

proc formatValuedThreads(group: SubgroupThreadIDs, commands: SubgroupCommands): string =
  result = "{"
  for threadId in threadsInGroup(group):
    result.addSep(", ", startLen = 1)
    result.addf("t$#: $#", threadId, formatValue(commands[threadId].t, commands[threadId].val))
  result.add("}")

proc formatBoolThreads(group: SubgroupThreadIDs, commands: SubgroupCommands): string =
  result = "{"
  for threadId in threadsInGroup(group):
    result.addSep(", ", startLen = 1)
    result.addf("t$#: $#", threadId, commands[threadId].bVal)
  result.add("}")

proc formatThreadList(group: SubgroupThreadIDs): string =
  result = "{"
  for threadId in threadsInGroup(group):
    result.addSep(", ", startLen = 1)
    result.addf("t$#", threadId)
  result.add("}")

proc formatThreadValues(group: SubgroupThreadIDs, commands: SubgroupCommands): string =
  let firstThreadId = group[0]
  let opKind = commands[firstThreadId].kind
  case opKind
  of subgroupBroadcast, subgroupBroadcastFirst, subgroupAdd, subgroupMin, subgroupMax,
      subgroupInclusiveAdd, subgroupExclusiveAdd, subgroupShuffle, subgroupShuffleXor:
    formatValuedThreads(group, commands)
  of subgroupBallot, subgroupAll, subgroupAny:
    formatBoolThreads(group, commands)
  else:
    formatThreadList(group)

proc debugSubgroupOp*(name: string, opId: uint32, group: SubgroupThreadIDs,
                      commands: SubgroupCommands, msg: string) =
  echo "[", name, " #", opId, "] inputs ", formatThreadValues(group, commands), " | ", msg
