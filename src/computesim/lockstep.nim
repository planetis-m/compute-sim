import core, subgroupops

type
  ThreadState = enum
    running, halted, finished

proc runThreads*(threads: SubgroupThreads) =
  var
    anyThreadsActive = true
    allThreadsHalted = false
    threadStates: array[SubgroupSize, ThreadState]
    commands: SubgroupCommands
    results: SubgroupResults
    minReconvergeId: uint32 = 0

  # Run until all threads are done
  while anyThreadsActive:
    # Run each active thread once
    for threadId in 0..<SubgroupSize:
      if threadStates[threadId] != finished:
        if threadStates[threadId] == running or
            (allThreadsHalted and commands[threadId].id == minReconvergeId):
          commands[threadId] = threads[threadId](results[threadId])
          if finished(threads[threadId]):
            threadStates[threadId] = finished
          elif commands[threadId].kind == reconverge:
            threadStates[threadId] = halted
          else:
            threadStates[threadId] = running

    anyThreadsActive = false
    allThreadsHalted = true
    minReconvergeId = InvalidId
    # Handle thread states
    for threadId in 0..<SubgroupSize:
      if threadStates[threadId] != finished:
        anyThreadsActive = true
      if threadStates[threadId] == running:
        allThreadsHalted = false
      if threadStates[threadId] == halted:
        minReconvergeId = min(minReconvergeId, commands[threadId].id)

    # Group matching operations
    var
      threadGroups: array[SubgroupSize, SubgroupThreadIDs]
      numGroups: uint32 = 0

    # Group by operation id
    for threadId in 0..<SubgroupSize:
      if threadStates[threadId] == running:
        var found = false
        for groupIdx in 0..<numGroups:
          let firstThreadId = threadGroups[groupIdx][0]
          if commands[firstThreadId].id == commands[threadId].id:
            # assert commands[firstThreadId].kind == commands[threadId].kind
            # Find first empty slot in group
            for slot in 0..<SubgroupSize:
              if threadGroups[groupIdx][slot] == InvalidId:
                threadGroups[groupIdx][slot] = threadId
                if slot + 1 < SubgroupSize:
                  threadGroups[groupIdx][slot + 1] = InvalidId
                break
            found = true
            break
        if not found:
          threadGroups[numGroups][0] = threadId
          threadGroups[numGroups][1] = InvalidId
          inc numGroups

    # Process operation groups
    for groupIdx in 0..<numGroups:
      let firstThreadId = threadGroups[groupIdx][0]
      let opKind = commands[firstThreadId].kind
      let opId = commands[firstThreadId].id
      case opKind:
      of subgroupBroadcast:
        execBroadcast(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupBroadcastFirst:
        execBroadcastFirst(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupAdd:
        execAdd(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupMin:
        execMin(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupMax:
        execMax(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupInclusiveAdd:
        execInclusiveAdd(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupExclusiveAdd:
        execExclusiveAdd(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupShuffle:
        execShuffle(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupShuffleXor:
        execShuffleXor(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupBallot:
        execBallot(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupElect:
        execElect(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupAll:
        execAll(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      of subgroupAny:
        execAny(results, commands, threadGroups[groupIdx], firstThreadId, opId)
      else:
        discard
