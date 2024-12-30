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
                threadGroups[groupIdx][slot] = threadId.uint32
                if slot + 1 < SubgroupSize:
                  threadGroups[groupIdx][slot + 1] = InvalidId
                break
            found = true
            break
        if not found:
          threadGroups[numGroups][0] = threadId.uint32
          threadGroups[numGroups][1] = InvalidId
          inc numGroups

    template execSubgroupOp(op: untyped) =
      op(results, commands, threadGroups[groupIdx], firstThreadId, opId)

    # Process operation groups
    for groupIdx in 0..<numGroups:
      let firstThreadId = threadGroups[groupIdx][0]
      let opKind = commands[firstThreadId].kind
      let opId = commands[firstThreadId].id
      case opKind:
      of subgroupBroadcast:
        execSubgroupOp(execBroadcast)
      of subgroupBroadcastFirst:
        execSubgroupOp(execBroadcastFirst)
      of subgroupAdd:
        execSubgroupOp(execAdd)
      of subgroupMin:
        execSubgroupOp(execMin)
      of subgroupMax:
        execSubgroupOp(execMax)
      of subgroupInclusiveAdd:
        execSubgroupOp(execInclusiveAdd)
      of subgroupExclusiveAdd:
        execSubgroupOp(execExclusiveAdd)
      of subgroupShuffle:
        execSubgroupOp(execShuffle)
      of subgroupShuffleXor:
        execSubgroupOp(execShuffleXor)
      of subgroupBallot:
        execSubgroupOp(execBallot)
      of subgroupElect:
        execSubgroupOp(execElect)
      of subgroupAll:
        execSubgroupOp(execAll)
      of subgroupAny:
        execSubgroupOp(execAny)
      of subgroupBarrier:
        execSubgroupOp(execBarrier)
      else:
        discard
