# (c) 2024 Antonis Geralis
import core, subgroupops

proc raiseDeadlockError(threadsAtBarrier: uint32) {.noinline, noreturn.} =
  raise newException(AssertionDefect,
    "Invalid shader: Deadlock detected - some threads terminated while others " &
    "are still waiting at a barrier. Threads at barrier: " & $threadsAtBarrier)

proc raiseNonUniformBarrierError(id1, id2: uint32) {.noinline, noreturn.} =
  raise newException(AssertionDefect,
    "Invalid shader: Barrier must be uniformly executed by all threads in a workgroup. " &
    "Found different barrier IDs: " & $id1 & " and " & $id2)

type
  ThreadState = enum
    running, halted, atSubBarrier, atBarrier, finished

proc runThreads*(threads: SubgroupThreads; b: BarrierHandle) =
  var
    anyThreadsActive = true
    allThreadsHalted = false
    threadStates: array[SubgroupSize, ThreadState]
    commands: SubgroupCommands
    results: SubgroupResults
    minReconvergeId: uint32 = 0
    barrierId = InvalidId
    activeThreadCount: uint32 = SubgroupSize # todo: make a parameter
    barrierThreadCount: uint32 = 0

  template canReconverge(): bool =
    (allThreadsHalted and commands[threadId].id == minReconvergeId)

  template canPassBarrier(): bool =
    (barrierThreadCount == activeThreadCount and commands[threadId].id == barrierId)

  # Run until all threads are done
  while anyThreadsActive:
    # Run each active thread once
    for threadId in 0..<SubgroupSize:
      if threadStates[threadId] != finished:
        if threadStates[threadId] == running or canReconverge or canPassBarrier:
          commands[threadId] = threads[threadId](results[threadId])
          if finished(threads[threadId]):
            threadStates[threadId] = finished
          elif commands[threadId].kind == barrier:
            threadStates[threadId] = atBarrier
          elif commands[threadId].kind == subgroupBarrier:
            threadStates[threadId] = atSubBarrier
          elif commands[threadId].kind == reconverge:
            threadStates[threadId] = halted
          else:
            threadStates[threadId] = running

    # Handle thread states
    anyThreadsActive = false
    allThreadsHalted = true
    minReconvergeId = InvalidId
    barrierId = InvalidId
    activeThreadCount = SubgroupSize
    barrierThreadCount = 0

    # First pass - handle barrier counts and checks
    for threadId in 0..<SubgroupSize:
      if threadStates[threadId] == atBarrier:
        inc barrierThreadCount
        if barrierId == InvalidId:
          barrierId = commands[threadId].id
        elif barrierId != commands[threadId].id:
          raiseNonUniformBarrierError(barrierId, commands[threadId].id)

    # Second pass - handle other thread states
    for threadId in 0..<SubgroupSize:
      if threadStates[threadId] != finished:
        anyThreadsActive = true
      case threadStates[threadId]
      of running:
        allThreadsHalted = false
      of halted, atSubBarrier:
        minReconvergeId = min(minReconvergeId, commands[threadId].id)
      of atBarrier: discard # already handled
      of finished:
        if barrierThreadCount > 0: # If any threads are waiting at a barrier
          raiseDeadlockError(barrierThreadCount)
        dec activeThreadCount

    # Group matching operations
    var
      threadGroups: array[SubgroupSize, SubgroupThreadIDs]
      numGroups: uint32 = 0

    # Group by operation id
    for threadId in 0..<SubgroupSize:
      if threadStates[threadId] != finished and
          (threadStates[threadId] == running or
          (threadStates[threadId] == atSubBarrier and canReconverge) or canPassBarrier):
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
        execSubgroupOp(execSubBarrier)
      of barrier:
        # Wait for all threads in workgroup (outside subgroup) using barrier sync
        wait b
        execSubgroupOp(execBarrier)
      else:
        discard
