# (c) 2024 Antonis Geralis
from std/algorithm import fill
import core, subgroupops, vectors

const
  debugWorkgroupX {.intdefine.} = 0
  debugWorkgroupY {.intdefine.} = 0
  debugWorkgroupZ {.intdefine.} = 0
  debugSubgroupID {.intdefine.} = 0

template shouldShowDebugOutput(debug: untyped) =
  let debug = when defined(debugSubgroup):
    workgroupID.x == debugWorkgroupX and
    workgroupID.y == debugWorkgroupY and
    workgroupID.z == debugWorkgroupZ and
    subgroupID == debugSubgroupID
  else:
    false

proc raiseDeadlockError(workgroupID: UVec3; subgroupID, threadsAtBarrier,
                        activeThreads: uint32) {.noinline, noreturn.} =
  raise newException(AssertionDefect,
    "Invalid shader: Deadlock detected in workgroup " & $workgroupID & ", subgroup " & $subgroupID & ". " &
    $threadsAtBarrier & " of " & $activeThreads & " active threads are waiting at barrier.")

proc raiseNonUniformBarrierError(workgroupID: UVec3; subgroupID, id1, id2: uint32) {.noinline, noreturn.} =
  raise newException(AssertionDefect,
    "Invalid shader: Barrier must be uniformly executed by all threads in workgroup " &
    $workgroupID & ", subgroup " & $subgroupID & ". Found different barrier IDs: " & $id1 & " and " & $id2)

type
  ThreadState = enum
    running, halted, atSubBarrier, atBarrier, finished

proc runThreads*(threads: SubgroupThreads, numActiveThreads: uint32; workgroupID: UVec3;
                 subgroupID: uint32; b: BarrierHandle) =
  var
    anyThreadsActive = true
    allThreadsHalted = false
    threadStates {.noinit.}: array[SubgroupSize, ThreadState]
    commands {.noinit.}: SubgroupCommands
    results {.noinit.}: SubgroupResults
    minReconvergeId: uint32 = 0
    barrierId = InvalidId
    barrierThreadCount: uint32 = 0

  shouldShowDebugOutput(showDebugOutput)
  threadStates.fill(running)

  template canReconverge(): bool =
    (allThreadsHalted and minReconvergeId < barrierId and commands[threadId].id == minReconvergeId)

  template canPassBarrier(): bool =
    (barrierThreadCount == numActiveThreads and commands[threadId].id == barrierId)

  # Run until all threads are done
  while anyThreadsActive:
    # Run each active thread once
    var madeProgress = false
    for threadId in 0..<numActiveThreads:
      if threadStates[threadId] != finished and
          threadStates[threadId] == running or canReconverge or canPassBarrier:
        madeProgress = true
        {.cast(gcsafe).}:
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
    barrierThreadCount = 0

    for threadId in 0..<numActiveThreads:
      if threadStates[threadId] != finished:
        anyThreadsActive = true
      case threadStates[threadId]
      of running:
        allThreadsHalted = false
      of halted, atSubBarrier:
        minReconvergeId = min(minReconvergeId, commands[threadId].id)
      of atBarrier:
        inc barrierThreadCount
        if barrierId == InvalidId:
          barrierId = commands[threadId].id
        elif barrierId != commands[threadId].id:
          raiseNonUniformBarrierError(workgroupID, subgroupID, barrierId, commands[threadId].id)
      of finished:
        discard

    if not madeProgress: # No thread could execute this iteration
      raiseDeadlockError(workgroupID, subgroupID, barrierThreadCount, numActiveThreads)

    # Group matching operations
    var
      threadGroups {.noinit.}: array[SubgroupSize, SubgroupThreadIDs]
      numGroups: uint32 = 0

    # Group by operation id
    for threadId in 0..<numActiveThreads:
      if threadStates[threadId] != finished and
          (threadStates[threadId] == running or
          (threadStates[threadId] == atSubBarrier and canReconverge) or canPassBarrier):
        var found = false
        for groupIdx in 0..<numGroups:
          let firstThreadId = threadGroups[groupIdx][0]
          if commands[firstThreadId].id == commands[threadId].id:
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

    template execSubgroupOp(op: untyped) =
      op(results, commands, threadGroups[groupIdx], firstThreadId, opId, showDebugOutput)

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
      of subgroupShuffleDown:
        execSubgroupOp(execShuffleDown)
      of subgroupShuffleUp:
        execSubgroupOp(execShuffleUp)
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
        # Note: Will deadlock silently if any subgroups have already completed
        wait b
        execSubgroupOp(execBarrier)
      else:
        discard
