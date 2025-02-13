# (c) 2024 Antonis Geralis
import std/[macros, strutils], core, vectors

proc raiseInvalidSubgroupOp(kind: SubgroupOp) {.noinline, noreturn.} =
  raise newException(AssertionDefect, "Invalid subgroup operation: " & $kind)

template validateTwoArgOp(op: untyped) =
  if node.len != 3:
    error($op & " expects exactly two arguments", node)
  result = op

template validateOneArgOp(op: untyped) =
  if node.len != 2:
    error($op & " expects exactly one argument", node)
  result = op

template validateBoolOp(op: untyped) =
  if node.len != 2:
    error($op & " expects exactly one argument (bool condition)", node)
  result = op

template validateNoArgOp(op: untyped) =
  if node.len != 1:
    error($op & " expects no arguments", node)
  result = op

proc getSubgroupOp(node: NimNode): SubgroupOp =
  if node[0].kind != nnkIdent:
    return invalid

  case normalize(node[0].strVal)
  of "subgroupbroadcast":
    validateTwoArgOp(subgroupBroadcast)
  of "subgroupshuffle":
    validateTwoArgOp(subgroupShuffle)
  of "subgroupshufflexor":
    validateTwoArgOp(subgroupShuffleXor)
  of "subgroupshuffledown":
    validateTwoArgOp(subgroupShuffleDown)
  of "subgroupshuffleup":
    validateTwoArgOp(subgroupShuffleUp)
  of "subgroupbroadcastfirst":
    validateOneArgOp(subgroupBroadcastFirst)
  of "subgroupadd":
    validateOneArgOp(subgroupAdd)
  of "subgroupmin":
    validateOneArgOp(subgroupMin)
  of "subgroupmax":
    validateOneArgOp(subgroupMax)
  of "subgroupinclusiveadd":
    validateOneArgOp(subgroupInclusiveAdd)
  of "subgroupexclusiveadd":
    validateOneArgOp(subgroupExclusiveAdd)
  of "subgroupballot":
    validateBoolOp(subgroupBallot)
  of "subgroupall":
    validateBoolOp(subgroupAll)
  of "subgroupany":
    validateBoolOp(subgroupAny)
  of "subgroupallequal":
    validateOneArgOp(subgroupAllEqual)
  of "subgroupelect":
    validateNoArgOp(subgroupElect)
  of "subgroupbarrier":
    validateNoArgOp(subgroupBarrier)
  of "subgroupmemorybarrier":
    validateNoArgOp(subgroupMemoryBarrier)
  of "barrier":
    validateNoArgOp(barrier)
  of "memorybarrier":
    validateNoArgOp(memoryBarrier)
  of "groupmemorybarrier":
    validateNoArgOp(groupMemoryBarrier)
  else:
    result = invalid

template binaryOpCommand(cmdId, opKind, cmdVal, cmdParam: untyped): untyped =
  SubgroupCommand(id: cmdId, kind: opKind, t: getValueType(cmdVal), val: toValue(cmdVal), dirty: cmdParam)

template unaryOpCommand(cmdId, opKind, cmdVal: untyped): untyped =
  SubgroupCommand(id: cmdId, kind: opKind, t: getValueType(cmdVal), val: toValue(cmdVal))

template boolOpCommand(cmdId, opKind, cmdVal: untyped): untyped =
  SubgroupCommand(id: cmdId, kind: opKind, bVal: cmdVal)

template scalarOpResult(iterArg, cmdVal: untyped): untyped =
  getValue[typeof(cmdVal)](iterArg.res)

template ballotResult(iterArg: untyped): untyped =
  uvec4(getValue[uint32](iterArg.res), 0, 0, 0)

proc genSubgroupOpCall(op: SubgroupOp; node, id, iterArg: NimNode): NimNode =
  # Generate the command part based on operation type
  if defined(debugSubgroup):
    warning("Assigning ID " & $id.intVal & " to " & repr(node), node)
  let cmdPart = case op
    of subgroupBroadcast, subgroupShuffle, subgroupShuffleXor,
        subgroupShuffleDown, subgroupShuffleUp:
      getAst(binaryOpCommand(id, newLit(op), node[1], node[2]))
    of subgroupBroadcastFirst, subgroupAdd, subgroupMin, subgroupMax,
        subgroupInclusiveAdd, subgroupExclusiveAdd, subgroupAllEqual:
      getAst(unaryOpCommand(id, newLit(op), node[1]))
    of subgroupBallot, subgroupAll, subgroupAny:
      getAst(boolOpCommand(id, newLit(op), node[1]))
    of subgroupElect, subgroupBarrier, subgroupMemoryBarrier, barrier,
        memoryBarrier, groupMemoryBarrier:
      quote do:
        SubgroupCommand(id: `id`, kind: `op`)
    else: nil # cannot happen
  # Generate the result handling part
  let resultPart = case op
    of subgroupBroadcast, subgroupShuffle, subgroupShuffleXor,
        subgroupShuffleDown, subgroupShuffleUp,
        subgroupBroadcastFirst, subgroupAdd, subgroupMin, subgroupMax,
        subgroupInclusiveAdd, subgroupExclusiveAdd:
      getAst(scalarOpResult(iterArg, node[1]))
    of subgroupAll, subgroupAny, subgroupElect, subgroupAllEqual:
      newTree(nnkDotExpr, iterArg, ident"bRes")
    of subgroupBallot:
      getAst(ballotResult(iterArg))
    of subgroupBarrier, subgroupMemoryBarrier, barrier, memoryBarrier,
        groupMemoryBarrier:
      newTree(nnkDiscardStmt, newEmptyNode())
    else: nil
  # Combine both parts
  result = quote do:
    discard SubgroupOp(`op`)
    yield `cmdPart`
    case `iterArg`.kind
    of `op`:
      `resultPart`
    else:
      raiseInvalidSubgroupOp(`op`)
  result[1].copyLineInfo(node)
  result[2].copyLineInfo(node)

proc generateTemplates(sym: NimNode; fields: openArray[string]): NimNode =
  result = newNimNode(nnkStmtList)
  for field in fields:
    let fieldIdent = ident(field)
    result.add quote do:
      template `fieldIdent`(): untyped {.used.} = `sym`.`fieldIdent`

proc generateWorkGroupTemplates(wgSym: NimNode): NimNode =
  generateTemplates(wgSym, [
    "gl_WorkGroupID",
    "gl_WorkGroupSize",
    "gl_NumWorkGroups",
    "gl_NumSubgroups",
    "gl_SubgroupID"
  ])

proc generateThreadTemplates(threadSym: NimNode): NimNode =
  generateTemplates(threadSym, [
    "gl_GlobalInvocationID",
    "gl_LocalInvocationID"
  ])

proc isDiscard(n: NimNode, op: SubgroupOp): bool =
  n.kind == nnkDiscardStmt and n[0].kind == nnkCall and n[0][0].strVal == "SubgroupOp" and
    n[0][1].kind == nnkIntLit and SubgroupOp(n[0][1].intVal) == op

proc isDiscardAny(n: NimNode, ops: set[SubgroupOp]): bool =
  n.kind == nnkDiscardStmt and n[0].kind == nnkCall and n[0][0].strVal == "SubgroupOp" and
    n[0][1].kind == nnkIntLit and SubgroupOp(n[0][1].intVal) in ops

proc optimizeReconvergePoints*(node: NimNode): NimNode =
  if node.kind == nnkStmtList:
    result = newNimNode(nnkStmtList)
    var i = 0
    while i < node.len:
      if i < node.len - 4 and
          isDiscard(node[i], reconverge) and
          isDiscardAny(node[i+2], {barrier, subgroupBarrier, subgroupMemoryBarrier,
          groupMemoryBarrier, memoryBarrier}):
        inc i, 2
      elif i < node.len - 5 and
          ((isDiscard(node[i], subgroupMemoryBarrier) and
            isDiscardAny(node[i+3], {barrier, subgroupBarrier})) or
           (isDiscardAny(node[i], {groupMemoryBarrier, memoryBarrier}) and
            isDiscard(node[i+3], barrier))):
        if defined(debugSubgroup):
          warning("Optimizing away " & $SubgroupOp(node[i][0][1].intVal), node[i+1])
        result.add node[i+3..i+5] # keep barrier
        inc i, 6
      else:
        result.add optimizeReconvergePoints(node[i])
        inc i
  else:
    result = copyNimNode(node)
    for child in node:
      result.add optimizeReconvergePoints(child)

macro computeShader*(prc: untyped): untyped =
  expectKind(prc, {nnkProcDef, nnkFuncDef})

  # Track divergent control flow with unique IDs for reconvergence points
  var yieldId = -1
  template newYieldId(): untyped =
    inc yieldId
    newIntLitNode(yieldId)

  proc genReconvergeCall(): NimNode =
    let id = newYieldId()
    quote do:
      discard SubgroupOp(`reconverge`)
      yield SubgroupCommand(id: `id`, kind: reconverge)

  # Transform AST to handle divergent control flow and subgroup operations
  # Inserts reconvergence points after control flow blocks
  proc traverseAndModify(node: NimNode): NimNode =
    # Prevent nested routine definitions
    if node.kind in RoutineNodes:
      error("Routine definitions are not allowed inside the shader body", node)

    if node.kind in CallNodes:
      let op = getSubgroupOp(node)
      if op != invalid:
        return genSubgroupOpCall(op, node, newYieldId(), ident"iterArg")

    template flatAdd(res: NimNode, genCall: untyped) =
      res = newStmtList(res)
      copyChildrenTo(genCall, res)

    case node.kind
    of nnkForStmt, nnkWhileStmt:
      result = copyNimTree(node)
      # Detect continue statements to insert appropriate reconvergence points
      # This ensures proper lockstep execution in divergent paths
      proc checkForContinue(n: NimNode): bool =
        result = false
        for child in n:
          if child.kind == nnkContinueStmt:
            return true
          elif child.kind in {nnkForStmt, nnkWhileStmt}:
            discard
          else:
            if checkForContinue(child):
              return true

      let loopBody = result.body
      let hasContinue = checkForContinue(loopBody)
      result.body = traverseAndModify(loopBody)
      if hasContinue:
        let transformed = genReconvergeCall()
        copyChildrenTo(result.body, transformed)
        result.body = transformed
      flatAdd(result, genReconvergeCall())

    elif node.kind in {nnkTryStmt, nnkCaseStmt} or (node.kind == nnkIfStmt and
        (node.len == 0 or node[0].kind != nnkElifExpr)):
      result = copyNimTree(node)
      for i in ord(result.kind == nnkCaseStmt) ..< result.len:
        result[i] = traverseAndModify(result[i])
      flatAdd(result, genReconvergeCall())
    else:
      result = copyNimNode(node)
      for child in node:
        let transformed = traverseAndModify(child)
        # If both are statement lists, add children directly to avoid nesting
        if result.kind == nnkStmtList and transformed.kind == nnkStmtList:
          copyChildrenTo(transformed, result)
        else:
          result.add(transformed)

  # Create the iterator that implements the divergent control flow
  let procName = prc.name
  let iterArg = ident"iterArg"
  var traversedBody = traverseAndModify(prc.body)
  # Apply optimization to remove unnecessary reconverge points
  traversedBody = optimizeReconvergePoints(traversedBody)
  # Create symbols for both contexts
  let wgSym = genSym(nskParam, "wg")
  let threadSym = genSym(nskParam, "thread")
  let tidSym = genSym(nskParam, "threadId")
  # Generate template declarations for both contexts
  let wgTemplates = generateWorkGroupTemplates(wgSym)
  let threadTemplates = generateThreadTemplates(threadSym)

  result = quote do:
    proc `procName`(): ThreadClosure =
      iterator (`iterArg`: SubgroupResult, `wgSym`: WorkGroupContext,
                `threadSym`: ThreadContext, `tidSym`: uint32): SubgroupCommand =
        template gl_SubgroupInvocationID(): uint32 {.used.} = `tidSym`
        `threadTemplates`
        `wgTemplates`
        `traversedBody`

  # Now inject the parameters and pragmas from original proc
  result.params.add prc.params[1..^1]
  result.pragma = prc.pragma
