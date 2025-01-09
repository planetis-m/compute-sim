# (c) 2024 Antonis Geralis
import std/[macros, strutils], core, vectors

proc raiseInvalidSubgroupOp(kind: SubgroupOp) {.noinline, noreturn.} =
  raise newException(AssertionDefect, "Invalid subgroup operation: " & $kind)

type
  PReadOnly[T] = distinct ptr T

proc `[]`[T](p: PReadOnly[T]): lent T = cast[ptr T](p)[]

proc transformParameters(params: NimNode): tuple[extraParams: seq[NimNode], templates: NimNode] =
  var extraParams: seq[NimNode] = @[]
  let templates = newStmtList()
  # Process each parameter after the return type
  for i in 1..<params.len:
    let param = params[i]
    expectKind param, nnkIdentDefs
    let paramType = param[^2]
    for j in 0..<param.len-2:
      let name = param[j]
      # Generate a new symbol for the parameter
      let genSym = genSym(nskParam, name.strVal)
      # Create the pointer type for the parameter
      let ptrType = newTree(nnkPtrTy,
          if paramType.kind == nnkVarTy: paramType[0] else: paramType)
      extraParams.add newIdentDefs(genSym, ptrType)
      # Create the template for this parameter
      let templateNode = if paramType.kind == nnkVarTy:
        quote do:
          template `name`: untyped = `genSym`[]
      else:
        quote do:
          template `name`: untyped = PReadOnly(`genSym`)[]
      templates.add templateNode
  (extraParams, templates)

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

const
  NotUseful = 0xFEAD0000
  Optimizable = 0xFEAD0001
  SubOptimizable = 0xFEAD0002
  GroupOptimizable = 0xFEAD0003
  Barrier = 0xFEAD0004
  SubBarrier = 0xFEAD0005

proc genSubgroupOpCall(op: SubgroupOp; node, id, iterArg: NimNode): NimNode =
  # Generate the command part based on operation type
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
  # Choose appropriate sentinel based on operation type
  let sentinel =
    case op
    of barrier: Barrier
    of subgroupBarrier: SubBarrier
    of subgroupMemoryBarrier: SubOptimizable
    of memoryBarrier, groupMemoryBarrier: GroupOptimizable
    else: NotUseful
  # Combine both parts
  result = quote do:
    discard `sentinel`
    yield `cmdPart`
    case `iterArg`.kind
    of `op`:
      `resultPart`
    else:
      raiseInvalidSubgroupOp(`op`)
  result[1].copyLineInfo(node)
  result[2].copyLineInfo(node)

proc generateEnvTemplates(envSym: NimNode): NimNode =
  result = newNimNode(nnkStmtList)
  # Define templates for each GlEnvironment field
  let envFields = [
    "gl_GlobalInvocationID",
    "gl_LocalInvocationID",
    "gl_WorkGroupID",
    "gl_WorkGroupSize",
    "gl_NumWorkGroups",
    "gl_NumSubgroups",
    "gl_SubgroupSize",
    "gl_SubgroupID",
    "gl_SubgroupInvocationID",
  ]
  for field in envFields:
    let fieldIdent = ident(field)
    result.add quote do:
      template `fieldIdent`(): untyped {.used.} = `envSym`.`fieldIdent`

template isDiscard(n: NimNode, val: int): bool =
  n.kind == nnkDiscardStmt and n[0].kind == nnkIntLit and n[0].intVal == val

proc optimizeReconvergePoints*(node: NimNode): NimNode =
  if node.kind == nnkStmtList:
    result = newNimNode(nnkStmtList)
    var i = 0
    while i < node.len:
      if i < node.len - 4 and
          ((isDiscard(node[i], Optimizable) and
          isDiscard(node[i+2], Barrier) or isDiscard(node[i+2], SubBarrier))):
        result.add node[i+2..i+4] # keep barrier
        inc i, 5
      elif i < node.len - 5 and
          ((isDiscard(node[i], SubOptimizable) and
          isDiscard(node[i+3], Barrier) or isDiscard(node[i+3], SubBarrier)) or
          (isDiscard(node[i], GroupOptimizable) and isDiscard(node[i+3], Barrier))):
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
      discard `Optimizable`
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
  # Create template declarations for GlEnvironment fields
  let envSym = genSym(nskParam, "env")
  let envTemplates = generateEnvTemplates(envSym)
  let (newParams, paramTemplates) = transformParameters(prc.params)

  result = quote do:
    proc `procName`(`envSym`: GlEnvironment): ThreadClosure =
      `envTemplates`
      `paramTemplates`
      iterator (`iterArg`: SubgroupResult): SubgroupCommand =
        `traversedBody`

  # Now inject the parameters and pragmas from original proc
  result.params.add newParams
