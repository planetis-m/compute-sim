import std/macros, core

proc wrapComputeImpl(compute, buffer, shared, params: NimNode): NimNode =
  let envSym = genSym(nskParam, "env")
  let bufferSym = genSym(nskParam, "buffer")
  let sharedSym = genSym(nskParam, "shared")
  let paramsSym = genSym(nskParam, "params")

  template makeTypeExpr(node: NimNode, default: typedesc): NimNode =
    if node != nil: node.getTypeInst
    else: quote do: `default`

  let bufferType = makeTypeExpr(buffer, int32)
  let sharedType = makeTypeExpr(shared, int32)
  let paramsType = makeTypeExpr(params, int32)

  let call = newTree(nnkCall, compute, envSym)
  proc addArgsFromType(node, param: NimNode, isAddr = false) =
    # need to now the compute parameter types!
    if node != nil:
      var typ = node.getTypeInst
      if typ.typeKind == ntyPtr:
        typ = typ[0]
        if typ.typeKind == ntyTuple:
          for i in 0..<typ.len:
            call.add quote do: addr `param`[`i`]
        else:
          call.add param
      else:
        if typ.typeKind == ntyTuple:
          for i in 0..<typ.len:
            call.add quote do: addr `param`[`i`]
        else:
          call.add quote do: addr `param`

  addArgsFromType(buffer, bufferSym)
  addArgsFromType(shared, sharedSym, isAddr = true)
  addArgsFromType(params, paramsSym)

  result = quote do:
    proc wCompute(`envSym`: GlEnvironment,
                  `bufferSym`: `bufferType`,
                  `sharedSym`: ptr `sharedType`,
                  `paramsSym`: `paramsType`): ThreadClosure {.nimcall.} =
      `call`

macro wrapCompute*(compute, buffer: typed): untyped =
  wrapComputeImpl(compute, buffer, nil, nil)

macro wrapCompute*(compute, buffer, shared: typed): untyped =
  wrapComputeImpl(compute, buffer, shared, nil)

macro wrapCompute*(compute, buffer, params: typed): untyped =
  wrapComputeImpl(compute, buffer, nil, params)

macro wrapCompute*(compute, buffer, shared, params: typed): untyped =
  wrapComputeImpl(compute, buffer, shared, params)
