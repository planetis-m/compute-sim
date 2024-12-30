type
  TVec*[N: static[int], T] = object
    data: array[N, T]

  TVec2*[T] = TVec[2, T]
  TVec3*[T] = TVec[3, T]
  TVec4*[T] = TVec[4, T]

  # Float32 vectors
  Vec2* = TVec2[float32]
  Vec3* = TVec3[float32]
  Vec4* = TVec4[float32]

  # Int32 vectors
  IVec2* = TVec2[int32]
  IVec3* = TVec3[int32]
  IVec4* = TVec4[int32]

  # UInt32 vectors
  UVec2* = TVec2[uint32]
  UVec3* = TVec3[uint32]
  UVec4* = TVec4[uint32]

  # Float64 vectors
  DVec2* = TVec2[float64]
  DVec3* = TVec3[float64]
  DVec4* = TVec4[float64]

  # Boolean vectors
  BVec2* = TVec2[bool]
  BVec3* = TVec3[bool]
  BVec4* = TVec4[bool]

# Accessors
template x*[N, T](v: TVec[N, T]): T = v.data[0]
template y*[N, T](v: TVec[N, T]): T = v.data[1]
template z*[N, T](v: TVec[N, T]): T = v.data[2]
template w*[N, T](v: TVec[N, T]): T = v.data[3]

template `x=`*[N, T](v: TVec[N, T]; val: T) = v.data[0] = val
template `y=`*[N, T](v: TVec[N, T]; val: T) = v.data[1] = val
template `z=`*[N, T](v: TVec[N, T]; val: T) = v.data[2] = val
template `w=`*[N, T](v: TVec[N, T]; val: T) = v.data[3] = val

# Array access
proc `[]`*[N, T](v: TVec[N, T], i: int): T {.inline.} = v.data[i]
proc `[]=`*[N, T](v: var TVec[N, T], i: int, val: T) {.inline.} = v.data[i] = val

# Template for generating 2D vector constructors
template defineVec2Constructors*(Vec2Type, Vec3Type, Vec4Type: typedesc,
                                 baseType: typedesc, constructorName: untyped) =
  proc constructorName*(x, y: baseType): Vec2Type =
    ## Constructs Vec2 from two components
    Vec2Type(data: [x, y])

  proc constructorName*(v: baseType): Vec2Type =
    ## Constructs Vec2 with the same value for both components
    Vec2Type(data: [v, v])

  proc constructorName*(v: Vec3Type): Vec2Type =
    ## Constructs Vec2 from first two components of Vec3
    Vec2Type(data: [v.x, v.y])

  proc constructorName*(v: Vec4Type): Vec2Type =
    ## Constructs Vec2 from first two components of Vec4
    Vec2Type(data: [v.x, v.y])

# Template for generating 3D vector constructors
template defineVec3Constructors*(Vec2Type, Vec3Type, Vec4Type: typedesc,
                                 baseType: typedesc, constructorName: untyped) =
  proc constructorName*(x, y, z: baseType): Vec3Type =
    ## Constructs Vec3 from three components
    Vec3Type(data: [x, y, z])

  proc constructorName*(v: baseType): Vec3Type =
    ## Constructs Vec3 with the same value for all components
    Vec3Type(data: [v, v, v])

  proc constructorName*(v: Vec2Type, z: baseType): Vec3Type =
    ## Constructs Vec3 from Vec2 and a z component
    Vec3Type(data: [v.x, v.y, z])

  proc constructorName*(v: Vec4Type): Vec3Type =
    ## Constructs Vec3 from first three components of Vec4
    Vec3Type(data: [v.x, v.y, v.z])

# Template for generating 4D vector constructors
template defineVec4Constructors*(Vec2Type, Vec3Type, Vec4Type: typedesc,
                                 baseType: typedesc, constructorName: untyped) =
  proc constructorName*(x, y, z, w: baseType): Vec4Type =
    ## Constructs Vec4 from four components
    Vec4Type(data: [x, y, z, w])

  proc constructorName*(v: baseType): Vec4Type =
    ## Constructs Vec4 with the same value for all components
    Vec4Type(data: [v, v, v, v])

  proc constructorName*(v: Vec2Type, z, w: baseType): Vec4Type =
    ## Constructs Vec4 from Vec2 and z, w components
    Vec4Type(data: [v.x, v.y, z, w])

  proc constructorName*(v: Vec3Type, w: baseType): Vec4Type =
    ## Constructs Vec4 from Vec3 and w component
    Vec4Type(data: [v.x, v.y, v.z, w])

  proc constructorName*(xy: Vec2Type, zw: Vec2Type): Vec4Type =
    ## Constructs Vec4 from two Vec2 components
    Vec4Type(data: [xy.x, xy.y, zw.x, zw.y])

{.push boundChecks: off.}

# Usage example for boolean vectors:
defineVec2Constructors(BVec2, BVec3, BVec4, bool, bvec2)
defineVec3Constructors(BVec2, BVec3, BVec4, bool, bvec3)
defineVec4Constructors(BVec2, BVec3, BVec4, bool, bvec4)

# Usage example for integer vectors:
defineVec2Constructors(IVec2, IVec3, IVec4, int32, ivec2)
defineVec3Constructors(IVec2, IVec3, IVec4, int32, ivec3)
defineVec4Constructors(IVec2, IVec3, IVec4, int32, ivec4)

# Usage example for unsigned integer vectors:
defineVec2Constructors(UVec2, UVec3, UVec4, uint32, uvec2)
defineVec3Constructors(UVec2, UVec3, UVec4, uint32, uvec3)
defineVec4Constructors(UVec2, UVec3, UVec4, uint32, uvec4)

# Usage example for float vectors:
defineVec2Constructors(Vec2, Vec3, Vec4, float32, vec2)
defineVec3Constructors(Vec2, Vec3, Vec4, float32, vec3)
defineVec4Constructors(Vec2, Vec3, Vec4, float32, vec4)

# Usage example for double vectors:
defineVec2Constructors(DVec2, DVec3, DVec4, float64, dvec2)
defineVec3Constructors(DVec2, DVec3, DVec4, float64, dvec3)
defineVec4Constructors(DVec2, DVec3, DVec4, float64, dvec4)

{.pop.}
