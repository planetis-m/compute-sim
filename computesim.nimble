# Package
version       = "1.2.7"
author        = "Antonis Geralis"
description   = "A compute shader emulator for learning and debugging GPU compute shaders."
license       = "MIT"

# Dependencies
requires "nim >= 2.2.0"
requires "threading#head"
requires "malebolgia#head"

import os

const
  ProjectUrl = "https://github.com/planetis-m/compute-sim"
  PkgDir = thisDir().quoteShell
  DocsDir = PkgDir / "docs"

task docs, "Generate documentation":
  # https://nim-lang.github.io/Nim/docgen.html
  withDir(PkgDir):
    let modules = [
      "computesim",
      "computesim/api",
      "computesim/vectors",
      "computesim/transform"
    ]
    for tmp in modules:
      let doc = DocsDir / (tmp.lastPathPart & ".html")
      let src = "src" / (tmp & ".nim")
      # Generate the docs for {src}
      exec("nim doc --verbosity:0 --git.url:" & ProjectUrl &
          " --git.devel:master --git.commit:master --out:" & doc & " " & src)
