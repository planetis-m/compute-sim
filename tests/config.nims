--path:"../src"
--gc:"arc"
--threads:on
--define:"useMalloc"
--cc:"clang"
when not defined(windows):
  --debugger:"native"
  --passc:"-fsanitize=thread -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
  --passl:"-fsanitize=thread -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
