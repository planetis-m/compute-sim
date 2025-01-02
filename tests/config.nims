--path:"../src"
--cc:"clang"
--gc:"arc"
--threads:on
--define:"release"
--define:"useMalloc"
when not defined(windows):
  --debugger:"native"
  --define:"noSignalHandler"
  --passc:"-fsanitize=thread -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
  --passl:"-fsanitize=thread -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
switch("define", "ThreadPoolSize=10")
