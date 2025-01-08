--path:"../src"
--cc:"clang"
--gc:"arc"
--threads:on
--define:"release"
--define:"useMalloc"
--define:"ThreadPoolSize=10"
#--define:debugSubgroup

when not defined(windows):
  --debugger:"native"
  --define:"noSignalHandler"
  when defined(asan):
    --passc:"-fsanitize=address -fno-omit-frame-pointer"
    --passl:"-fsanitize=address -fno-omit-frame-pointer"
  else: # default to tsan
    --passc:"-fsanitize=thread -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
    --passl:"-fsanitize=thread -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"
