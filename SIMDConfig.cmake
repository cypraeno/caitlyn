include(CheckCXXCompilerFlag)

# Check and add support for various SIMD instructions
#check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
#if(COMPILER_SUPPORTS_AVX512)
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
#endif()

check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
endif()

check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
if(COMPILER_SUPPORTS_AVX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()

check_cxx_compiler_flag("-msse4.2" COMPILER_SUPPORTS_SSE42)
if(COMPILER_SUPPORTS_SSE42)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()