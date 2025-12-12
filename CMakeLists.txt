cmake_minimum_required(VERSION 3.23)
project(FlashMLA LANGUAGES CXX CUDA)

# Options
set(FLASH_MLA_ARCH "sm120" CACHE STRING "Target arch: sm100 or sm120")
option(FLASH_MLA_FORCE_FALLBACK "Force fallback implementations (forward, and optionally bwd variants)" OFF)
option(FLASH_MLA_SM120_DISABLE_VARLEN_BWD "Disable SM120 varlen backward while iterating" OFF)
option(FLASH_MLA_SM120_DISABLE_MLA_BWD "Disable SM120 MLA backward while iterating" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Dependencies
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Base sources
set(BASE_SOURCES
  csrc/pybind.cpp
  csrc/smxx/get_mla_metadata.cu
  csrc/smxx/mla_combine.cu
)

set(VARIANT_SOURCES)
set(VARIANT_DEFINES)
set(GENCODE_FLAGS)
set(OUT_NAME)

if(FLASH_MLA_ARCH STREQUAL "sm120")
  list(APPEND VARIANT_SOURCES
    csrc/sm120/prefill/dense/fmha_cutlass_fwd_sm120.cu
    csrc/sm120/prefill/dense/fmha_cutlass_bwd_sm120.cu)
  list(APPEND VARIANT_DEFINES FLASH_MLA_BUILD_SM120 FLASH_MLA_DISABLE_SM100)
  if(FLASH_MLA_FORCE_FALLBACK)
    list(APPEND VARIANT_DEFINES FLASH_MLA_FORCE_FALLBACK)
  endif()
  if(FLASH_MLA_SM120_DISABLE_VARLEN_BWD)
    list(APPEND VARIANT_DEFINES FLASH_MLA_SM120_DISABLE_VARLEN_BWD)
  endif()
  if(FLASH_MLA_SM120_DISABLE_MLA_BWD)
    list(APPEND VARIANT_DEFINES FLASH_MLA_SM120_DISABLE_MLA_BWD)
  endif()
  set(GENCODE_FLAGS -gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120)
  set(OUT_NAME cuda_sm120)
elseif(FLASH_MLA_ARCH STREQUAL "sm100")
  list(APPEND VARIANT_SOURCES
    csrc/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu
    csrc/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu)
  list(APPEND VARIANT_DEFINES FLASH_MLA_BUILD_SM100 FLASH_MLA_DISABLE_SM90)
  if(FLASH_MLA_FORCE_FALLBACK)
    list(APPEND VARIANT_DEFINES FLASH_MLA_FORCE_FALLBACK)
  endif()
  set(GENCODE_FLAGS -gencode arch=compute_100a,code=sm_100a -gencode arch=compute_100a,code=compute_100a)
  set(OUT_NAME cuda_sm100)
else()
  message(FATAL_ERROR "Unknown FLASH_MLA_ARCH: ${FLASH_MLA_ARCH}. Expected sm100 or sm120")
endif()

add_library(${OUT_NAME} MODULE ${BASE_SOURCES} ${VARIANT_SOURCES})

# Include dirs
target_include_directories(${OUT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/sm90
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/include
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/cutlass/tools/util/include
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(${OUT_NAME} PRIVATE Torch::Torch CUDA::cudart)

target_compile_definitions(${OUT_NAME} PRIVATE ${VARIANT_DEFINES})

if(MSVC)
  target_compile_options(${OUT_NAME} PRIVATE /Zc:__cplusplus /permissive- /EHsc /DNOMINMAX /DWIN32_LEAN_AND_MEAN /D_HAS_EXCEPTIONS=1 /utf-8 /W0)
endif()

# NVCC flags aligned with setup.py
target_compile_options(${OUT_NAME} PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    -O3 -std=c++17 -DNDEBUG -D_USE_MATH_DEFINES -Wno-deprecated-declarations
    --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
    --ptxas-options=-v,--register-usage-level=10
    -include msvc_compat.h
    -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__
    ${GENCODE_FLAGS}
  >)

# Ensure Python module naming/suffix and output dir
set_target_properties(${OUT_NAME} PROPERTIES
  PREFIX ""
  OUTPUT_NAME ${OUT_NAME}
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/flash_mla"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/flash_mla"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build/cmake/${OUT_NAME}"
)

if(WIN32)
  set_target_properties(${OUT_NAME} PROPERTIES SUFFIX ".pyd")
endif()
