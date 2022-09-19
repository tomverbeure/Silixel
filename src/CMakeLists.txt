CMAKE_MINIMUM_REQUIRED(VERSION 3.17 FATAL_ERROR)

ADD_SUBDIRECTORY(LibSL)

#============================================================
# silixel
#============================================================

# ADD_DEFINITIONS(-DUSE_GLUX -DOPENGL -DOPENGL4)

SET(SHADERS
	sh_simul
	sh_posedge
	sh_outports
	sh_init
	sh_visu
)

# This is a cmake macro that's defined in LibSL
AUTO_BIND_SHADERS( ${SHADERS} )

ADD_EXECUTABLE(silixel
	silixel.cc
	blif.cc
	blif.h
	sh_simul.cs
	sh_simul.h
	sh_posedge.cs
	sh_posedge.h
	sh_outports.cs
	sh_outports.h
	sh_init.cs
	sh_init.h
	sh_visu.fp
	sh_visu.vp
	sh_visu.h
	simul_cpu.cc
	simul_cpu.h
	simul_gpu.cc
	simul_gpu.h
	read.cc
	read.h
	analyze.cc
	analyze.h
	)

IF(LINUX)
TARGET_LINK_LIBRARIES(silixel LibSL LibSL_gl4core freeglut)
ELSE()
TARGET_LINK_LIBRARIES(silixel LibSL LibSL_gl4core)
ENDIF()

# SRC_PATH #define is used to find the .blif file.
ADD_DEFINITIONS(-DSRC_PATH=\"${CMAKE_SOURCE_DIR}\")

#============================================================
# silixel_cpu
#============================================================

ADD_EXECUTABLE(silixel_cpu
	silixel_cpu.cc
	simul_cpu.cc
	simul_cpu.h
	blif.cc
	blif.h
	read.cc
	read.h
	analyze.cc
	analyze.h
	)

TARGET_LINK_LIBRARIES(silixel_cpu LibSL)

#============================================================
# silixel_cuda
#============================================================

PROJECT(silixel_cuda LANGUAGES CUDA CXX)

ADD_EXECUTABLE(silixel_cuda
	silixel_cuda.cc
    dummy.cu        # There must be at least 1 .cu file otherwise CUDA link libraries aren't included!
	blif.cc
	blif.h
	sh_simul.cs
	sh_simul.h
	sh_posedge.cs
	sh_posedge.h
	sh_outports.cs
	sh_outports.h
	sh_init.cs
	sh_init.h
	sh_visu.fp
	sh_visu.vp
	sh_visu.h
	simul_cpu.cc
	simul_cpu.h
	simul_gpu.cc
	simul_gpu.h
	simul_cuda.cc
	simul_cuda.h
	read.cc
	read.h
	analyze.cc
	analyze.h
	)

TARGET_COMPILE_FEATURES(silixel_cuda PUBLIC cxx_std_11)
TARGET_INCLUDE_DIRECTORIES(silixel_cuda PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
TARGET_INCLUDE_DIRECTORIES(silixel_cuda PUBLIC Common)
SET_TARGET_PROPERTIES(silixel_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION on)

IF(LINUX)
TARGET_LINK_LIBRARIES(silixel_cuda LibSL LibSL_gl4core freeglut)
ELSE()
TARGET_LINK_LIBRARIES(silixel_cuda LibSL LibSL_gl4core)
ENDIF()

# IF(WIN32)
# INSTALL(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/release/silixel.exe" DESTINATION ${CMAKE_SOURCE_DIR}/bin)
# ELSE(WIN32)
# INSTALL(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/silixel" DESTINATION ${CMAKE_SOURCE_DIR}/bin)
# ENDIF(WIN32)
