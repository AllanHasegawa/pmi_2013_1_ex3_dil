cmake_minimum_required(VERSION 2.8)

project(pmi_ex3)

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored"
)

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored"
)


#packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/headers)

#set(CMAKE_CXX_FLAGS_RELEASE "-std=c++0x")

if(CUDA_FOUND)
	# nvcc flags
	set(CUDA_NVCC_FLAGS "-Xcompiler -Wall; 
			    -gencode;arch=compute_20,code=sm_20;")
			    
			    #-gencode;arch=compute_11,code=sm_11; 
			    #-gencode;arch=compute_12,code=sm_12;
			    #-gencode;arch=compute_13,code=sm_13;

	if (UNIX)
		set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
	endif (UNIX)

	set(SRCS main.cpp)
	cuda_compile(CUDA_DIL dil.cu)

	cuda_add_executable(ex3_dil ${SRCS} ${CUDA_DIL})
	target_link_libraries(ex3_dil ${OpenCV_LIBS})
	
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()
