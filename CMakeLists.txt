cmake_minimum_required(VERSION 3.9)


set (ProjectName "equi_match_test")
project(${ProjectName})

set(SOURCE_FILES
        main.cpp
        )


set(CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )

find_package (Eigen3)
include_directories("/usr/include/eigen3")

include_directories(".")


find_package (Boost 1.58.0 REQUIRED COMPONENTS system filesystem)
if (Boost_FOUND)
    include_directories (SYSTEM ${Boost_INCLUDE_DIR})
endif ()

SET(GDAL_INCLUDE_DIR "/usr/include/gdal")
include_directories( ${GDAL_INCLUDE_DIR} )

find_package(OpenMP)
if (OPENMP_FOUND)
    message("-- Building with OpenMP.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message("flags set: ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    if (UNIX)
        # At least on Linux, we need pthreads to be enabled for mutex to
        # compile.  This may not work on Windows or Android.
        find_package(Threads REQUIRED)
    endif (UNIX)
else (OPENMP_FOUND)
    message("-- Failed to find OpenMP")
    update_cache_variable(OPENMP OFF)
endif (OPENMP_FOUND)

#todo: refactor the odometry and vtrajectory into seperate parts; to reduce compile time and complexity.

include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()

# example usage
enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-pedantic")


add_executable(${ProjectName} ${SOURCE_FILES} main.cpp)
target_link_libraries (${ProjectName} gdal ${Boost_LIBRARIES} ${OpenCV_LIBS})
