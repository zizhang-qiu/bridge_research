cmake_minimum_required(VERSION 3.10)
project(bridge_research)

set(CMAKE_CXX_STANDARD 17)
set(CONDA_PREFIX "D:/ProgramData/Anaconda/envs/bridge_research")
#SET(CONDA_PREFIX $ENV{CONDA_PREFIX})
#MESSAGE(CONDA_PREFIX)
if (WIN32)
    MESSAGE("Compiling on Windows")
    set(PYTHON_EXECUTABLE "${CONDA_PREFIX}/python.exe")
    set(PYTHON_INCLUDE_DIR "${CONDA_PREFIX}/include")
    set(PYTHON_LIBRARIES "${CONDA_PREFIX}/libs/python3.lib")
else ()
    MESSAGE("Compiling on Linux")
    set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPYBIND11_COMPILER_TYPE=\\\"_gcc\\\" -DPYBIND11_STDLIB=\\\"_libstdcpp\\\" -DPYBIND11_BUILD_ABI=\\\"_cxxabi1011\\\"")
    set(CMAKE_CUDA_FLAGS "-std=c++17")
    SET(CUDA_HOME /usr/local/cuda-11.8)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
    #    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /usr/local/cuda-11.8)
    set(CUDA_NVCC_EXECUTABLE /usr/local/cuda-11.8/bin/nvcc)
    set(CMAKE_CUDA_EXTENSIONS ON)
    set(PYTHON_EXECUTABLE /root/miniconda3/bin/python)
endif ()


find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

add_subdirectory(third_party/pybind11)
include_directories(third_party/pybind11/include)
execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
        OUTPUT_VARIABLE TorchPath
)

message(STATUS TorchPath=${TorchPath})


list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

find_package(Torch REQUIRED)
if (WIN32)
    set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/torch_python.lib")
else ()
    set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python.so")
endif ()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_library(bridge_double_dummy_solver OBJECT
        third_party/dds/include/dll.h
        third_party/dds/include/portab.h
        third_party/dds/src/ABsearch.cpp
        third_party/dds/src/ABsearch.h
        third_party/dds/src/ABstats.cpp
        third_party/dds/src/ABstats.h
        third_party/dds/src/CalcTables.cpp
        third_party/dds/src/CalcTables.h
        third_party/dds/src/dds.cpp
        third_party/dds/src/dds.h
        third_party/dds/src/DealerPar.cpp
        third_party/dds/src/debug.h
        third_party/dds/src/dump.cpp
        third_party/dds/src/dump.h
        third_party/dds/src/File.cpp
        third_party/dds/src/File.h
        third_party/dds/src/Init.cpp
        third_party/dds/src/Init.h
        third_party/dds/src/LaterTricks.cpp
        third_party/dds/src/LaterTricks.h
        third_party/dds/src/Memory.cpp
        third_party/dds/src/Memory.h
        third_party/dds/src/Moves.cpp
        third_party/dds/src/Moves.h
        third_party/dds/src/Par.cpp
        third_party/dds/src/parallel.h
        third_party/dds/src/PBN.cpp
        third_party/dds/src/PBN.h
        third_party/dds/src/PlayAnalyser.cpp
        third_party/dds/src/PlayAnalyser.h
        third_party/dds/src/QuickTricks.cpp
        third_party/dds/src/QuickTricks.h
        third_party/dds/src/Scheduler.cpp
        third_party/dds/src/Scheduler.h
        third_party/dds/src/SolveBoard.cpp
        third_party/dds/src/SolveBoard.h
        third_party/dds/src/SolverIF.cpp
        third_party/dds/src/SolverIF.h
        third_party/dds/src/System.cpp
        third_party/dds/src/System.h
        third_party/dds/src/ThreadMgr.cpp
        third_party/dds/src/ThreadMgr.h
        third_party/dds/src/Timer.cpp
        third_party/dds/src/Timer.h
        third_party/dds/src/TimerGroup.cpp
        third_party/dds/src/TimerGroup.h
        third_party/dds/src/TimerList.cpp
        third_party/dds/src/TimerList.h
        third_party/dds/src/TimeStat.cpp
        third_party/dds/src/TimeStat.h
        third_party/dds/src/TimeStatList.cpp
        third_party/dds/src/TimeStatList.h
        third_party/dds/src/TransTable.h
        third_party/dds/src/TransTableL.cpp
        third_party/dds/src/TransTableL.h
        third_party/dds/src/TransTableS.cpp
        third_party/dds/src/TransTableS.h
        )
target_include_directories(bridge_double_dummy_solver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(bridge_double_dummy_solver PUBLIC DDS_NO_STATIC_INIT)

add_library(_br encode_bridge.h bridge_actor.h model_locker.h logging.h bridge_scoring.h tests/test_bridge_scoring.h
        bridge_state.h types.h bridge_env.h thread_loop.h context.h utils.h replay_buffer.h tests/cards_and_ddts.h base.h bluechip_utils.h generate_deals.h str_utils.h span.h)
target_include_directories(_br PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(_br PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_br PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_br PUBLIC
        bridge_double_dummy_solver
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARIES})

pybind11_add_module(rl_cpp SHARED pybind.cc)
target_include_directories(rl_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(rl_cpp PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rl_cpp PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rl_cpp PUBLIC
        bridge_double_dummy_solver
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARIES})
set_target_properties(rl_cpp PROPERTIES OUTPUT_NAME "rl_cpp")


add_executable(test test.cc tests/test_bluechip_utils.h)
set_target_properties(test PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(test PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(test PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(test PUBLIC
        ${TORCH_LIBRARIES}
        ${PYTHON_LIBRATIES}
        bridge_double_dummy_solver)



