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
include_directories(${TORCH_INCLUDE_DIRS})

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

add_library(_rl rl/model_locker.h rl/logging.h
        rl/types.h rl/thread_loop.h rl/context.h rl/utils.h
        rl/tensor_dict.h rl/base.h rl/context.cc rl/span.h rl/str_utils.h
        rl/thread_loop.cc rl/model_locker.cc)
target_include_directories(_rl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/rl)
target_include_directories(_rl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(_rl PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_rl PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_rl PUBLIC
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARIES})
set_target_properties(_rl PROPERTIES CXX_STANDARD 17)

add_library(bridge_lib bridge_state.cc bridge_scoring.cc bluechip_utils.cc bridge_scoring.h bridge_actor.h
        bridge_envs.h bridge_envs.cc bridge_constants.h replay_buffer.h bridge_actor.cc search.h cards_and_ddts.h
        imp_env.h imp_env.cc multi_agent_transition_buffer.h multi_agent_transition_buffer.cc replay_buffer.cc
        search.cc bridge_deal.cc bridge_deal.h bridge_thread_loop.h bridge_thread_loop.cc)
target_include_directories(bridge_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(bridge_lib PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(bridge_lib PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(bridge_lib PUBLIC
        _rl
        ${TORCH_LIBRARIES}
        bridge_double_dummy_solver)

pybind11_add_module(rl_cpp SHARED pybind.cc)
target_include_directories(rl_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(rl_cpp PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rl_cpp PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rl_cpp PUBLIC
        bridge_lib)
set_target_properties(rl_cpp PROPERTIES OUTPUT_NAME "rl_cpp")

if (WIN32)

    add_executable(test test.cc)
    set_target_properties(test PROPERTIES LINKER_LANGUAGE CXX)

    target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_include_directories(test PUBLIC ${TORCH_INCLUDE_DIRS})
    target_include_directories(test PUBLIC ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(test PUBLIC
            bridge_lib
            ${TORCH_LIBRARIES}
            ${PYTHON_LIBRATIES}
            bridge_double_dummy_solver)

    add_subdirectory(third_party/googletest)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    include_directories(
            ${gtest_SOURCE_DIR}/include
            ${gtest_SOURCE_DIR}
    )
    add_executable(bluechip_test tests/bluechip_utils_test.cc)
    target_include_directories(bluechip_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_include_directories(bluechip_test PUBLIC ${TORCH_INCLUDE_DIRS})
    target_include_directories(bluechip_test PUBLIC ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(bluechip_test PUBLIC
            bridge_lib
            gtest gtest_main)
    include(GoogleTest)
    gtest_discover_tests(bluechip_test)

    add_executable(bridge_scoring_test tests/bridge_scoring_test.cc)
    target_include_directories(bridge_scoring_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_include_directories(bridge_scoring_test PUBLIC ${TORCH_INCLUDE_DIRS})
    target_include_directories(bridge_scoring_test PUBLIC ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(bridge_scoring_test PUBLIC
            bridge_lib
            gtest gtest_main)
    include(GoogleTest)
    gtest_discover_tests(bridge_scoring_test)

    add_executable(bridge_state_test tests/bridge_state_test.cc)
    target_include_directories(bridge_state_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_include_directories(bridge_state_test PUBLIC ${TORCH_INCLUDE_DIRS})
    target_include_directories(bridge_state_test PUBLIC ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(bridge_state_test PUBLIC
            bridge_lib
            gtest gtest_main)
    include(GoogleTest)
    gtest_discover_tests(bridge_state_test)
endif ()




