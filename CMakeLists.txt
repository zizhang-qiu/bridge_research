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
    SET(CUDA_HOME /usr/local/cuda-11.6)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.6/bin/nvcc)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.6")
    #    set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /usr/local/cuda-11.8)
    set(CUDA_NVCC_EXECUTABLE /usr/local/cuda-11.6/bin/nvcc)
    set(CMAKE_CUDA_EXTENSIONS ON)
    set(PYTHON_EXECUTABLE /home/work/miniconda3/envs/bridge_research/bin/python)
endif ()


find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

add_subdirectory(cpp/third_party/pybind11)
include_directories(cpp/third_party/pybind11/include)
#add_subdirectory(third_party/z3)
#include_directories(third_party/z3/src/api)
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

find_package(Threads REQUIRED)
add_library(bridge_double_dummy_solver OBJECT
        cpp/third_party/dds/include/dll.h
        cpp/third_party/dds/include/portab.h
        cpp/third_party/dds/src/ABsearch.cpp
        cpp/third_party/dds/src/ABsearch.h
        cpp/third_party/dds/src/ABstats.cpp
        cpp/third_party/dds/src/ABstats.h
        cpp/third_party/dds/src/CalcTables.cpp
        cpp/third_party/dds/src/CalcTables.h
        cpp/third_party/dds/src/dds.cpp
        cpp/third_party/dds/src/dds.h
        cpp/third_party/dds/src/DealerPar.cpp
        cpp/third_party/dds/src/debug.h
        cpp/third_party/dds/src/dump.cpp
        cpp/third_party/dds/src/dump.h
        cpp/third_party/dds/src/File.cpp
        cpp/third_party/dds/src/File.h
        cpp/third_party/dds/src/Init.cpp
        cpp/third_party/dds/src/Init.h
        cpp/third_party/dds/src/LaterTricks.cpp
        cpp/third_party/dds/src/LaterTricks.h
        cpp/third_party/dds/src/Memory.cpp
        cpp/third_party/dds/src/Memory.h
        cpp/third_party/dds/src/Moves.cpp
        cpp/third_party/dds/src/Moves.h
        cpp/third_party/dds/src/Par.cpp
        cpp/third_party/dds/src/parallel.h
        cpp/third_party/dds/src/PBN.cpp
        cpp/third_party/dds/src/PBN.h
        cpp/third_party/dds/src/PlayAnalyser.cpp
        cpp/third_party/dds/src/PlayAnalyser.h
        cpp/third_party/dds/src/QuickTricks.cpp
        cpp/third_party/dds/src/QuickTricks.h
        cpp/third_party/dds/src/Scheduler.cpp
        cpp/third_party/dds/src/Scheduler.h
        cpp/third_party/dds/src/SolveBoard.cpp
        cpp/third_party/dds/src/SolveBoard.h
        cpp/third_party/dds/src/SolverIF.cpp
        cpp/third_party/dds/src/SolverIF.h
        cpp/third_party/dds/src/System.cpp
        cpp/third_party/dds/src/System.h
        cpp/third_party/dds/src/ThreadMgr.cpp
        cpp/third_party/dds/src/ThreadMgr.h
        cpp/third_party/dds/src/Timer.cpp
        cpp/third_party/dds/src/Timer.h
        cpp/third_party/dds/src/TimerGroup.cpp
        cpp/third_party/dds/src/TimerGroup.h
        cpp/third_party/dds/src/TimerList.cpp
        cpp/third_party/dds/src/TimerList.h
        cpp/third_party/dds/src/TimeStat.cpp
        cpp/third_party/dds/src/TimeStat.h
        cpp/third_party/dds/src/TimeStatList.cpp
        cpp/third_party/dds/src/TimeStatList.h
        cpp/third_party/dds/src/TransTable.h
        cpp/third_party/dds/src/TransTableL.cpp
        cpp/third_party/dds/src/TransTableL.h
        cpp/third_party/dds/src/TransTableS.cpp
        cpp/third_party/dds/src/TransTableS.h
)
target_include_directories(bridge_double_dummy_solver PUBLIC cpp)
target_compile_features(bridge_double_dummy_solver PRIVATE cxx_std_17)

add_library(_rl cpp/rl/model_locker.h cpp/rl/logging.h
        cpp/rl/types.h cpp/rl/thread_loop.h cpp/rl/context.h cpp/rl/utils.h
        cpp/rl/tensor_dict.h cpp/rl/base.h cpp/rl/context.cc cpp/rl/span.h cpp/rl/str_utils.h
        cpp/rl/thread_loop.cc cpp/rl/model_locker.cc cpp/rl/str_utils.cc cpp/rl/logger.h cpp/rl/logger.cc cpp/rl/batcher.cc cpp/rl/batcher.h cpp/rl/batch_runner.cc cpp/rl/batch_runner.h)
target_include_directories(_rl PUBLIC cpp/rl)
target_include_directories(_rl PUBLIC cpp)
target_include_directories(_rl PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_rl PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_rl PUBLIC
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARIES})
set_target_properties(_rl PROPERTIES CXX_STANDARD 17)

add_library(bridge_lib cpp/bridge_lib/bridge_state.cc cpp/bridge_lib/bridge_scoring.cc cpp/bridge_lib/bluechip_utils.cc cpp/bridge_lib/bridge_scoring.h cpp/bridge_actor.h
        cpp/bridge_envs.h cpp/bridge_envs.cc cpp/bridge_lib/bridge_constants.h cpp/replay_buffer.h cpp/bridge_actor.cc cpp/bridge_lib/cards_and_ddts.h
        cpp/imp_env.h cpp/imp_env.cc cpp/multi_agent_transition_buffer.h cpp/multi_agent_transition_buffer.cc cpp/replay_buffer.cc
        cpp/bridge_lib/bridge_deal.cc cpp/bridge_lib/bridge_deal.h cpp/bridge_thread_loop.h cpp/bridge_thread_loop.cc cpp/transition.h cpp/transition.cc cpp/bridge_lib/bridge_state.h cpp/bridge_lib/dds_call.h cpp/bridge_lib/dds_call.cc cpp/sayc_bot.h cpp/sayc_bot.cc cpp/is_mcts.h cpp/bridge_lib/bridge_utils.h cpp/bridge_lib/bridge_utils.cc cpp/score_predictor.h cpp/search.h cpp/search.cc cpp/bridge_lib/encoder.h cpp/bridge_lib/encoder.cc cpp/bridge_lib/bridge_game.h cpp/bridge_lib/game_parameters.h cpp/bridge_lib/game_parameters.cc cpp/bridge_lib/bridge_card.h cpp/bridge_lib/bridge_move.h cpp/bridge_lib/bridge_card.cc cpp/bridge_lib/bridge_move.cc cpp/bridge_lib/bridge_call.cc cpp/bridge_lib/bridge_call.cc cpp/bridge_lib/bridge_call.h
        cpp/test_belief_thread_loop.cc
        cpp/test_belief_thread_loop.h
)
target_include_directories(bridge_lib PUBLIC cpp)
target_include_directories(bridge_lib PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(bridge_lib PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(bridge_lib PUBLIC "cpp/third_party/z3/src/api")

target_link_libraries(bridge_lib PUBLIC
        _rl
        ${TORCH_LIBRARIES}
        bridge_double_dummy_solver)
target_compile_options(bridge_lib PUBLIC ${Z3_COMPONENT_CXX_FLAGS})

pybind11_add_module(rl_cpp SHARED cpp/pybind.cc)
target_include_directories(rl_cpp PUBLIC cpp)
target_include_directories(rl_cpp PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rl_cpp PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rl_cpp PUBLIC
        bridge_lib)
set_target_properties(rl_cpp PROPERTIES OUTPUT_NAME "rl_cpp")

if (WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "/MT")
    set(CMAKE_CXX_FLAGS_DEBUG "/MTd")

    add_executable(test cpp/test.cc)
    set_target_properties(test PROPERTIES LINKER_LANGUAGE CXX)

    target_include_directories(test PUBLIC cpp)
    target_include_directories(test PUBLIC ${TORCH_INCLUDE_DIRS})
    target_include_directories(test PUBLIC ${PYTHON_INCLUDE_DIRS})
    target_include_directories(bridge_lib PUBLIC "cpp/third_party/z3/src/api")
    target_link_libraries(test PUBLIC
            bridge_lib
            ${TORCH_LIBRARIES}
            ${PYTHON_LIBRATIES}
            bridge_double_dummy_solver)

    add_subdirectory(cpp/third_party/googletest)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    include_directories(
            ${gtest_SOURCE_DIR}/include
            ${gtest_SOURCE_DIR}
    )
    #    add_executable(bluechip_test cpp/tests/bluechip_utils_test.cc)
    #    target_include_directories(bluechip_test PUBLIC .)
    #    target_include_directories(bluechip_test PUBLIC ${TORCH_INCLUDE_DIRS})
    #    target_include_directories(bluechip_test PUBLIC ${PYTHON_INCLUDE_DIRS})
    #    target_link_libraries(bluechip_test PUBLIC
    #            bridge_lib
    #            gtest gtest_main)
    #    include(GoogleTest)
    #    gtest_discover_tests(bluechip_test)
    #
    #    add_executable(bridge_scoring_test cpp/tests/bridge_scoring_test.cc)
    #    target_include_directories(bridge_scoring_test PUBLIC .)
    #    target_include_directories(bridge_scoring_test PUBLIC ${TORCH_INCLUDE_DIRS})
    #    target_include_directories(bridge_scoring_test PUBLIC ${PYTHON_INCLUDE_DIRS})
    #    target_link_libraries(bridge_scoring_test PUBLIC
    #            bridge_lib
    #            gtest gtest_main)
    #    include(GoogleTest)
    #    gtest_discover_tests(bridge_scoring_test)
    #
    #    add_executable(bridge_state_test cpp/tests/bridge_state_test.cc)
    #    target_include_directories(bridge_state_test PUBLIC .)
    #    target_include_directories(bridge_state_test PUBLIC ${TORCH_INCLUDE_DIRS})
    #    target_include_directories(bridge_state_test PUBLIC ${PYTHON_INCLUDE_DIRS})
    #    target_link_libraries(bridge_state_test PUBLIC
    #            bridge_lib
    #            gtest gtest_main)
    #    include(GoogleTest)
    #    gtest_discover_tests(bridge_state_test)
    #
    #    add_executable(concurrent_queue_test cpp/tests/concurrent_queue_test.cc)
    #    target_include_directories(concurrent_queue_test PUBLIC .)
    #    target_include_directories(concurrent_queue_test PUBLIC ${TORCH_INCLUDE_DIRS})
    #    target_include_directories(concurrent_queue_test PUBLIC ${PYTHON_INCLUDE_DIRS})
    #    target_link_libraries(concurrent_queue_test PUBLIC
    #            bridge_lib
    #            gtest gtest_main)
    #    include(GoogleTest)
    #    gtest_discover_tests(concurrent_queue_test)
    #
    #    add_executable(str_utils_test cpp/tests/str_utils_test.cc)
    #    target_include_directories(str_utils_test PUBLIC .)
    #    target_include_directories(str_utils_test PUBLIC ${TORCH_INCLUDE_DIRS})
    #    target_include_directories(str_utils_test PUBLIC ${PYTHON_INCLUDE_DIRS})
    #    target_link_libraries(str_utils_test PUBLIC
    #            bridge_lib
    #            gtest gtest_main)
    #    include(GoogleTest)
    #    gtest_discover_tests(str_utils_test)
    #
    #    add_executable(dds_test cpp/tests/dds_test.cc)
    #    target_include_directories(dds_test PUBLIC .)
    #    target_include_directories(dds_test PUBLIC ${TORCH_INCLUDE_DIRS})
    #    target_include_directories(dds_test PUBLIC ${PYTHON_INCLUDE_DIRS})
    #    target_link_libraries(dds_test PUBLIC
    #            bridge_lib
    #            gtest gtest_main)
    #    include(GoogleTest)
    #    gtest_discover_tests(dds_test)
endif ()




