cmake_minimum_required(VERSION 3.23)
project(bridge_research)

set(CMAKE_CXX_STANDARD 17)
set(ABSL_PROPAGATE_CXX_STD ON)
set(CONDA_PREFIX "D:/ProgramData/Anaconda/envs/bridge_research")
set(PYTHON_EXECUTABLE "${CONDA_PREFIX}/python.exe")
set(PYTHON_INCLUDE_DIR "${CONDA_PREFIX}/include")
set(PYTHON_LIBRARIES "${CONDA_PREFIX}/libs/python3.lib")
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/pybind11)
include_directories(third_party/abseil-cpp/absl)
include_directories(third_party/pybind11/include)
set(CUDA_HOME "D:/cuda-11.7")
execute_process(
          COMMAND ${PYTHON_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
          OUTPUT_VARIABLE TorchPath
        )

message(STATUS TorchPath=${TorchPath})


list(APPEND CMAKE_PREFIX_PATH ${TorchPath})

find_package(Torch REQUIRED)
set(TORCH_PYTHON_LIBRARIES  "${TorchPath}/lib/torch_python.lib")

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

add_library(_br bridge_actor.h model_locker.h logging.h tests/test_bridge_scoring.h
        bridge_state.h  types.h bridge_env.h thread_loop.h context.h utils.h replay_buffer.h tests/cards_and_ddts.h base.h bluechip_utils.h)
target_include_directories(_br PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(_br PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_br PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(_br PUBLIC third_party/zlib)
target_link_libraries(_br PUBLIC
        bridge_double_dummy_solver
        absl::base
        absl::algorithm
        absl::flags
        absl::optional
        absl::random_random
        absl::strings
        absl::str_format
        absl::time
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARIES})

pybind11_add_module(rl_cpp SHARED pybind.cc)
target_include_directories(rl_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(rl_cpp PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(rl_cpp PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rl_cpp PUBLIC
        absl::base
        absl::algorithm
        absl::flags
        absl::optional
        absl::random_random
        absl::strings
        absl::str_format
        absl::time
        bridge_double_dummy_solver
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARIES})
set_target_properties(rl_cpp PROPERTIES OUTPUT_NAME "rl_cpp")



add_executable(test test.cc tests/test_bluechip_utils.h tests/test_imp_env.h)
set_target_properties(test PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(test PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(test PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(test PUBLIC third_party/zlib)
target_link_libraries(test PUBLIC
        absl::base
        absl::algorithm
        absl::flags
        absl::optional
        absl::random_random
        absl::strings
        absl::str_format
        absl::time
        ${TORCH_LIBRARIES}
        ${PYTHON_LIBRATIES}
        bridge_double_dummy_solver)



