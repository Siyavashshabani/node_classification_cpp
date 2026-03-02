cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(graph_node_classifaction_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point this to your LibTorch folder, or export CMAKE_PREFIX_PATH before cmake:
find_package(Torch REQUIRED)

add_executable(graph_node_classifaction_cpp src/main.cpp)
# add_executable(graph_node_classifaction_cpp src/main_opt.cpp)


target_link_libraries(graph_node_classifaction_cpp "${TORCH_LIBRARIES}")
set_property(TARGET graph_node_classifaction_cpp PROPERTY CXX_STANDARD 17)


# Recommended for GCC/Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(graph_node_classifaction_cpp PRIVATE -O3)
endif()


