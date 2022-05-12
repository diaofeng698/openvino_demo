cmake_minimum_required(VERSION 3.5)
project(openvino_detection_main)
set(CMAKE_CXX_STANDARD 14)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/cpp ${CMAKE_CURRENT_SOURCE_DIR}/cpp/common/utils/include)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cpp)


add_executable(openvino_detection_main main.cpp)
target_link_libraries(openvino_detection_main openvino_detection)




