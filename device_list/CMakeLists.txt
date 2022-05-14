cmake_minimum_required(VERSION 3.5)
project(device_list)
set(CMAKE_CXX_STANDARD 14)
include_directories(
    ../cpp_main/cpp/common/utils/include/
)


find_package(OpenCV REQUIRED)
#set(InferenceEngine_DIR "/opt/intel/openvino_2021.4.752/deployment_tools/inference_engine/share")
find_package(InferenceEngine)
#set(ngraph_DIR "/opt/intel/openvino_2021.4.752/deployment_tools/ngraph/cmake")
find_package(ngraph REQUIRED)
#set(ngraph_LIBRARIES "/opt/intel/openvino_2021.4.752/deployment_tools/ngraph/lib/")
#set(ngraph_INCLUDE_DIRS "/opt/intel/openvino_2021.4.752/deployment_tools/ngraph/include/")

add_executable(device_list main.cpp)

target_include_directories(device_list
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
        PUBLIC ${ngraph_INCLUDE_DIRS}
)
target_link_libraries(device_list
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)



