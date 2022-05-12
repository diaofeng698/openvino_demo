cmake_minimum_required(VERSION 3.5)
project(openvino_detection)
set(CMAKE_CXX_STANDARD 14)
include_directories(
    ./common/utils/include/
    # /opt/intel/openvino_2021.4.752/deployment_tools/inference_engine/include/
    # /opt/intel/openvino_2021.4.752/deployment_tools/ngraph/include/
)


find_package(OpenCV REQUIRED)
#set(InferenceEngine_DIR "/opt/intel/openvino_2021.4.752/deployment_tools/inference_engine/share")
find_package(InferenceEngine)
#set(ngraph_DIR "/opt/intel/openvino_2021.4.752/deployment_tools/ngraph/cmake")
find_package(ngraph REQUIRED)
#set(ngraph_LIBRARIES "/opt/intel/openvino_2021.4.752/deployment_tools/ngraph/lib/")
#set(ngraph_INCLUDE_DIRS "/opt/intel/openvino_2021.4.752/deployment_tools/ngraph/include/")

# add_executable(openvino_detection main.cpp openvino_detection.cpp)
add_library(openvino_detection SHARED openvino_detection.cpp)

target_include_directories(openvino_detection
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
        PUBLIC ${ngraph_INCLUDE_DIRS}
)
target_link_libraries(openvino_detection
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)
