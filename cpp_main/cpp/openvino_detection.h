#include <inference_engine.hpp>
#include <iterator>
#include <memory>
#include <samples/classification_results.h>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
using namespace InferenceEngine;

#define tcout std::cout
#define file_name_t std::string
#define imread_t cv::imread
#define ClassificationResult_t ClassificationResult

struct OpenvinoInferenceResult
{
    int class_idx;
    float probability;
};

class OpenvinoInference
{
  public:
    file_name_t input_model_;
    file_name_t input_image_path_{""};
    std::string device_name_;
    OpenvinoInferenceResult openvino_result_;
    OpenvinoInference(file_name_t input_model, std::string device_name);
    ~OpenvinoInference();
    bool Initialization();
    bool Inference(int rawdata_height, int rawdata_width, auto *rawdata);
    bool Inference(file_name_t input_image_path);
    bool Inference(cv::Mat image);

  private:
    Core ie_;
    CNNNetwork network_;
    ExecutableNetwork executable_network_;
    InferRequest infer_request_;
    bool ReadModel();
    bool ConfigureInputOutput();
    void LoadingModel();
    void CreateInferRequest();
    bool PreProcessing(cv::Mat input);
    bool PrepareInput(cv::Mat image);
    bool PrepareInput(int rawdata_height, int rawdata_width, auto *rawdata);
    bool PrepareInput(file_name_t input_image_path);
    void DoSyncInference();
    void ProcessOutput();
    std::string input_name_;
    std::string output_name_;
    size_t modeldata_batch_{0};
    size_t modeldata_height_{0};
    size_t modeldata_width_{0};
    size_t modeldata_num_channels_{0};
    int class_num_{0};
};
