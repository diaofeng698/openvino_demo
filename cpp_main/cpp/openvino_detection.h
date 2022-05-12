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
    std::string input_name_;
    std::string output_name_;
    size_t modeldata_batch_{0};
    size_t modeldata_height_{0};
    size_t modeldata_width_{0};
    size_t modeldata_num_channels_{0};
    size_t rawdata_height_{0};
    size_t rawdata_width_{0};
    void *data_;
    int class_num_{0};
    OpenvinoInferenceResult openvino_result_;
    OpenvinoInference(file_name_t input_model, file_name_t input_image_path, std::string device_name);
    OpenvinoInference(file_name_t input_model, int rawdata_height, int rawdata_width, std::string device_name,
                      void *data);
    ~OpenvinoInference();
    bool Inference();

  private:
    ExecutableNetwork executable_network_;
    Core ie_;
    CNNNetwork network_;
    InferRequest infer_request_;
    bool ReadModel();
    bool ConfigureInputOoutput();
    void LoadingModel();
    void CreateInferRequest();
    bool PrepareInput();
    void DoSyncInference();
    void ProcessOutput();
};
