#include <samples/classification_results.h>

#include <inference_engine.hpp>
#include <iterator>
#include <memory>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include <string>
#include <vector>

using namespace InferenceEngine;

#define tcout std::cout
#define file_name_t std::string
#define imread_t cv::imread
#define ClassificationResult_t ClassificationResult

class OpenvinoInference
{
  public:
    file_name_t input_model_;
    file_name_t input_image_path_;
    std::string device_name_;
    std::string input_name_;
    std::string output_name_;
    size_t batch{0};
    size_t height{0};
    size_t width{0};
    size_t num_channels{0};
    OpenvinoInference(file_name_t input_model, file_name_t input_image_path, std::string device_name);
    ~OpenvinoInference();

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
    bool Inference();
};
