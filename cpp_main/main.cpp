#include "openvino_detection.h"

int main()
{
    file_name_t input_model = "../model/model_DAD_3_7.xml";
    file_name_t input_image_path = "../model/phone_interact.jpg";
    std::string device_name = "CPU";

    OpenvinoInference *infer;
    infer = new OpenvinoInference(input_model, device_name);
    infer->Inference(input_image_path);
    return 0;
}