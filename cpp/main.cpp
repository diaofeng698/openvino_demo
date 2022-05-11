#include "openvino_detection.h"

int main()
{
    file_name_t input_model = "../model/model_DAD_3_7.xml";
    file_name_t input_image_path = "../model/phone_interact.jpg";
    std::string device_name = "CPU";
    OpenvinoInference infer(input_model, input_image_path, device_name);
    return 0;
}