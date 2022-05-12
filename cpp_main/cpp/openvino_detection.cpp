#include "openvino_detection.h"

OpenvinoInference::OpenvinoInference(file_name_t input_model, std::string device_name)
{
    std::cout << "Openvino Inference Start Initialization " << std::endl;
    input_model_ = input_model;
    device_name_ = device_name;
    if (Initialization())
        std::cout << "Openvino Initialization Failed, Please check ! " << std::endl;
}

OpenvinoInference::~OpenvinoInference()
{
    std::cout << "OpenvinoInference Destructor Finished! " << std::endl;
}

bool OpenvinoInference::Initialization()
{
    if (ReadModel())
        return EXIT_FAILURE;
    if (ConfigureInputOutput())
        return EXIT_FAILURE;
    LoadingModel();
    CreateInferRequest();
    return EXIT_SUCCESS;
}

bool OpenvinoInference::Inference(file_name_t input_image_path)
{

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    if (PrepareInput(input_image_path))
        return EXIT_FAILURE;
    DoSyncInference();
    ProcessOutput();
    std::cout << "Openvino Inference Finished" << std::endl;
    gettimeofday(&tv2, NULL);
    double diff_time = ((double)(tv2.tv_usec - tv1.tv_usec) / 1000.0) + ((double)(tv2.tv_sec - tv1.tv_sec) * 1000.0);
    std::cout << "Openvino Whole Infer Time [ms] : " << diff_time << std::endl;
    return EXIT_SUCCESS;
}

bool OpenvinoInference::Inference(int rawdata_height, int rawdata_width, auto *rawdata)
{

    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    if (PrepareInput(rawdata_height, rawdata_width, rawdata))
        return EXIT_FAILURE;
    DoSyncInference();
    ProcessOutput();
    std::cout << "Openvino Inference Finished" << std::endl;
    gettimeofday(&tv2, NULL);
    double diff_time = ((double)(tv2.tv_usec - tv1.tv_usec) / 1000.0) + ((double)(tv2.tv_sec - tv1.tv_sec) * 1000.0);
    std::cout << "Openvino Whole Infer Time [ms] : " << diff_time << std::endl;
    return EXIT_SUCCESS;
}

bool OpenvinoInference::Inference(cv::Mat image)
{
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    if (PrepareInput(image))
        return EXIT_FAILURE;
    DoSyncInference();
    ProcessOutput();
    std::cout << "Openvino Inference Finished" << std::endl;
    gettimeofday(&tv2, NULL);
    double diff_time = ((double)(tv2.tv_usec - tv1.tv_usec) / 1000.0) + ((double)(tv2.tv_sec - tv1.tv_sec) * 1000.0);
    std::cout << "Openvino Whole Infer Time [ms] : " << diff_time << std::endl;
    return EXIT_SUCCESS;
}

bool OpenvinoInference::ReadModel()
{
    if (access(input_model_.c_str(), F_OK) != -1)
    {
        network_ = ie_.ReadNetwork(input_model_);
        if (network_.getOutputsInfo().size() != 1)
        {
            throw std::logic_error("Sample supports topologies with 1 output only");
            return EXIT_FAILURE;
        }
        if (network_.getInputsInfo().size() != 1)
        {
            throw std::logic_error("Sample supports topologies with 1 input only");
            return EXIT_FAILURE;
        }
    }
    else
    {
        std::cout << "Model File Missed, Please check ! " << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

bool OpenvinoInference::ConfigureInputOutput()
{

    // --------------------------- Prepare input blobs
    if (network_.getInputsInfo().empty())
    {
        std::cerr << "Network inputs info is empty" << std::endl;
        return EXIT_FAILURE;
    }
    InputInfo::Ptr input_info = network_.getInputsInfo().begin()->second;
    input_name_ = network_.getInputsInfo().begin()->first;

    // input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    // input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::FP32);

    // --------------------------- Prepare output blobs
    if (network_.getOutputsInfo().empty())
    {
        std::cerr << "Network outputs info is empty" << std::endl;
        return EXIT_FAILURE;
    }
    DataPtr output_info = network_.getOutputsInfo().begin()->second;
    output_name_ = network_.getOutputsInfo().begin()->first;

    output_info->setPrecision(Precision::FP32);

    return EXIT_SUCCESS;
}

void OpenvinoInference::LoadingModel()
{
    executable_network_ = ie_.LoadNetwork(network_, device_name_);
}

void OpenvinoInference::CreateInferRequest()
{
    infer_request_ = executable_network_.CreateInferRequest();
}

bool OpenvinoInference::PreProcessing(cv::Mat input)
{

    Blob::Ptr inputBlob = infer_request_.GetBlob(input_name_);
    SizeVector dims = inputBlob->getTensorDesc().getDims();

    modeldata_batch_ = dims[0];
    modeldata_height_ = dims[1];
    modeldata_width_ = dims[2];
    modeldata_num_channels_ = dims[3];

    cv::Mat resized_img;
    cv::resize(input, resized_img, cv::Size(modeldata_width_, modeldata_height_));
    std::cout << "resized inputH " << resized_img.rows << " inputW " << resized_img.cols << " inputChannel "
              << resized_img.channels() << std::endl;

    cv::Mat gray_img;
    cv::cvtColor(resized_img, gray_img, cv::COLOR_BGR2GRAY);
    std::cout << "gray inputH " << gray_img.rows << " inputW " << gray_img.cols << " inputChannel "
              << gray_img.channels() << std::endl;

    Blob::Ptr outputBlob = infer_request_.GetBlob(output_name_);
    SizeVector output_dims = outputBlob->getTensorDesc().getDims();

    class_num_ = output_dims[1];

    MemoryBlob::Ptr minput = as<MemoryBlob>(inputBlob);
    if (!minput)
    {
        std::cout << "We expect MemoryBlob from inferRequest, but by fact we "
                     "were not able to cast inputBlob to MemoryBlob"
                  << std::endl;
        return EXIT_FAILURE;
    }
    // locked memory holder should be alive all time while access to its
    // buffer happens
    auto minputHolder = minput->wmap();

    auto data = minputHolder.as<PrecisionTrait<Precision::FP32>::value_type *>();
    if (data == nullptr)
    {
        throw std::runtime_error("Input blob has not allocated buffer");
        return EXIT_FAILURE;
    }

    for (int b = 0, volImg = modeldata_num_channels_ * modeldata_height_ * modeldata_width_; b < modeldata_batch_; b++)
    {
        for (int idx = 0, volChl = modeldata_height_ * modeldata_width_; idx < volChl; idx++)
        {

            for (int c = 0; c < modeldata_num_channels_; ++c)
            {
                data[b * volImg + idx * modeldata_num_channels_ + c] = gray_img.data[idx];
            }
        }
    }
}

bool OpenvinoInference::PrepareInput(int rawdata_height, int rawdata_width, auto *rawdata)
{
    if (rawdata == nullptr)
    {
        std::cout << "Input Frame Load Failed, Please check !" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "raw inputH " << rawdata_height << " inputW " << rawdata_width << " inputChannel " << 3 << std::endl;
    cv::Mat input_frame(rawdata_height, rawdata_width, CV_8UC3, rawdata);
    if (PreProcessing(input_frame))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

bool OpenvinoInference::PrepareInput(cv::Mat image)
{
    if (image.data == nullptr)
    {
        std::cout << "Input Image Load Failed, Please check !" << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "raw inputH " << image.rows << " inputW " << image.cols << " inputChannel " << image.channels()
              << std::endl;
    if (PreProcessing(image))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}

bool OpenvinoInference::PrepareInput(file_name_t input_image_path)
{

    if (access(input_image_path.c_str(), F_OK) != -1)
    {
        cv::Mat input_file = imread_t(input_image_path);
        if (!input_file.data)
        {
            std::cout << "Image File Load Failed" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "raw inputH " << input_file.rows << " inputW " << input_file.cols << " inputChannel "
                  << input_file.channels() << std::endl;
        if (PreProcessing(input_file))
            return EXIT_FAILURE;
    }
    else
    {
        std::cout << "Image File Missed, Please check ! " << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void OpenvinoInference::DoSyncInference()
{
    /* Running the request synchronously */
    infer_request_.Infer();
}

void OpenvinoInference::ProcessOutput()
{
    Blob::Ptr output = infer_request_.GetBlob(output_name_);
    // Print classification results
    ClassificationResult_t classificationResult(output, {input_image_path_}, modeldata_batch_, class_num_);
    classificationResult.print();
    openvino_result_.class_idx = classificationResult._max_idx;
    openvino_result_.probability = classificationResult._max_prob;

    std::cout << "Inference Result ID : " << openvino_result_.class_idx << " Conf : " << openvino_result_.probability
              << std::endl;
}