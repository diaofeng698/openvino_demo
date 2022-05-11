#include "openvino_detection.h"

OpenvinoInference::OpenvinoInference(file_name_t input_model, file_name_t input_image_path, std::string device_name)
{
    input_model_ = input_model;
    input_image_path_ = input_image_path;
    device_name_ = device_name;
    std::cout << "Initial Successfully! " << std::endl;
    Inference();
}

OpenvinoInference::~OpenvinoInference()
{
    std::cout << "Destructor Finished! " << std::endl;
}

bool OpenvinoInference::Inference()
{
    try
    {
        if (ReadModel())
            return EXIT_FAILURE;
        if (ConfigureInputOoutput())
            return EXIT_FAILURE;
        LoadingModel();
        CreateInferRequest();
        if (PrepareInput())
            return EXIT_FAILURE;
        DoSyncInference();
        ProcessOutput();
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Openvino Inference Finished" << std::endl;
    return EXIT_SUCCESS;
}

bool OpenvinoInference::ReadModel()
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
    return EXIT_SUCCESS;
}

bool OpenvinoInference::ConfigureInputOoutput()
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

bool OpenvinoInference::PrepareInput()
{

    cv::Mat image = imread_t(input_image_path_);

    if (!image.data)
    {
        std::cout << "Frame Load Failed" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "raw inputH " << image.rows << " inputW " << image.cols << " inputChannel " << image.channels()
              << std::endl;

    cv::Mat resized_img;
    cv::resize(image, resized_img, cv::Size(224, 224));
    std::cout << "resized inputH " << resized_img.rows << " inputW " << resized_img.cols << " inputChannel "
              << resized_img.channels() << std::endl;

    cv::Mat gray_img;
    cv::cvtColor(resized_img, gray_img, cv::COLOR_BGR2GRAY);
    std::cout << "gray inputH " << gray_img.rows << " inputW " << gray_img.cols << " inputChannel "
              << gray_img.channels() << std::endl;

    Blob::Ptr inputBlob = infer_request_.GetBlob(input_name_);
    SizeVector dims = inputBlob->getTensorDesc().getDims();

    batch = dims[0];
    height = dims[1];
    width = dims[2];
    num_channels = dims[3];

    Blob::Ptr outputBlob = infer_request_.GetBlob(output_name_);
    SizeVector output_dims = outputBlob->getTensorDesc().getDims();

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

    for (int b = 0, volImg = num_channels * height * width; b < batch; b++)
    {
        for (int idx = 0, volChl = height * width; idx < volChl; idx++)
        {

            for (int c = 0; c < num_channels; ++c)
            {
                data[b * volImg + idx * num_channels + c] = gray_img.data[idx];
            }
        }
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
    ClassificationResult_t classificationResult(output, {input_image_path_});
    classificationResult.print();
}