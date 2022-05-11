# Install

## Core Components

- tar -xvzf l_openvino_toolkit_p_2021.4.752.tgz
- cd l_openvino_toolkit_p_2021.4.752
- sudo ./install.sh

## Install Dependencies

- cd opt/intel/openvino_2021/install_dependencies
- sudo -E ./install_openvino_dependencies.sh

## Env Variable

- echo source /opt/intel/openvino_2021/bin/setupvars.sh >> ~/.bashrc
- source ~/.bashrc

## MO Env

- cd /opt/intel/openvino_2021/deployment_tools/model_optimizer
- python -m pip install -r requirements.txt

# ONNX Model Convert

- cd /opt/intel/openvino_2021/deployment_tools/model_optimizer
- python mo_onnx.py --input_model /home/fdiao/dl_cv/OpenVINO/model/model_DAD_3_7.onnx --output_dir /home/fdiao/dl_cv/OpenVINO/model/ --input_shape [1,224,224,3]

# Run Our MobileNet

## Python

- cd \<YourFolder\>/openvino/python

- python classification.py -m /home/fdiao/dl_cv/OpenVINO/openvino/python/model/model_DAD_3_7.xml -i /home/fdiao/dl_cv/OpenVINO/openvino/python/model/phone_interact.jpg -d CPU

## CPP

- cd \<YourFolder\>/openvino/cpp/build
- cmake ..
- make
- ./classification_test ../model/model_DAD_3_7.xml ../model/phone_interact.jpg CPU

# Other

## Model Download

- cd /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader
- python -m pip install -r requirements.in
- python downloader.py --name alexnet --output_dir /data/home/jianfeng/DF_WS/openvino/python

## Convert

- cd /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader
- python converter.py --name alexnet --download_dir /data/home/jianfeng/DF_WS/openvino/python --output_dir /data/home/jianfeng/DF_WS/openvino/python

## Run Demo

- /opt/intel/openvino_2021/deployment_tools/inference_engine/samples/python/hello_classification

- python hello_classification.py -m /data/home/jianfeng/DF_WS/openvino/python/public/alexnet/FP32/alexnet.xml -i car.bmp -d CPU
