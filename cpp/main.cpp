#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

const std::vector<std::string> OBJECT_CLASSES = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

const std::vector<std::string> CORRUPTION_CLASSES = {
    "Clean", "Noise", "Blur", "Weather", "Digital"
};

std::vector<float> preprocess_image(const cv::Mat& bgr_img) {
    cv::Mat rgb, resized, float_img;
    cv::cvtColor(bgr_img, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(32, 32));
    resized.convertTo(float_img, CV_32FC3, 1.0f / 255.0f);

    const float mean[3] = {0.4914f, 0.4822f, 0.4465f};
    const float std_dev[3] = {0.2023f, 0.1994f, 0.2010f};

    std::vector<float> tensor_data(1 * 3 * 32 * 32);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 32; ++h) {
            for (int w = 0; w < 32; ++w) {
                float pixel = float_img.at<cv::Vec3f>(h, w)[c];
                tensor_data[c * 32 * 32 + h * 32 + w] = (pixel - mean[c]) / std_dev[c];
            }
        }
    }
    return tensor_data;
}

int argmax(const float* data, size_t size) {
    return static_cast<int>(std::distance(data, std::max_element(data, data + size)));
}

int main(int argc, char* argv[]) {
    std::string model_path = (argc > 1) ? argv[1] : "./models/corrupt_detect_int8.onnx";

    cv::Mat img(32, 32, CV_8UC3, cv::Scalar(120, 160, 200));
    std::vector<float> input_values = preprocess_image(img);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CorruptDetectCPP");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, 3, 32, 32};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_values.data(), input_values.size(),
        input_shape.data(), input_shape.size()
    );

    const char* input_names[] = {"input"};
    const char* output_names[] = {"object_logits", "corruption_logits"};

    for (int i = 0; i < 50; ++i) {
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);
    }

    const int num_runs = 1000;
    std::vector<double> latencies_ms;
    latencies_ms.reserve(num_runs);
    std::vector<Ort::Value> outputs;

    for (int i = 0; i < num_runs; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::sort(latencies_ms.begin(), latencies_ms.end());
    double p50 = latencies_ms[num_runs / 2];
    double p90 = latencies_ms[static_cast<int>(num_runs * 0.90)];

    const float* obj_logits = outputs[0].GetTensorData<float>();
    const float* corr_logits = outputs[1].GetTensorData<float>();

    std::cout << "==========================================\n";
    std::cout << "CorruptDetect C++ ONNX Runtime Benchmark\n";
    std::cout << "==========================================\n";
    std::cout << "Model Path          : " << model_path << "\n";
    std::cout << "Predicted Object    : " << OBJECT_CLASSES[argmax(obj_logits, 10)] << "\n";
    std::cout << "Detected Condition  : " << CORRUPTION_CLASSES[argmax(corr_logits, 5)] << "\n";
    std::cout << "Latency p50 (1k run): " << p50 << " ms\n";
    std::cout << "Latency p90 (1k run): " << p90 << " ms\n";
    std::cout << "==========================================\n";
    return 0;
}
