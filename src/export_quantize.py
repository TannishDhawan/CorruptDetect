import os
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from sklearn.metrics import f1_score
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    quant_pre_process,
)
from model import CorruptDetectNet
from dataset import MultiTaskCIFAR10


class CIFARCalibrationReader(CalibrationDataReader):
    def __init__(self, num_samples: int = 200):
        ds = MultiTaskCIFAR10(root="./data", split="train", corrupt=True)
        self.samples = [
            {"input": ds[i][0].unsqueeze(0).numpy()} for i in range(num_samples)
        ]
        self.iterator = iter(self.samples)

    def get_next(self):
        return next(self.iterator, None)


def get_sensitive_depthwise_nodes(onnx_path: str):
    """
    Finds 3x3 depthwise Conv nodes (group > 1) and the initial stem Conv.
    Depthwise convs account for <2% of MobileNetV3 bytes but cause severe INT8
    quantization noise if quantized aggressively.
    """
    model = onnx.load(onnx_path)
    exclude = []
    conv_count = 0
    for node in model.graph.node:
        if node.op_type == "Conv":
            is_depthwise = any(attr.name == "group" and attr.i > 1 for attr in node.attribute)
            if conv_count == 0 or is_depthwise:
                exclude.append(node.name)
            conv_count += 1
    return exclude


def evaluate_onnx(onnx_path: str):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)
    for _ in range(50):
        sess.run(None, {"input": dummy})
    t0 = time.perf_counter()
    for _ in range(500):
        sess.run(None, {"input": dummy})
    latency_ms = ((time.perf_counter() - t0) / 500) * 1000.0
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    clean_ds = MultiTaskCIFAR10(root="./data", split="test", corrupt=False)
    clean_correct = 0
    n_eval = 2000
    for i in range(n_eval):
        img, obj_y, _ = clean_ds[i]
        obj_logits, _ = sess.run(None, {"input": img.unsqueeze(0).numpy()})
        if int(np.argmax(obj_logits)) == obj_y:
            clean_correct += 1
    clean_acc = 100.0 * clean_correct / n_eval

    corr_ds = MultiTaskCIFAR10(root="./data", split="test", corrupt=True)
    corr_obj_correct, corr_obj_total = 0, 0
    corr_preds, corr_targets = [], []
    for i in range(n_eval):
        img, obj_y, corr_y = corr_ds[i]
        obj_logits, corr_logits = sess.run(None, {"input": img.unsqueeze(0).numpy()})
        if corr_y != 0:
            corr_obj_total += 1
            if int(np.argmax(obj_logits)) == obj_y:
                corr_obj_correct += 1
        corr_preds.append(int(np.argmax(corr_logits)))
        corr_targets.append(corr_y)

    corrupt_acc = 100.0 * corr_obj_correct / corr_obj_total
    macro_f1 = 100.0 * f1_score(corr_targets, corr_preds, average="macro", zero_division=0)
    return size_mb, latency_ms, clean_acc, corrupt_acc, macro_f1


def main():
    fp32_path = "./models/corrupt_detect_fp32.onnx"
    prep_path = "./models/corrupt_detect_prep.onnx"
    int8_path = "./models/corrupt_detect_int8.onnx"

    # 1. Export PyTorch -> FP32 ONNX
    model = CorruptDetectNet(pretrained=False)
    model.load_state_dict(torch.load("./models/robust_best.pth", map_location="cpu", weights_only=True))
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["object_logits", "corruption_logits"],
        dynamo=False,
    )

    # 2. ONNX Runtime Graph Pre-Processing (Shape inference + Conv/BN fusion)
    quant_pre_process(input_model_path=fp32_path, output_model_path=prep_path)

    # 3. Exclude sensitive 3x3 depthwise convs (<2% of weights) & run Per-Channel INT8 PTQ
    exclude_nodes = get_sensitive_depthwise_nodes(prep_path)
    print(f"Pre-processed ONNX graph. Excluding {len(exclude_nodes)} sensitive depthwise/stem nodes from INT8.")

    calibration_reader = CIFARCalibrationReader(num_samples=200)
    quantize_static(
        model_input=prep_path,
        model_output=int8_path,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        nodes_to_exclude=exclude_nodes,
        per_channel=True,
        reduce_range=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    print(f"Saved Per-Channel INT8 ONNX -> {int8_path}\n")


    # Strip unused ai.onnx.ml opset tag for C++ ORT SDK compatibility
    m = onnx.load(int8_path)
    for i in reversed(range(len(m.opset_import))):
        if m.opset_import[i].domain == 'ai.onnx.ml':
            del m.opset_import[i]
    onnx.save(m, int8_path)

    # 4. Benchmark FP32 vs INT8
    fp32_size, fp32_lat, fp32_clean, fp32_corr, fp32_f1 = evaluate_onnx(fp32_path)
    int8_size, int8_lat, int8_clean, int8_corr, int8_f1 = evaluate_onnx(int8_path)

    print("| Precision | Size (MB) | Compression | Python ORT Latency (ms) | Clean Acc | Corrupted Acc | Detect Macro-F1 |")
    print("|-----------|-----------|-------------|-------------------------|-----------|---------------|-----------------|")
    print(f"| FP32      | {fp32_size:9.2f} |       1.00x | {fp32_lat:23.3f} | {fp32_clean:8.2f}% | {fp32_corr:12.2f}% | {fp32_f1:14.2f}% |")
    print(f"| INT8      | {int8_size:9.2f} | {fp32_size/int8_size:10.2f}x | {int8_lat:23.3f} | {int8_clean:8.2f}% | {int8_corr:12.2f}% | {int8_f1:14.2f}% |")


if __name__ == "__main__":
    main()
