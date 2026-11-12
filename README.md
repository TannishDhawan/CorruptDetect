# CorruptDetect: Multi-Task Edge Vision & C++ Inference Engine

End-to-end computer vision and edge inference pipeline built with **PyTorch, ONNX Runtime (Python & C++17), OpenCV, and Qualcomm AI Hub**.

The system trains a modified **MobileNetV3-Small** architecture with dual classification heads to simultaneously:
1. **Classify Objects Under Image Corruptions** (10-class CIFAR-10).
2. **Detect Image Corruption Families** (`Clean`, `Noise`, `Blur`, `Weather`, `Digital`).

It includes a complete quantization analysis (resolving MobileNetV3 depthwise convolution collapse under INT8 Post-Training Quantization), a native **C++17 inference benchmark CLI**, and on-device profiling on a physical **Samsung Galaxy S24 (Snapdragon 8 Gen 3)**.

---

## 1. Multi-Task Robustness & Corruption Detection Results

Evaluated on the strictly held-out 10,000-image CIFAR-10 test split (with 5,000 training images held out for validation during checkpoint selection to guarantee zero test leakage):

| Model Variant | Clean Test Acc (10k) | Corrupted Test Acc (8k) | Noise Acc | Blur Acc | Weather Acc | Digital Acc | Corruption Detect Acc | Corruption Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Clean-Trained)** | **88.36%** | 52.45% | 40.75% | 41.60% | 86.50% | 40.95% | — | — |
| **CorruptDetect (Multi-Task)** | 86.72% | **82.81% (+30.36%)** | **81.00%** | **81.10%** | **88.30%** | **80.85%** | **95.08%** | **95.01%** |

### Key Architecture Modifications
* **Preserving Early Spatial Resolution:** Standard `MobileNetV3-Small` uses a stride-2 initial stem convolution designed for $224\times224$ ImageNet inputs, which downsamples $32\times32$ images too aggressively. Modifying `features[0][0].stride` from `(2, 2)` to `(1, 1)` preserves spatial feature maps across early inverted residual blocks.
* **Joint Multi-Task Objective:** Trained with $\mathcal{L} = \mathcal{L}_{\text{object}} + 0.6 \mathcal{L}_{\text{corruption}}$ using AdamW and cosine annealing.

---

## 2. Quantization Engineering: Solving Depthwise Convolution Collapse

When applying naive per-tensor static INT8 Post-Training Quantization (PTQ) to MobileNetV3, accuracy collapsed from **85.80% to 18.70%** due to $3\times3$ depthwise separable convolutions having only 9 weights per channel with wide inter-channel variance.

**Engineering Fix:**
1. Applied ONNX graph pre-processing (`quant_pre_process`) for Conv/BatchNorm fusion and shape inference.
2. Enabled **per-channel symmetric weight quantization** (`per_channel=True, reduce_range=True`).
3. Excluded the 12 sensitive depthwise (`group > 1`) and stem `Conv` nodes (`<2%` of total parameters) while quantizing all $1\times1$ pointwise convolutions and linear layers to INT8.

| Model Precision | Model Size | Compression | Clean Acc (2k Eval) | Corrupted Acc | Detect Macro-F1 | C++ ORT p50 Latency | C++ ORT p90 Latency | Snapdragon 8 Gen 3 (S24) Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 ONNX** | 4.40 MB | 1.00x | 85.80% | 82.19% | 95.04% | 1.28 ms | 4.29 ms | **0.116 ms** |
| **Naive INT8 (Per-Tensor)** | 1.29 MB | 3.42x | 18.70% | 16.00% | 20.41% | 0.78 ms | 1.08 ms | — |
| **Selective Per-Channel INT8** | **1.52 MB** | **2.90x (-65.5%)** | **85.50% (-0.30%)** | **80.25%** | **94.69% (-0.35%)** | **0.86 ms (1.49x faster)** | **0.99 ms (4.33x faster)** | — |

* **Qualcomm AI Hub On-Device Profile:** Profiled on a cloud-hosted **Samsung Galaxy S24 (Snapdragon 8 Gen 3)** achieving **0.116 ms** inference latency (`~8,620 inf/s`, Compile Job: `jp42v14qg`, Profile Job: `j5ml3mkyg`).

---

## 3. Repository Structure

```text
CorruptDetect/
├── src/
│   ├── model.py              # Modified MobileNetV3-Small dual-head architecture
│   ├── dataset.py            # Zero-leakage train/val/test split + seeded corruptions
│   ├── train.py              # Baseline & multi-task robust training loop
│   ├── evaluate.py           # Clean, per-corruption, and Macro-F1 evaluation harness
│   ├── export_quantize.py    # ONNX export, graph pre-processing, selective per-channel INT8 PTQ
│   └── profile_qai_hub.py    # Qualcomm AI Hub compile & Snapdragon S24 profiling script
├── cpp/
│   ├── CMakeLists.txt        # CMake build configuration for OpenCV + ONNX Runtime C++ API
│   └── main.cpp              # C++17 NCHW preprocessing & 1,000-run p50/p90 latency benchmark
├── models/
│   ├── corrupt_detect_fp32.onnx
│   └── corrupt_detect_int8.onnx
└── requirements.txt
