import torch
import qai_hub as hub
from model import CorruptDetectNet

def main():
    # 1. Load your trained Robust MobileNetV3 model and trace it
    model = CorruptDetectNet(pretrained=False)
    model.load_state_dict(torch.load("./models/robust_best.pth", map_location="cpu", weights_only=True))
    model.eval()

    input_shape = (1, 3, 32, 32)
    example_input = torch.randn(input_shape)
    pt_trace = torch.jit.trace(model, example_input)

    # 2. Target a physical cloud-hosted Samsung Galaxy S24 (Snapdragon 8 Gen 3)
    device = hub.Device("Samsung Galaxy S24 (Family)")

    # 3. Submit compilation job to Qualcomm AI Hub
    print("Submitting compile job to Qualcomm AI Hub...")
    compile_job = hub.submit_compile_job(
        model=pt_trace,
        device=device,
        input_specs=dict(input=input_shape),
    )
    target_model = compile_job.get_target_model()

    # 4. Submit on-device profiling job on Snapdragon hardware
    print("Submitting on-device profile job on Samsung Galaxy S24...")
    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
    )

    print(f"\nDashboard URL: {profile_job.url}")
    print("Waiting for physical device profiling to finish (usually 1-2 mins)...")
    profile_data = profile_job.download_profile()

    exec_summary = profile_data.get("execution_summary", {})
    est_time_us = exec_summary.get("estimated_inference_time", 0)
    peak_mem_bytes = exec_summary.get("inference_memory_peak_range", [0, 0])

    print("\n==============================================")
    print("   QUALCOMM SNAPDRAGON (S24) PROFILE SUMMARY  ")
    print("==============================================")
    print(f"Device              : Samsung Galaxy S24 (Snapdragon 8 Gen 3)")
    print(f"Estimated Latency   : {est_time_us / 1000.0:.3f} ms")
    print(f"Peak Memory Range   : {peak_mem_bytes}")
    print(f"Full Dashboard Link : {profile_job.url}")
    print("==============================================")

if __name__ == "__main__":
    main()
