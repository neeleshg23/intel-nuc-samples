import torch
import intel_extension_for_pytorch as ipex
import json
import argparse
import gc
import sys

def print_device_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"Intel Extension for PyTorch version: {ipex.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"XPU available: {torch.xpu.is_available()}")
    if torch.xpu.is_available():
        print(f"XPU device count: {torch.xpu.device_count()}")
        print(f"Current XPU device: {torch.xpu.current_device()}")
        print(f"XPU device name: {torch.xpu.get_device_name()}")

def get_event_dict(event):
    event_dict = {}
    for attr in dir(event):
        if not attr.startswith('_') and not callable(getattr(event, attr)):
            try:
                value = getattr(event, attr)
                if isinstance(value, (int, float, str, bool)):
                    event_dict[attr] = value
                elif isinstance(value, torch.Size):
                    event_dict[attr] = str(value)
            except Exception:
                pass  # Ignore attributes that can't be easily serialized
    return event_dict

def profile(inC, outC, batch, dtype, n_iters=100):
    print_device_info()

    if dtype == "float32":
        torch_dtype = torch.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        raise RuntimeError(f"Invalid dtype: {dtype}")

    try:
        # Create input tensor and model
        X = torch.randn(batch, inC, dtype=torch_dtype).to('xpu')
        W = torch.randn(outC, inC, dtype=torch_dtype).to('xpu')

        print(f"Input shape: {X.shape}")
        print(f"Weight shape: {W.shape}")

        # Try different methods
        print("Trying different matrix multiplication methods:")

        print("1. Using torch.matmul")
        result = torch.matmul(X, W.t())
        print("torch.matmul successful")

        print("\n2. Using @")
        result = X @ W.t()
        print("@ operator successful")

        print("\n3. Using nn.Linear")
        model = torch.nn.Linear(inC, outC, bias=False).to('xpu')
        model.weight.data = W
        result = model(X)
        print("nn.Linear successful")

        print("\n4. Using F.linear")
        import torch.nn.functional as F
        result = F.linear(X, W)
        print("F.linear successful")

        # Profiling
        print("\nStarting profiling...")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(n_iters):
                with torch.no_grad():
                    result = torch.matmul(X, W.t())
                torch.xpu.synchronize()
        print("Profiling completed.")

        # Process profiling results
        events = [get_event_dict(event) for event in prof.events()]

        # Create summary statistics
        key_averages = prof.key_averages()
        summary = {}
        for event in key_averages:
            for attr, value in get_event_dict(event).items():
                if isinstance(value, (int, float)):
                    summary[f"total_{attr}"] = summary.get(f"total_{attr}", 0) + value

        # Compile final profiling data
        profiling_data = {
            "events": events,
            "summary": summary,
            "config": {
                "batch": batch,
                "input_channels": inC,
                "output_channels": outC,
                "dtype": dtype,
                "iterations": n_iters,
            }
        }

        # Write profiling data to JSON file
        with open("profiling.json", "w") as fp:
            json.dump(profiling_data, fp, indent=2)

        print(f"MatMul profiling completed. Detailed results saved to 'profiling.json'")
        if 'total_cpu_time' in summary and 'total_xpu_time' in summary:
            print(f"Total CPU time: {summary['total_cpu_time']:.3f} us, Total XPU time: {summary['total_xpu_time']:.3f} us")

        return profiling_data

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def define_and_parse_args():
    parser = argparse.ArgumentParser(description="Profiling a MatMul model on Intel XPU")
    parser.add_argument("--batch", "-b", type=int, required=True, help="MatMul batch")
    parser.add_argument("--input-channels", "-c", type=int, required=True, help="MatMul input channels")
    parser.add_argument("--output-channels", "-k", type=int, required=True, help="MatMul output channels")
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"],
                        help="Select the target dtype (default: %(default)s)")
    parser.add_argument("--iterations", "-i", type=int, default=100, help="Number of iterations for profiling")
    return parser.parse_args()

if __name__ == "__main__":
    args = define_and_parse_args()
    profile(args.input_channels, args.output_channels, args.batch, dtype=args.dtype, n_iters=args.iterations)
    gc.collect()
