"""Measure peak inference memory of an ONNX model in a fresh process.

    python -m src.nas.memprobe model.onnx 32 [runs] [threads]

Prints JSON: peak RSS of the process (MB), RSS after the session is created, and the extra
memory used while running inference. Running in a separate process keeps the training
process's memory (PyTorch, CUDA, datasets) out of the measurement.

Peak RSS comes from /proc/self/status VmHWM on Linux, getrusage(ru_maxrss) on macOS and the peak
working set on Windows.
"""
import json
import sys


def peak_rss_mb():
    # Linux: VmHWM belongs to this process's address space and is reset by exec. (ru_maxrss is NOT:
    # it is inherited from the parent across fork/exec, so it would report the training process's peak.)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except OSError:
        pass
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KB on Linux
    except ImportError:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2 ** 20


def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 2 ** 20


def main():
    path, res = sys.argv[1], int(sys.argv[2])
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    import numpy as np
    import onnxruntime as ort
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, res, res).astype(np.float32)
    name = sess.get_inputs()[0].name
    loaded = rss_mb()
    peak_before = peak_rss_mb()
    for _ in range(runs):
        sess.run(None, {name: x})
    after = rss_mb()
    peak = peak_rss_mb()
    # If inference pushed the high-water mark up, the peak happened during inference;
    # otherwise the best estimate of inference memory is the resident growth.
    infer = peak - loaded if peak > peak_before else after - loaded
    print(json.dumps({"peak_rss_mb": round(peak, 2), "session_rss_mb": round(loaded, 2),
                      "inference_mem_mb": round(max(0.0, infer), 2)}))


if __name__ == "__main__":
    main()
