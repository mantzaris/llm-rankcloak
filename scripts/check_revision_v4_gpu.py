"""One bounded local CUDA generation/replay check. Never falls back to CPU."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from rankcloak.reproducibility import sha256_file
from rankcloak.revision_v4_stage1 import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/revision_v4/provenance/gpu_smoke.json'
GPU = 'GPU-10d1f16f-9e79-08bb-b2ba-3353c04422cf'


def query(arguments):
    return subprocess.check_output(['nvidia-smi'] + arguments, text=True, timeout=10).strip()


def busy():
    rows = query(['--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'])
    return [r for r in rows.splitlines() if r.startswith(GPU + ',') and f', {os.getpid()},' not in r]


def main():
    started = time.monotonic()
    report = {'schema_version':'rankcloak-v4-gpu-smoke-v1', 'gpu_uuid':GPU,
              'pid':os.getpid(), 'maximum_wall_seconds':570, 'scope':'16 forced tokens and 16 replayed tokens, one Qwen Q4 model',
              'source_sha256':sha256_file(Path(__file__)), 'historical_results_changed':False}
    stop = threading.Event()

    def deadline():
        if not stop.wait(570):
            write_json(OUT, {**report, 'status':'timeout', 'elapsed_seconds':time.monotonic()-started})
            os._exit(124)

    threading.Thread(target=deadline, daemon=True).start()
    samples = []
    try:
        report['gpu_inventory'] = query(['--query-gpu=index,uuid,name,memory.total,memory.used,driver_version', '--format=csv'])
        report['other_compute_processes_before'] = busy()
        if report['other_compute_processes_before']:
            report['status'] = 'skipped_busy_device'
            return
        requirements = json.loads((ROOT/'configs/revision_v3/generation_requirements.json').read_text())
        report['backend_packages'] = {name:importlib.metadata.version(name) for name in ['llama-cpp-python','nvidia-cuda-runtime-cu12','nvidia-cublas-cu12']}
        if report['backend_packages']['llama-cpp-python'] != requirements['required_backend']['version']:
            raise RuntimeError('pinned backend version mismatch')
        artifact = next(a for a in requirements['artifacts'] if a['model_id']=='qwen2_5_7b_instruct_q4_k_m')
        path = ROOT / artifact['expected_path']
        report['model'] = artifact
        report['observed_model_sha256'] = sha256_file(path)
        if not path.exists() or path.stat().st_size != artifact['size_bytes'] or report['observed_model_sha256'] != artifact['sha256']:
            raise RuntimeError('pinned model bytes unavailable or mismatched')
        if busy():
            report['status'] = 'skipped_device_became_busy'
            return
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
        from rankcloak.model_io import load_llama_cpp_model, llama_cpp_gpu_offload_supported, make_context_token_ids
        if not llama_cpp_gpu_offload_supported():
            raise RuntimeError('CUDA offload backend unavailable')
        from llama_cpp import llama_cpp
        report['llama_cpp_system_info'] = llama_cpp.llama_print_system_info().decode()
        report['gpu_offload_supported'] = True

        def monitor():
            while not stop.is_set():
                try:
                    rows = query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
                    for line in rows.splitlines():
                        parts = [v.strip() for v in line.split(',')]
                        if len(parts)==3 and parts[0]==GPU and parts[1]==str(os.getpid()):
                            samples.append({'seconds':time.monotonic()-started,'process_gpu_memory_mib':int(parts[2])})
                except (ValueError, subprocess.SubprocessError):
                    pass
                stop.wait(0.25)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        execution_started = time.monotonic()
        model = load_llama_cpp_model(path, n_ctx=256, n_threads=4, n_gpu_layers=-1, logits_all=True, verbose=True)
        try:
            from rankcloak.rank_codec import generate_token_ids_from_ranks, recover_ranks_from_generated_ids
            ranks = list(range(1,17))
            context = make_context_token_ids(model, 'Explain how a public library serves its community.')
            generated = generate_token_ids_from_ranks(model, context, ranks)
            replay = recover_ranks_from_generated_ids(model, context, generated['generated_token_ids'])
            report.update({'context_token_ids':context,'requested_ranks':ranks,
                'generated_token_ids':generated['generated_token_ids'],'generated_text':generated['generated_text'],
                'recovered_ranks':replay['ranks'], 'exact_rank_replay':ranks==replay['ranks'],
                'n_gpu_layers':model.rankcloak_n_gpu_layers,'n_batch':model.rankcloak_n_batch,
                'n_ubatch':model.rankcloak_n_ubatch,
                'execution_including_load_seconds':time.monotonic()-execution_started,
                'deterministic_environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUDA_LAUNCH_BLOCKING','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION','GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F','CUBLAS_WORKSPACE_CONFIG']}})
            if ranks != replay['ranks']:
                raise RuntimeError('smoke rank replay mismatch')
        finally:
            model.close()
        stop.set()
        monitor_thread.join(timeout=2)
        report['gpu_memory_samples'] = samples
        report['sampled_peak_process_gpu_memory_mib'] = max((s['process_gpu_memory_mib'] for s in samples),default=None)
        report['memory_scope'] = 'nvidia-smi process allocation sampled every 0.25 seconds, not an allocator-exact peak'
        if not samples:
            raise RuntimeError('no process-specific CUDA allocation observed')
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'unavailable_or_failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        stop.set()
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(OUT, report)
        print(json.dumps({k:report[k] for k in ['status','elapsed_seconds']}))


if __name__ == '__main__':
    main()
