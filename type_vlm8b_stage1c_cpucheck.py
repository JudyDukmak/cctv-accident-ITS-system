"""Stage 1c, load-only check (no frames, no generate, no answers): which modules are stored on the CPU, and how.
type_vlm8b_stage1c.py's after-load listing was empty because accelerate offloads the CPU entries of the device_map:
their parameters in the model are on the meta device and the weights are kept on the CPU in the module's AlignDevicesHook
weights_map. This script loads with the AMENDMENT_2 step 1 and step 2 device maps (same load function) and records,
for every module with an offload hook: execution device, offload flag, and device / dtype / shape of the stored weights.
Writes results/type_vlm8b/stage1c_cpu_check.json (refuses to overwrite). Run with .venv-vlm, HF_HUB_OFFLINE=1.
"""
import gc
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import type_vlm8b as T
import type_vlm8b_stage1c as C

OUT_JSON = T.OUT / "stage1c_cpu_check.json"


def main():
    import torch
    T.check_plan()
    T.refuse(OUT_JSON)
    res = {"hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "steps": []}
    for st in C.STEPS[:2]:
        rs = {"step": st["step"], "cpu": st["cpu"]}
        model = C.load(st["cpu"], rs)
        rs["hf_device_map"] = {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()}
        off = {}
        for name, mod in model.named_modules():
            hook = getattr(mod, "_hf_hook", None)
            if hook is None or not getattr(hook, "offload", False):
                continue
            wm = getattr(hook, "weights_map", None)
            stored = {}
            if wm is not None:
                # PrefixedDataset: iterating yields the full keys, so read the underlying loader with them
                ds = getattr(wm, "dataset", wm)
                for k in list(iter(wm)):
                    t = ds[k]
                    sd = getattr(ds, "state_dict", {})
                    idx = getattr(ds, "index", {})
                    stored[k] = {"device": str(t.device), "dtype": str(t.dtype), "shape": list(t.shape),
                                 "loader": type(ds).__name__, "held_in_cpu_state_dict": k in sd,
                                 "index_entry": {a: str(b) for a, b in idx.get(k, {}).items()}}
                    del t
            off[name] = {"hook": type(hook).__name__, "execution_device": str(hook.execution_device),
                         "offload": hook.offload, "param_devices_in_model": sorted({str(p.device) for p in mod.parameters(recurse=False)}),
                         "stored_weights": stored}
        rs["offloaded_modules"] = off
        rs["gpu_params_by_dtype_gb"] = {}
        for p in model.parameters():
            if p.device.type == "cuda":
                k = str(p.dtype)
                rs["gpu_params_by_dtype_gb"][k] = rs["gpu_params_by_dtype_gb"].get(k, 0) + p.numel() * p.element_size() / 2 ** 30
        print(json.dumps({k: rs[k] for k in ("step", "hf_device_map", "offloaded_modules", "mem_after_load_alloc_gb",
                                             "linear4bit_modules")}, indent=1, default=str))
        res["steps"].append(rs)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    OUT_JSON.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
