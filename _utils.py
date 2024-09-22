import torch


def has_affordable_memory(device: torch.device) -> bool:
    free_memory, _ = torch.cuda.mem_get_info(device)
    free_memory_gb = free_memory / (1024**3)
    return free_memory_gb > 24


def is_newer_than_ada_lovelace(device: torch.device) -> int:
    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    return cc_major * 10 + cc_minor >= 89
