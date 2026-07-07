import pytest, torch

def pytest_report_header(config):
    if torch.cuda.is_available():
        return [
            f"CUDA available: {torch.cuda.is_available()}",
            f"Device: {torch.cuda.get_device_name(0)}, count={torch.cuda.device_count()}",
        ]
    return ["CUDA not available"]

def pytest_collection_modifyitems(config, items):
    for item in items:
        if "cuda" in item.keywords and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="CUDA not available"))
