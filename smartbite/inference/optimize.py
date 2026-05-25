from __future__ import annotations
import torch
import torch.nn as nn


def optimize_torchscript(model: nn.Module, output_path: str, example_input: torch.Tensor) -> str:
    model.eval()
    traced = torch.jit.trace(model, example_input)
    traced.save(output_path)
    return output_path


def optimize_onnx(model: nn.Module, output_path: str, example_input: torch.Tensor) -> str:
    model.eval()
    torch.onnx.export(
        model, example_input, output_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
    )
    return output_path


def quantize_model(model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    prepared = torch.quantization.prepare(model, inplace=False)
    with torch.no_grad():
        for _ in range(10):
            prepared(calibration_data)
    return torch.quantization.convert(prepared, inplace=False)
