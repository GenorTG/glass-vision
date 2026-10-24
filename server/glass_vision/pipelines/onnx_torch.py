"""Run an ONNX graph with PyTorch ops — for the PP-OCR models (`ppocr.py`), not a general ONNX runtime.

Why not ONNX Runtime: with onnxruntime-gpu 1.31 + cuDNN 9 on this server, every *new input shape* costs the CUDA
execution provider 2–7 s per model before the first result (its Conv kernels build cuDNN execution plans for many
engine configs; profiled: >99 % of the time in Conv; nothing is cached across sessions, with any
`cudnn_conv_algo_search` that keeps the steady speed). OCR feeds a new shape almost every call (one detector input
per image size, recognizer batches of varying widths, ten recognizers), so reads took seconds instead of ~0.1 s.
torch's cuDNN path costs ~0.1–0.3 s for a new shape and nothing after, and torch is already loaded for translation.
The ONNX files (PaddlePaddle's official exports) stay the model format: weights are read from them unchanged.

Only the operators those graphs use are implemented, with ONNX semantics for the opsets they declare (7, 11, 14);
anything else raises at load time, so a changed upstream export fails loudly instead of computing something else.
`tests/test_pipeline_text.py` checks the outputs against onnxruntime (CPU) on every model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def _same_pads(x: Tensor, kernel: Any, strides: Any, auto_pad: bytes | str) -> tuple[int, int, int, int]:
    """ONNX auto_pad SAME_UPPER / SAME_LOWER → (top, left, bottom, right): output size = ceil(input / stride)."""
    if auto_pad in (b"VALID", "VALID"):
        return 0, 0, 0, 0
    upper = auto_pad in (b"SAME_UPPER", "SAME_UPPER")
    begin, end = [], []
    for size, k, s in zip(x.shape[2:], kernel, strides, strict=True):
        total = max((-(-size // s) - 1) * s + k - size, 0)
        small, large = total // 2, total - total // 2
        begin.append(small if upper else large)
        end.append(large if upper else small)
    return begin[0], begin[1], end[0], end[1]


class OnnxTorch:
    """One ONNX graph (single input) executed node by node with torch ops on `device`."""

    def __init__(self, path: Path, device: str, *, half: bool = False) -> None:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        self.opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 13)
        self.device = torch.device(device)
        self.half = half and self.device.type == "cuda"  # fp16 autocast (convolutions, matmuls)
        graph = model.graph
        self.values: dict[str, Any] = {
            t.name: self._tensor(numpy_helper.to_array(t)) for t in graph.initializer
        }  # fmt: skip
        known = set(self.values)
        self.input = next(i.name for i in graph.input if i.name not in known)
        self.outputs = [o.name for o in graph.output]
        self.nodes: list[tuple[Callable[..., Any], list[str], list[str], dict[str, Any], list[str]]] = []
        alias: dict[str, str] = {}
        for node in graph.node:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if node.op_type == "Constant":
                self.values[node.output[0]] = self._tensor(numpy_helper.to_array(attrs["value"]))
                continue
            if node.op_type == "Identity":  # aliases: resolved here instead of a call per run
                alias[node.output[0]] = alias.get(node.input[0], node.input[0])
                continue
            op = getattr(self, f"_op_{node.op_type}", None)
            if op is None:
                raise NotImplementedError(f"{path.parent.name}: ONNX op {node.op_type} not supported")
            inputs = [alias.get(name, name) for name in node.input]
            self.nodes.append((op, inputs, list(node.output), attrs, []))
        self.outputs = [alias.get(name, name) for name in self.outputs]
        # Free every intermediate right after its last consumer: holding them all costs ~18 GB on a 2560 px image.
        last_use: dict[str, int] = {}
        for k, (_, inputs, _, _, _) in enumerate(self.nodes):
            for name in inputs:
                if name and name not in self.values:
                    last_use[name] = k
        for name, k in last_use.items():
            if name not in self.outputs:
                self.nodes[k][4].append(name)

    def _tensor(self, array: np.ndarray) -> Tensor:
        # Integer tensors (shapes, axes, slice bounds) stay on the CPU: they are read back as Python ints.
        t = torch.from_numpy(np.array(array))
        return t if t.dtype in (torch.int64, torch.int32) else t.to(self.device)

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        return [y.float().cpu().numpy() for y in self.run_tensor(torch.from_numpy(x))]

    @torch.inference_mode()
    def run_tensor(self, x: Tensor) -> list[Tensor]:
        if self.half:
            with torch.autocast("cuda", dtype=torch.float16):
                return [y.float() for y in self._run(x)]
        return self._run(x)

    def _run(self, x: Tensor) -> list[Tensor]:
        values = dict(self.values)
        values[self.input] = x.to(self.device)
        for op, inputs, outputs, attrs, done in self.nodes:
            result = op(*[values[name] if name else None for name in inputs], **attrs)
            if len(outputs) == 1:
                values[outputs[0]] = result
            else:
                for name, value in zip(outputs, result, strict=False):
                    values[name] = value
            for name in done:
                del values[name]
        return [values[name] for name in self.outputs]

    # --- elementwise ----------------------------------------------------------------------------------------------
    @staticmethod
    def _op_Identity(x: Tensor) -> Tensor:
        return x

    @staticmethod
    def _op_Add(a: Tensor, b: Tensor) -> Tensor:
        return a + b

    @staticmethod
    def _op_Sub(a: Tensor, b: Tensor) -> Tensor:
        return a - b

    @staticmethod
    def _op_Mul(a: Tensor, b: Tensor) -> Tensor:
        return a * b

    @staticmethod
    def _op_Div(a: Tensor, b: Tensor) -> Tensor:
        if not a.is_floating_point() and not b.is_floating_point():
            return torch.div(a, b, rounding_mode="trunc")
        return a / b

    @staticmethod
    def _op_Pow(a: Tensor, b: Tensor) -> Tensor:
        return torch.pow(a, b.to(a.device))

    @staticmethod
    def _op_Sqrt(x: Tensor) -> Tensor:
        return torch.sqrt(x)

    @staticmethod
    def _op_Erf(x: Tensor) -> Tensor:
        return torch.erf(x)

    @staticmethod
    def _op_Relu(x: Tensor) -> Tensor:
        return torch.relu(x)

    @staticmethod
    def _op_Sigmoid(x: Tensor) -> Tensor:
        return torch.sigmoid(x)

    @staticmethod
    def _op_HardSigmoid(x: Tensor, alpha: float = 0.2, beta: float = 0.5) -> Tensor:
        return torch.clamp(x * alpha + beta, 0.0, 1.0)

    @staticmethod
    def _op_MatMul(a: Tensor, b: Tensor) -> Tensor:
        return torch.matmul(a, b)

    def _op_Softmax(self, x: Tensor, axis: int | None = None) -> Tensor:
        if self.opset >= 13:
            return torch.softmax(x, dim=-1 if axis is None else axis)
        axis = 1 if axis is None else axis % x.dim()  # opset < 13: softmax over all dims from `axis` on, flattened
        if axis == x.dim() - 1:
            return torch.softmax(x, dim=-1)
        shape = x.shape
        return torch.softmax(x.reshape(int(np.prod(shape[:axis], dtype=np.int64)), -1), dim=1).reshape(shape)

    # --- convolution / normalization / pooling --------------------------------------------------------------------
    @staticmethod
    def _op_Conv(
        x: Tensor,
        w: Tensor,
        b: Tensor | None = None,
        *,
        dilations: list[int] = (1, 1),  # type: ignore[assignment]
        group: int = 1,
        pads: list[int] = (0, 0, 0, 0),  # type: ignore[assignment]
        strides: list[int] = (1, 1),  # type: ignore[assignment]
        kernel_shape: list[int] | None = None,
        auto_pad: bytes = b"NOTSET",
    ) -> Tensor:
        if auto_pad in (b"NOTSET", "NOTSET"):
            top, left, bottom, right = pads
        else:
            k = w.shape[2:] if kernel_shape is None else kernel_shape
            dilated = [(kk - 1) * d + 1 for kk, d in zip(k, dilations, strict=True)]
            top, left, bottom, right = _same_pads(x, dilated, strides, auto_pad)
        if (top, left) != (bottom, right):
            x = F.pad(x, (left, right, top, bottom))
            top = left = 0
        return F.conv2d(x, w, b, tuple(strides), (top, left), tuple(dilations), group)

    @staticmethod
    def _op_ConvTranspose(
        x: Tensor,
        w: Tensor,
        b: Tensor | None = None,
        *,
        dilations: list[int] = (1, 1),  # type: ignore[assignment]
        group: int = 1,
        pads: list[int] = (0, 0, 0, 0),  # type: ignore[assignment]
        strides: list[int] = (1, 1),  # type: ignore[assignment]
        output_padding: list[int] = (0, 0),  # type: ignore[assignment]
        kernel_shape: list[int] | None = None,
    ) -> Tensor:
        top, left, bottom, right = pads
        if (top, left) != (bottom, right):
            raise NotImplementedError("ConvTranspose with asymmetric pads")
        return F.conv_transpose2d(x, w, b, tuple(strides), (top, left), tuple(output_padding), group, tuple(dilations))

    @staticmethod
    def _op_BatchNormalization(
        x: Tensor, scale: Tensor, bias: Tensor, mean: Tensor, var: Tensor, epsilon: float = 1e-5, **_: Any
    ) -> Tensor:
        return F.batch_norm(x, mean, var, scale, bias, False, 0.0, epsilon)

    @staticmethod
    def _op_GlobalAveragePool(x: Tensor) -> Tensor:
        return x.mean(dim=(2, 3), keepdim=True)

    @staticmethod
    def _op_AveragePool(
        x: Tensor,
        *,
        kernel_shape: list[int],
        strides: list[int] | None = None,
        pads: list[int] = (0, 0, 0, 0),  # type: ignore[assignment]
        ceil_mode: int = 0,
        count_include_pad: int = 0,
    ) -> Tensor:
        top, left, bottom, right = pads
        if (top, left) != (bottom, right):
            raise NotImplementedError("AveragePool with asymmetric pads")
        return F.avg_pool2d(
            x,
            tuple(kernel_shape),
            tuple(strides or kernel_shape),
            (top, left),
            bool(ceil_mode),
            bool(count_include_pad),
        )

    @staticmethod
    def _op_MaxPool(
        x: Tensor,
        *,
        kernel_shape: list[int],
        strides: list[int] | None = None,
        pads: list[int] | None = None,
        auto_pad: bytes = b"NOTSET",
        ceil_mode: int = 0,
        dilations: list[int] = (1, 1),  # type: ignore[assignment]
        storage_order: int = 0,
    ) -> Tensor:
        strides = list(strides or [1, 1])
        if auto_pad in (b"NOTSET", "NOTSET"):
            top, left, bottom, right = pads or (0, 0, 0, 0)
        else:
            top, left, bottom, right = _same_pads(x, kernel_shape, strides, auto_pad)
        x = F.pad(x, (left, right, top, bottom), value=float("-inf"))
        return F.max_pool2d(x, tuple(kernel_shape), tuple(strides), 0, tuple(dilations), bool(ceil_mode))

    def _op_ReduceMean(self, x: Tensor, axes_input: Tensor | None = None, *, axes=None, keepdims: int = 1) -> Tensor:
        axes = axes if axes is not None else (axes_input.tolist() if axes_input is not None else None)
        if axes is None:
            return x.mean(keepdim=bool(keepdims)) if keepdims else x.mean()
        return x.mean(dim=tuple(axes), keepdim=bool(keepdims))

    @staticmethod
    def _op_Resize(
        x: Tensor,
        roi: Tensor | None = None,
        scales: Tensor | None = None,
        sizes: Tensor | None = None,
        *,
        mode: bytes = b"nearest",
        coordinate_transformation_mode: bytes = b"half_pixel",
        nearest_mode: bytes = b"round_prefer_floor",
        **_: Any,
    ) -> Tensor:
        if (mode, coordinate_transformation_mode, nearest_mode) != (b"nearest", b"asymmetric", b"floor"):
            raise NotImplementedError(f"Resize {mode!r}/{coordinate_transformation_mode!r}/{nearest_mode!r}")
        if sizes is not None and sizes.numel():
            return F.interpolate(x, size=[int(v) for v in sizes.tolist()[2:]], mode="nearest")
        factors = [float(v) for v in scales.tolist()[2:]]  # type: ignore[union-attr]
        size = [int(np.floor(d * f)) for d, f in zip(x.shape[2:], factors, strict=True)]
        return F.interpolate(x, size=size, scale_factor=None, mode="nearest") if factors[0] != factors[1] else (
            F.interpolate(x, scale_factor=factors[0], mode="nearest")
        )  # fmt: skip

    # --- shapes ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _op_Shape(x: Tensor) -> Tensor:
        return torch.tensor(list(x.shape), dtype=torch.int64)

    @staticmethod
    def _op_Concat(*xs: Tensor, axis: int) -> Tensor:
        if all(x.device.type == "cpu" and not x.is_floating_point() for x in xs):
            return torch.cat(list(xs), dim=axis)
        device = next(x.device for x in xs if x.device.type != "cpu" or x.is_floating_point())
        return torch.cat([x.to(device) for x in xs], dim=axis)

    @staticmethod
    def _op_Reshape(x: Tensor, shape: Tensor, allowzero: int = 0) -> Tensor:
        dims = shape.tolist()
        if not allowzero:
            dims = [x.shape[i] if d == 0 else d for i, d in enumerate(dims)]
        return x.reshape(dims)

    @staticmethod
    def _op_Transpose(x: Tensor, perm: list[int]) -> Tensor:
        return x.permute(*perm)

    def _op_Squeeze(self, x: Tensor, axes_input: Tensor | None = None, *, axes=None) -> Tensor:
        axes = axes if axes is not None else (axes_input.tolist() if axes_input is not None else None)
        if axes is None:
            return x.squeeze()
        for axis in sorted((a % x.dim() for a in axes), reverse=True):
            x = x.squeeze(axis)
        return x

    def _op_Unsqueeze(self, x: Tensor, axes_input: Tensor | None = None, *, axes=None) -> Tensor:
        axes = axes if axes is not None else axes_input.tolist()  # type: ignore[union-attr]
        rank = x.dim() + len(axes)
        for axis in sorted(a % rank for a in axes):
            x = x.unsqueeze(axis)
        return x

    def _op_Slice(
        self,
        x: Tensor,
        starts_in: Tensor | None = None,
        ends_in: Tensor | None = None,
        axes_in: Tensor | None = None,
        steps_in: Tensor | None = None,
        *,
        starts=None,
        ends=None,
        axes=None,
    ) -> Tensor:
        starts = starts if starts is not None else starts_in.tolist()  # type: ignore[union-attr]
        ends = ends if ends is not None else ends_in.tolist()  # type: ignore[union-attr]
        if axes is None:
            axes = axes_in.tolist() if axes_in is not None else list(range(len(starts)))
        steps = steps_in.tolist() if steps_in is not None else [1] * len(starts)
        index: list[slice] = [slice(None)] * x.dim()
        for start, end, axis, step in zip(starts, ends, axes, steps, strict=True):
            if step != 1:
                raise NotImplementedError("Slice with step != 1")
            size = x.shape[axis]
            start = max(0, min(size, start + size if start < 0 else start))
            end = max(0, min(size, end + size if end < 0 else end))
            index[axis % x.dim()] = slice(start, end)
        return x[tuple(index)]
