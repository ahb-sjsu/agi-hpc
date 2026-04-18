"""ONNX-direct strategies for Erebus.

Instead of generating Python transforms and then trying to compile
them to ONNX, these strategies ask the LLM to write onnx.helper
code directly. The output IS the submission format.

The LLM gets:
  - Task examples (input/output grids)
  - ONNX opset 10 constraints
  - Available ops list
  - The compiler primitives library as reference
  - Prior failure context (if any)

Output: build_onnx() function returning onnx.ModelProto
"""

ONNX_CONSTRAINTS = """ONNX CONSTRAINTS:
- Input: 'input' shape [1,10,30,30] float32 one-hot (channel c = 1.0 where color=c)
- Output: 'output' shape [1,10,30,30] float32 one-hot
- Opset 10, ir_version 10
- Grid content is in top-left of 30x30 canvas, rest is zeros
- ALLOWED ops: Conv, Gather, Reshape, Concat, Pad, Slice, Mul, Add,
  Relu, ReduceMax, ArgMax, Squeeze, Unsqueeze, Transpose, Identity,
  Clip, Cast, Sub, MatMul
- BANNED ops: Loop, Scan, If, NonZero, Unique, Where, Resize, Script, Function
- Max file size: 1.44MB
- Scoring: max(1, 25 - ln(MACs + memory + params))
  Lower cost = higher score. Prefer small graphs."""

ONNX_DIRECT = (
    "Write a Python function build_onnx() that returns an onnx.ModelProto "
    "solving this ARC task.\n\n"
    "{examples}\n\n"
    f"{ONNX_CONSTRAINTS}\n\n"
    "Use onnx.helper to build the graph manually:\n"
    "  from onnx import TensorProto, helper\n"
    "  import numpy as np\n\n"
    "TECHNIQUES that work in opset 10:\n"
    "- Dynamic grid detection: ReduceMax on non-bg channels + ArgMax\n"
    "- Color remap: 1x1 Conv with permutation weight matrix\n"
    "- Spatial remap (flip/rotate/transpose/crop): Flatten to [1,C,H*W], "
    "Gather with computed indices, Reshape back\n"
    "- Dynamic masking: Clip(Cast(diff)+1, 0, 1) for >= 0 condition\n"
    "- INT64 clamping: Cast(INT64->FLOAT) -> Clip -> Cast(FLOAT->INT64)\n"
    "- Tiling: Concat copies along axis\n\n"
    "Write ONLY the build_onnx() function. ```python ... ```"
)

ONNX_WITH_ANALYSIS = (
    "You analyzed this ARC task and found: {analysis}\n\n"
    "Now write a Python function build_onnx() that implements this "
    "transformation as an ONNX model.\n\n"
    "{examples}\n\n"
    f"{ONNX_CONSTRAINTS}\n\n"
    "Write ONLY the build_onnx() function. ```python ... ```"
)

ONNX_FROM_PYTHON = (
    "This Python transform solves an ARC task correctly:\n"
    "```python\n{python_code}\n```\n\n"
    "Convert it to an equivalent ONNX model. The ONNX model must "
    "produce identical output for the same inputs.\n\n"
    f"{ONNX_CONSTRAINTS}\n\n"
    "Write ONLY the build_onnx() function. ```python ... ```"
)

ONNX_DIAGNOSTIC = (
    "Previous ONNX attempt failed: {diagnosis}\n"
    "Error type: {error_type}\n\n"
    "Try a DIFFERENT ONNX graph structure:\n\n"
    "{examples}\n\n"
    f"{ONNX_CONSTRAINTS}\n\n"
    "Write ONLY the build_onnx() function. ```python ... ```"
)
