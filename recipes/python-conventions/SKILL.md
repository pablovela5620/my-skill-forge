---
name: python-conventions
description: Python conventions for every repo — typing, dataclasses, return types, docstrings. Use whenever touching Python code.
---

# Python Conventions

Every section below is a check, not a suggestion: code isn't done until it
passes all of them, and a review applies every one.

## Runtime type checking (beartype)

Projects activate beartype conditionally on the pixi environment:

```python
if os.environ.get("PIXI_DEV_MODE") == "1":
    from beartype.claw import beartype_this_package
    beartype_this_package()
```

- Full runtime type checking only in the dev environment; zero overhead in
  production/default environments.
- Never add `@beartype` decorators manually — the package-level claw covers
  everything.

## Annotations: PEP 526 everywhere, jaxtyping for arrays

Annotate variables at assignment, including intermediates — the verbosity is
deliberate: it keeps code self-documenting and gives beartype something to
validate at runtime.

Every array annotation carries BOTH dtype and shape:

```python
from jaxtyping import Float, UInt8, Int

rgb: UInt8[np.ndarray, "h w 3"] = load_image(path)
intrinsics: Float[np.ndarray, "3 3"] = calibration.K
indices: Int[np.ndarray, "n"] = np.argsort(scores)
```

Named/constrained axes are encouraged: `Float32[ndarray, "n_verts=778 3"]`.

Mirror the annotation in the variable NAME with shape/axis/colorspace
suffixes: `depth_hw`, `frames_rgb`, `bgr_hwc`, `points_xyz`. Redundant with
the type by design — it makes shape/format bugs visible at every call site.

## Type aliases: TypeAlias, never PEP 695

beartype does not support PEP 695 `type X = ...` statements (ruff's UP040 is
ignored for exactly this reason). Always:

```python
from typing import TypeAlias
ImageBGR: TypeAlias = UInt8[ndarray, "H W 3"]
DeviceChoice: TypeAlias = Literal["auto", "cuda", "cpu"]
```

## Imports & structure

- Absolute first-party imports (`from pkg.module import X`); relative imports
  are legacy, not the target style.
- pathlib over os.path.
- CLI entry points use tyro, not argparse.
- Config serialization is pyserde (`@serde`, `serde.json`) — never introduce
  pydantic.
- CLI tools and demos use plain `print()`; don't introduce logging frameworks
  into packages that don't already have one.

## Torch patterns

- Device selection: a `DeviceChoice` Literal alias + a
  `resolve_device(device: DeviceChoice = "auto") -> str` helper ("auto" →
  cuda if available else cpu; explicit "cuda" raises RuntimeError when
  unavailable). Pass the resolved device explicitly to
  `.to(device=..., dtype=...)` — never rely on implicit device inference.
- Axis manipulation via `einops.rearrange`/`repeat`, not manual
  `.reshape()`/`.permute()` chains.
- Float-typed defaults are written `0.0`, never `0` — beartype distinguishes
  int from float strictly.
- Never blanket `except Exception` around instrumented code without
  re-raising `BeartypeException` first.

## Dataclass documentation

Each field gets a docstring line directly beneath it (same for pyserde
`@serde` classes):

```python
@dataclass
class NerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config."""

    data: Path = Path()
    """Directory or explicit json file path specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: int | None = None
    """How much to downscale images; auto-chosen when None."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """Dataset split strategy; see each mode's field below."""
```

For everything that isn't a dataclass field, follow Google-style docstrings,
always including the full jaxtyping shape + dtype for array parameters.

## Return types: dataclass vs NamedTuple vs tuple

Decision checklist:

1. Named concept used beyond one function → `@dataclass(slots=True)`
   (add `frozen=True` when immutability is wanted). Field annotations +
   docstrings stay adjacent; beartype validates per-field.
2. Tuple semantics needed (hashable, positional unpacking) with a small,
   stable set of fields → `NamedTuple` with jaxtyping-annotated fields
   (declare `__slots__ = ()` to prevent attribute drift).
3. Plain tuple ONLY when unpacked immediately and never crossing a module
   boundary — and even then, annotate the receiving variables.
4. Consumers likely to grow (extra fields, defaults, methods) → start with a
   dataclass to avoid churn.

Do not unpack a call directly into untyped names — route through an annotated
intermediate so beartype actually checks the values:

```python
# no: verts, joints = mano_layer(so3, trans)
results: tuple[
    Float32[ndarray, "n_frames 778 3"], Float32[ndarray, "n_frames 21 3"]
] = mano_layer(so3, trans)
verts: Float32[ndarray, "n_frames 778 3"] = results[0]
joints: Float32[ndarray, "n_frames 21 3"] = results[1]
```

When a two-item tuple must travel further, define a `TypeAlias` (never a
PEP 695 `type` statement — see Type aliases above) or upgrade to a NamedTuple:

```python
ManoResults: TypeAlias = tuple[Float32[ndarray, "n 778 3"], Float32[ndarray, "n 21 3"]]
```

## New-package tooling baseline

ruff: `line-length = 150`, `select = ["E","F","UP","B","SIM","I"]`,
`ignore = ["E501","F722","F821","UP037","UP040"]` — F722/F821 suppress
jaxtyping forward-ref false positives; UP037/UP040 protect jaxtyping quotes
and the TypeAlias rule. Typechecking is pyrefly (workspace-level config).
