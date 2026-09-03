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

## Names carry meaning; types carry dtype and shape

The jaxtyping annotation is the only place dtype and shape live, and it is
checked (beartype at function boundaries, pyrefly statically). Names never
repeat it: `cam_T_world`, not `cam_T_world_v44`; `points_xyz`, not
`points_xyz_n3`; no `_f64` / `_np` / `_t` dtype tags. Names keep what the type
cannot say: units (`_px`, `_m`, `_deg`), frame direction (`cam_T_world`,
`dst_R_src`), layout conventions (`_xy`/`_uv`, `_wh`/`_hw`, `_rgb`/`_bgr`).
One vs. many is singular/plural or a role word, never a suffix.

## Static shape checking (pyrefly)

pyrefly checks jaxtyping shapes once the `shape_extensions` package resolves:
pin `pyrefly-torch-stubs` and `pyrefly-numpy-stubs` (lockstep with the pyrefly
version) in the dev feature. Prove it is on with a probe that mismatches a
rank; it must fail. The numpy stubs (pyrefly 1.3 line) still lack `einsum`
and batched `@`: adopt per package as coverage allows, baseline meanwhile.

## Type aliases: TypeAlias, never PEP 695

beartype does not support PEP 695 `type X = ...` statements (ruff's UP040 is
ignored for exactly this reason). Always:

```python
from typing import TypeAlias
ImageBGR: TypeAlias = UInt8[ndarray, "H W 3"]
DeviceChoice: TypeAlias = Literal["auto", "cuda", "cpu"]
```

Strings with a fixed set of values are `Literal` aliases, never bare `str`;
reuse the alias for params, returns, dict keys, and fields. No `Any`/`object`
where a canonical type or a small `Protocol` fits.

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
  `resolve_device(device: DeviceChoice = "auto") -> Literal["cuda", "cpu"]` helper ("auto" →
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
