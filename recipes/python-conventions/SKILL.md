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
- The dev environment checks everything, annotated locals included, and is
  allowed to be slower. Never pass `BeartypeConf(claw_is_pep526=False)` to
  speed a run up: long jobs (corpus conversions, registrations, full-length
  captures, training) run in the prod environment, which never activates beartype.
  If one hot local makes the dev gate measurably too slow, drop that local's
  annotation; do not invent aliases for speed.
- Why annotated locals cost time: every `x: Float32[ndarray, "3"] = ...` runs
  the subscription again, jaxtyping builds a new class each time, and beartype
  compiles and caches a new checker for it (~45 µs per line, and memory is
  never freed). This is dev-only cost; see the policy above.
- beartype caches transformed bytecode in `__pycache__/*.opt-beartype*.pyc`
  keyed by beartype version, not by conf. After changing a `BeartypeConf`,
  delete those files or the old checks stay in.
- No `jaxtyping.jaxtyped` and no `install_import_hook`. The hook replaces the
  claw for its modules, so local checks stop. The decorator runs in prod too
  and raises `jaxtyping.TypeCheckError`, not `BeartypeException`.
  `jaxtyped(typechecker=None)` under the claw turns off every check on that
  function, dtype included.

## Annotations: PEP 526 everywhere, jaxtyping for arrays

Annotate variables at assignment, including intermediates — the verbosity is
deliberate: it keeps code self-documenting and gives beartype something to
validate at runtime.

Every array annotation carries BOTH dtype and shape:

```python
from jaxtyping import Float64, Int64, UInt8

rgb: UInt8[np.ndarray, "h w 3"] = load_image(path)
intrinsics: Float64[np.ndarray, "3 3"] = calibration.K
order: Int64[np.ndarray, "n"] = np.argsort(scores)
```

Numpy arrays use a fixed width (`Float32`, `Float64`, `Int64`) whenever the
producer fixes it: `argsort` returns int64, and a float64 value under
`Float32` fails beartype. Generic `Float`/`Int` on a numpy array only for
genuine dtype polymorphism, marked on the line with
`# jaxtyping: generic-dtype`. Torch tensors may use generic `Float` where
autocast or half precision varies the dtype. Serialized fields always fix the
width.

### Shapes: what each spelling enforces

beartype checks each array **on its own**: dtype, rank, literal sizes, and a
name repeated inside one string (`"n n"`). It does **not** compare a name
across arguments or with the return value: `f(a: "n 3", b: "n 3")` accepts
n=2 with n=5. Axis names are documentation, so they must still be correct.

| Spelling | Meaning | Checked now |
|---|---|---|
| `"n 3"` | exactly one leading axis | rank and the `3` |
| `"*batch 3"` | any number of leading axes, zero included, shared by every array that says `*batch` | only the `3` |
| `"*#batch 4 4"` | broadcasts to `*batch` | only the `4 4` |
| `"... 3"` | anonymous leading axes, related to nothing | only the `3` |
| `"n_verts=778 3"` | `n_verts` is a label, `778` is checked | the `778` and the `3` |

Choose by reading the body, not the caller:

1. The body indexes an axis (`[:, i]`, `shape[0]`, `dim=1`, `axis=0`,
   `reshape(x.shape[0], ...)`) → name every axis up to it. A branch on
   `ndim in (3, 4)` → a union of fixed ranks.
2. The body flattens and restores any leading dims (`reshape(-1, k)` …
   `reshape(*lead, ...)`) and two or more arrays share them → `*batch`; an
   input that goes through `broadcast_to` → `*#batch`. One variadic per string.
3. One array whose leading axes nothing else refers to → `"..."`. Never put
   `"..."` in a union with a stricter shape: the union accepts anything.

```python
# no: "..." hides that X is rank 5; a rank-3 X silently gives an outer product
def proj(X: Float[Tensor, "... 4"], intrinsics: Float[Tensor, "... 4"]) -> Float[Tensor, "..."]: ...
# yes
def proj(X: Float[Tensor, "b e ps ps 4"], intrinsics: Float[Tensor, "b e 4"]) -> Float[Tensor, "b e ps ps 2"]: ...
```

Spelling: one concept keeps one spelling per package (`ps`, not `ps`/`p`/`P`),
and locals use the signature's names. No symbolic axes (`"n+1"`): without `jaxtyped` they
raise `AnnotationError` on every call. Unpacking into pre-declared locals
(`r: X; r, s = f()`) is never checked: annotate the function's return instead.

## Names carry meaning; types carry dtype and shape

The jaxtyping annotation is the only place dtype and shape live, and beartype
checks it in the dev environment (pyrefly sees only `ndarray`/`Tensor` until
shape checking is on, below). Names never
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

Use `TypeAlias`, never PEP 695 `type X = ...` (keep ruff's UP040 ignored).
beartype accepts both, but `type` does not parse on Python 3.10/3.11, which
some environments importing shared packages still run, and beartype's error
for a `type` alias shows only the alias name, not the expected shape.

```python
from typing import TypeAlias
ImageBGR: TypeAlias = UInt8[ndarray, "h w 3"]
DeviceChoice: TypeAlias = Literal["auto", "cuda", "cpu"]
```

Aliases are for names that repeat, not for speed. Define them once at module
level; nesting one at a use site (`Shaped[ImageBGR, "t"]`) builds a new class
every time it runs.

Strings with a fixed set of values are `Literal` aliases, never bare `str`;
reuse the alias for params, returns, dict keys, and fields. No `Any`/`object`
where a canonical type or a small `Protocol` fits.

## Imports & structure

- Absolute first-party imports (`from pkg.module import X`); relative imports
  are legacy, not the target style.
- pathlib over os.path.
- CLI entry points use tyro, not argparse.
- Serialization goes through pyserde (see **Serialization** below) — never
  introduce pydantic.
- CLI tools and demos use plain `print()`; don't introduce logging frameworks
  into packages that don't already have one.

## Serialization

pyserde is the door through which data enters Python. Anything that crosses a
boundary gets a `@serde` dataclass stating what the data is: a file, an HTTP
response, a dataset's own JSON / YAML / pickle, a catalog row, a model's output
dict. Fields are checked on the way in (strict mode *is* beartype); the rest of
the code holds typed objects, never dicts.

- **The Rerun catalog comes first.** Data registered on the catalog is read from
  the catalog (`CatalogClient`, the dataloader) and never re-parsed from the raw
  files it was converted from, nor copied into a document we own. pyserde has two
  places in that flow: ingest (raw third-party format → typed record → catalog)
  and what the catalog does not carry (gate files, IMU noise models, run reports,
  tool configs).
- **Records yes, streams no.** A record read whole goes through pyserde, arrays
  included (calibrations, hand models, keypoint rows, reports, whole-sequence
  pose tables). A stream you iterate does not (frames, depth maps, masks, point
  clouds, trajectories): it lives in npz/npy, Parquet/Arrow or Rerun, and
  pyserde carries only the metadata that names it. The test is the role of the
  data, not a size threshold.
- **Arrays carry jaxtyping with an explicit dtype**: `Float32[ndarray, "n 3"]`,
  never `Float[...]` or bare `ndarray` — generic `Float`/`Int` do not fix the
  width and a bare `ndarray` round-trips as float64. Cast explicitly at a
  conversion boundary when the source width differs. Use jaxtyping, not
  numpy's own typing alias, which breaks beartype at decoration time.
- **Formats for files we own:** JSON when a program writes it (`serde.json`;
  orjson is picked up automatically), TOML when a person edits it
  (`serde.toml`). YAML and pickle only when a third party hands us that format.
- **Strictness is per schema.** Files we own:
  `@serde(type_check=coerce, deny_unknown_fields=True)` above the frozen slots
  dataclass for hand-written TOML (`from_toml` does not widen `30` to `30.0`;
  `from_json` does), strict + `deny_unknown_fields=True` for machine-written
  JSON. Third-party formats read partially: unknown fields allowed; their schema
  is not ours to police.
- **Validation lives at the door.** Types come from the class. Cross-field rules
  go in `__post_init__` (pyserde runs it on load; its exception propagates
  unwrapped). Rules that need context live in one loader that wraps
  `SerdeError` *and* the parser's own error (`TOMLDecodeError`,
  `json.JSONDecodeError`) into a `ValueError` naming the source — pyserde's
  message names the field, not the path. Callers never re-validate. Never
  `except Exception` around a decode: `BeartypeException` propagates.
- **Properties do not serialise.** A report with computed columns gets a flat
  report dataclass at the write boundary. Our JSON outputs write an unscored
  number as `X | None` → `null`, never NaN; an input that legitimately carries
  non-finite cells keeps a custom field decoder.
- **Rust extensions own their formats** (serde derive on the Rust side). Python
  asks the extension for typed accessors instead of parsing `to_json()` output;
  the JSON methods stay for files.
- **Environment:** `tomli-w` and `orjson` are declared beside every `pyserde`
  declaration in `pixi.toml` (the conda `pyserde` ships no extras; without
  `tomli-w` even `from_toml` fails to import). Self-referential classes are
  decorated after the class body with `serde.serde(Node)`;
  `to_dict(reuse_instances=False)` when JSON-ready primitives are needed.
- **Not pyserde:** per-frame loops over large arrays (npz/Parquet/Rerun);
  formats it has no codec for — CSV (`csv.DictReader` → `from_dict(Row, ...)`),
  PGM, Arrow, protobuf, streaming JSONL; a measured hot path (msgspec, held in
  reserve, nothing needs it yet).
- **Rollout:** convert a hand-rolled `json.load`/`yaml.safe_load` + dict-indexing
  site when already editing that file, plus one deliberate pass per package with
  an owner; no big-bang. Raw-format access stays in contract tests and
  malformed-input fixtures.

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
