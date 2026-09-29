# Fork-only fixed-size-list reductions

This study belongs to `luisquintanilla/arrow-dotnet`, on a new dependent branch
targeting `feat-compute-aggregation-kernels`. It does not change the branch
feeding `apache/arrow-dotnet#379` and is not an upstream API proposal.

## Reference and gather baseline

A fixed-size-list stores its child values row-major. Reducing the entire child
array is not a component reduction. The study returns one nullable double per
component, in component order:

- Skip null parent rows entirely, even when their child storage contains numbers.
- Skip null children only for their own component. Means use that component's
  count; empty/all-null components are null, not zero.
- Index within the child's logical slice at `(parent.Offset + row) * width +
  component`. The child adds its own offset. Parent slices retain the full child.
- Promote float observations to double **before** accumulating. Gathering uses
  `DoubleArray` even for float input so the compared algorithms share semantics.
- Sum in row order from positive zero. NaN propagates, opposing infinities produce
  NaN, finite overflow produces infinity, and a sum of negative zeros is positive
  zero. There is no domain-specific nonfinite validation or normalization.

The test-only `FixedSizeListAggregationReference` is shared by the test and
benchmark projects. The strided reference loops component-first. The gather
comparison builds one contiguous double array per component, then calls the
existing primitive `Sum`/`Mean` kernels. Their SIMD reduction order can differ
for general floating-point data; dyadic synthetic fixtures avoid confounding
order differences in the correctness matrix.

The first coherent reference commit precedes production kernel changes.
Reference correctness: 39 passing net8.0 cases, including both input types,
1,024/65,536 rows, widths 2/3/5/16, null-free/validity cases, independent slices,
promotion, null components and IEEE behavior. A 128-case Release BenchmarkDotNet
Dry/in-process pilot exercised reference and gather first. That one-iteration
pilot is **not** steady-state performance evidence.

## One-pass prototype

`FixedSizeListAggregations.SumByComponent` and `MeanByComponent` accept only
float/double fixed-size-list children and return caller-owned `DoubleArray`
results. A generic scalar loop visits each row once, maintaining double sums and
independent counts. It uses no generic-math/runtime-only features, so old targets
share the same implementation. Inputs are not modified; results do not borrow
their buffers. Existing primitive kernels are unchanged.

Structural validation rejects type mismatches, unsupported children, negative
lengths/offsets, out-of-range null counts, short/missing validity/value buffers,
missing/mismatched children and children that cannot cover the parent slice.
Widths must be positive and fit an addressable double result buffer. The
`FixedSizeListType` constructor already disallows zero/negative widths.
Arrow null-count metadata is trusted, not recomputed to audit every bitmap bit.
Trailing child storage is permitted, as required for parent slicing.

The benchmark `OnePassPrototype` includes its sums/counts arrays and result
allocation/disposal. A null-free case avoids bitmap tests; validity cases retain
the same per-component counts and parent exclusion semantics.

## Reproducing the study

Run commands only in an isolated experiment checkout. The actual evaluated SDK
here is 10.0.112, inherited from an ancestor `global.json`. Tests are SDK-style,
xUnit 2.9.3 on VSTest, with net8.0 and Windows net472 targets. Library targets
remain netstandard2.0, net8.0 and net462. BenchmarkDotNet is 0.15.8.

```powershell
dotnet test test\Apache.Arrow.Compute.Tests\Apache.Arrow.Compute.Tests.csproj -c Release -f net8.0 --no-restore --filter "FullyQualifiedName~FixedSizeListAggregationsTests"
dotnet build src\Apache.Arrow.Compute\Apache.Arrow.Compute.csproj -c Release --no-restore
dotnet test test\Apache.Arrow.Compute.Tests\Apache.Arrow.Compute.Tests.csproj -c Release --no-restore
dotnet build test\Apache.Arrow.Compute.Benchmarks\Apache.Arrow.Compute.Benchmarks.csproj -c Release --no-restore
dotnet run --project test\Apache.Arrow.Compute.Benchmarks\Apache.Arrow.Compute.Benchmarks.csproj -c Release --no-build --no-restore -- --filter "*FixedSizeListAggregationBenchmarks*" --job Short --launchCount 1 --warmupCount 2 --iterationCount 5 --iterationTime 100 --exporters json csv --buildTimeout 600 --artifacts <measurement-directory>
```

Default data is deterministic synthetic numeric input, not inference output.
The default matrix is 1,024/65,536 rows, widths 2/3/5/16, both float/double, with
and without validity, and sum/mean. Set
`$env:ARROW_COMPONENT_LARGE_SWEEP = "1"` to add 1,048,576 rows in an **opt-in**
larger study, then unset it with `$env:ARROW_COMPONENT_LARGE_SWEEP = $null`.
Other flag values fail explicitly. The larger sweep was not measured here.
`GatherOnly` includes gathering, allocation and disposal. `GatherThenKernel`
includes gathering and reduction. `GatheredKernelOnly` reuses arrays prepared
outside the measurement; it does **not** represent end-to-end reduction latency.
Every reduction arm includes allocating and disposing an equivalent result.
MemoryDiagnoser reports managed allocations, not all native Arrow buffer bytes.

## Immutable baseline packages

Baseline source is the clean, unchanged commit
`6aafc634d65cf46ff7fe1a665c4c66fd6002b989`. Core, Compute and Scalars all use
`23.0.0-exp.decisions.1.g6aafc634d65c`. Release builds passed all library TFMs;
the existing compute suite passed 16/16 cases on net8.0 and 16/16 on net472.

The shared experiment artifact root contains `feeds\arrow-baseline\receipt.json`
and three nupkgs. The receipt records source, SDK/runtime, clean-tree status,
package hashes/sizes, nuspec dependency groups, build dependency closure and
raw evidence. Packing uses command-line version/repository overrides, not
source changes. The recorded packaging tool transforms only nuspec project
dependency ranges to exact `[version]`; other ZIP entries are unchanged.
Packages are unsigned NuGet artifacts; existing assembly strong naming remains.

Consumer restores must explicitly opt into a separate local configuration,
verify receipt hashes, use isolated caches and exact locks, and source-map the
three experimental IDs locally, other IDs to the approved public source.
Neither repository-root nor global NuGet configuration is changed. These
packages are not published publicly and their versions/bytes are never replaced.

## Measured comparison

The sustained sweep measured commit
`34d33f8219eb98184855ea6631bdb008066a3d34`: 320/320 out-of-process benchmarks,
64 parameter cases times five arms. The final opt-in row source retains the
same default matrix. Machine: Intel Core i9-11950H, 8 physical/16 logical cores,
Windows 11 10.0.26200.9448, x64 .NET 8.0.31/RyuJIT, SDK 10.0.112. Release,
BenchmarkDotNet 0.15.8, one launch, two warmups and five requested 100 ms
iterations per benchmark; execution 9m44s, total 10m42s.

The coordinator reserved the heavy-CPU window and peers held real inference
and SDCA training. Lightweight peer builds/tests/transport could still occur;
this is not a dedicated performance laboratory. No heavy overlap was reported.
Short-run sample variation, upper-outlier removal and the raw confidence
intervals are retained. Some independent gather-only and gather-plus-kernel
point means are inconsistent in magnitude: **do not subtract them** to
estimate kernel time, or interpret near ties as robust wins.

Variance is substantial: retained sample counts are 4 or 5; 66/320 benchmarks
have coefficient of variation above 10%, 28 above 25%, and the maximum is
112.5% (pre-gathered mean, 1024 rows, width 16, double with nulls).
`prototype-benchmark-variance.json` records these diagnostics against the same
raw report hash. The point-mean rankings are **exploratory, not robust
promotion evidence**. A longer dedicated run would be needed before making
strong performance claims; no additional sustained run competed with the
subsequent model/learning window.

Ratio below means **prototype mean time / comparison mean time**, over the
64 cases, not a pooled operation-time or speedup estimate.

| Comparison | Minimum ratio | Median ratio | Maximum ratio | Prototype slower (point means) |
|---|---:|---:|---:|---:|
| Strided reference | 0.0525 | 0.2282 | 0.6638 | 0/64 |
| Gather then kernel | 0.0132 | 0.0864 | 0.4047 | 0/64 |
| Pre-gathered kernel only | 0.1380 | 2.1825 | 12.1892 | **58/64** |

Selected **double input** point means in microseconds; `Mean` here denotes the
operation, not the statistic. Null scenarios include both null parents and
component-specific child nulls.

| Rows x width | Nulls | Operation | Strided | Gather + kernel | Gather only | Pre-gathered kernel | Prototype |
|---|---|---|---:|---:|---:|---:|---:|
| 1024 x 3 | No | Sum | 30.669 | 115.198 | 50.534 | 0.947 | 5.387 |
| 1024 x 3 | No | Mean | 21.589 | 146.159 | 65.662 | 0.870 | 5.730 |
| 1024 x 3 | Yes | Sum | 99.145 | 142.138 | 107.758 | 20.941 | 33.335 |
| 1024 x 3 | Yes | Mean | 88.408 | 159.526 | 193.528 | 21.104 | 35.656 |
| 65536 x 16 | No | Sum | 23470.088 | 61447.180 | 49631.175 | 2113.275 | 2082.615 |
| 65536 x 16 | No | Mean | 19091.324 | 51319.840 | 50094.640 | 404.665 | 2405.376 |
| 65536 x 16 | Yes | Sum | 41686.920 | 92952.825 | 73845.925 | 10665.663 | 12232.704 |
| 65536 x 16 | Yes | Mean | 49144.300 | 87199.620 | 75127.560 | 8651.700 | 8765.751 |

Managed allocations: width 3 prototype 608 B versus reference/pre-gathered
520 B; at 1024 rows, gather-plus-kernel about 27 KB. Width 16 prototype
912 B versus reference/pre-gathered 672 B; at 65536 rows gathering costs
about 8.53 MB. This excludes native Arrow buffer storage. The prototype is
**not** allocation-free and does not beat already contiguous reductions
universally. Repeated reductions over reused gathered arrays may justify
materialization; end-to-end one-shot gathering loses in this synthetic study.
These are single-machine observations, not inference improvements or claims
about all Arrow types/data. API review and upstream promotion remain out of scope.

The shared artifact root contains `measurements\arrow\prototype-benchmarks`
with raw JSON/CSV/HTML and logs, `prototype-benchmark-summary.json`, and
`prototype-normalized.csv` with explicit operation/statistic column names.
BenchmarkDotNet's raw CSV has two `Mean` headings (operation parameter and
timing metric); JSON is the authoritative input to the normalization tool.
Raw full-JSON SHA-256:
`81c3ac4e30b1505aca2136e61492fd03501c9df66773c44f9b8c7c5cd264b21f`.

## Compatible prototype package

`feeds\arrow-prototype\receipt.json` is a separate READY receipt for **Compute
only**, version `23.0.0-exp.decisions.2.g34d33f8219eb`, from clean kernel
commit `34d33f8219eb98184855ea6631bdb008066a3d34`.
Package SHA-256
`c002456f618cb18ae7cf76f16450d30ef130abefd7890f9d62586274afa6bbbd`,
size 66,732 bytes. Core and Scalars source is unchanged; their baseline
packages are neither replaced nor repacked.

All prototype Compute nuspec groups exact-pin baseline Core
`[23.0.0-exp.decisions.1.g6aafc634d65c]`, which in turn exact-pins baseline
Scalars. The original pack metadata is transformed as recorded; compiled ZIP
entries are unchanged. This deliberately supports the adapter's exact baseline
Core dependency, rather than forcing a conflicting experimental Core version.

A packages-only consumer (no project/source references) restored with explicit
opt-in unsigned configuration, exact source maps and a fresh isolated cache,
then restored in locked mode. Its net8.0 and Windows net472 executables both
passed null/count/ABI checks and reported the expected source-identifying Core
and Compute assembly versions. Cached nupkg hashes, sizes and source locations
were checked again. The receipt includes the baseline receipt hash, complete
resolved consumer dependency closure, lock, configuration hash and logs.
Source-map Compute to the prototype feed, Core/Scalars to the baseline feed,
and remaining IDs to the approved public source. Selecting this prototype is
explicit; the adapter need not be repacked or silently upgraded.

Consumer reproduction uses the project/lock and configuration saved under
`measurements\arrow\prototype-consumer-evidence` and
`prototype-consumer.NuGet.Config`. Copy the small consumer project into an
ignored subdirectory of an isolated worktree; pass the artifact config/cache
paths explicitly. Verify package receipt hashes before restoring:

```powershell
dotnet restore <consumer-project> --configfile <prototype-config> --packages <fresh-isolated-cache> --locked-mode
dotnet run --project <consumer-project> -c Release -f net8.0 --no-restore
dotnet build <consumer-project> -c Release -f net472 --no-restore
& <consumer-net472-executable>
```

## Requirement evidence

Final compute regression suite passes 75/75 on both net8.0 and Windows net472
(59 focused cases plus 16 preserved primitive cases). The packaged kernel
commit's receipt records its preceding 74/74 runs on both targets; the final
commit adds a valid-zero/null distinction test and optional benchmark rows,
not production kernel changes.

| Requirement | Exact evidence |
|---|---|
| "Return nullable DoubleArray with one slot per component" | `AllApproaches_MatchStudyMatrix`, `Reductions_SkipNullParentsAndUseComponentCounts` |
| "float accumulation promotes to double" | `Reductions_PromoteFloatBeforeAccumulation` (cancellation and float-range overflow) |
| "Skip null parents entirely" | `Reductions_SkipNullParentsAndUseComponentCounts` (non-null numbers stored under null parent) |
| "null children only skip that component; use its own valid count for means" | `Reductions_SkipNullParentsAndUseComponentCounts` |
| "empty/all-null component is null, never zero" | `Reductions_EmptyOrAllNullParentsReturnNull`, `Reductions_ValidZeroIsNotAnEmptyComponent` |
| "Correct parent/child slice offsets" | `Reductions_RespectIndependentParentAndChildSlices` (both types, both offsets, mixed validity) |
| "Explicit IEEE/nonfinite reference behavior" | `Reductions_NonfiniteValuesFollowIeeeArithmetic`, `Reductions_DoubleOverflowRemainsInfinity` |
| "Reject unsupported types/list widths/malformed shapes explicitly" | `UnsupportedChildren_AreRejected`, `MismatchedChildType_IsRejected`, `InvalidListWidths_AreRejectedAtConstruction`, `UnaddressableResultWidth_IsRejected`, `MalformedParentShapes_AreRejected`, `ShortValidityBitmaps_AreRejected`, `MissingValidityBitmap_IsRejected`, `ShortChildValueBuffer_IsRejected`, `NullInput_IsRejected` |
| "rows 1024 and 65536, widths 2,3,5,16" | `AllApproaches_MatchStudyMatrix`; all 320 recorded benchmark cases |
| "Separate allocations/gather cost from kernel-only time" | Five benchmark arms, normalized allocation/time columns, raw BDN evidence |
| "Preserve existing primitive kernel behavior and old target frameworks" | Existing `AggregationsTests` 16/16 on both targets; Release all-TFM library build; package-only net472 execution |

Assertions were reviewed against the implementation, not just test counts:
exact values, component length/null count, rejected exception types, signed
zero and independent result ownership (`ResultOwnsBuffersIndependentlyOfInput`)
are checked. No corpus or model is used by these tests or benchmarks.
