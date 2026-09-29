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

## Reproducing the study

Run commands only in an isolated experiment checkout. The actual evaluated SDK
here is 10.0.112, inherited from an ancestor `global.json`. Tests are SDK-style,
xUnit 2.9.3 on VSTest, with net8.0 and Windows net472 targets. Library targets
remain netstandard2.0, net8.0 and net462. BenchmarkDotNet is 0.15.8.

```powershell
dotnet test test\Apache.Arrow.Compute.Tests\Apache.Arrow.Compute.Tests.csproj -c Release -f net8.0 --no-restore --filter "FullyQualifiedName~FixedSizeListAggregationsTests"
dotnet run --project test\Apache.Arrow.Compute.Benchmarks\Apache.Arrow.Compute.Benchmarks.csproj -c Release --no-build --no-restore -- --filter "*FixedSizeListAggregationBenchmarks*" --job Short --exporters json csv --artifacts <measurement-directory>
```

Default data is deterministic synthetic numeric input, not inference output.
The default matrix is 1,024/65,536 rows, widths 2/3/5/16, both float/double, with
and without validity, and sum/mean. Larger sweeps are explicitly opt-in by
changing the benchmark parameter configuration in a separate measured study.
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
