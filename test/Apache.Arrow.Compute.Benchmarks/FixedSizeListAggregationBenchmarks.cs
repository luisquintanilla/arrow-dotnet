// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements. See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using Apache.Arrow.Compute.Testing;
using BenchmarkDotNet.Attributes;

namespace Apache.Arrow.Compute.Benchmarks
{
    [MemoryDiagnoser]
    public class FixedSizeListAggregationBenchmarks
    {
        [Params(1024, 65536)]
        public int Rows { get; set; }

        [Params(2, 3, 5, 16)]
        public int Width { get; set; }

        [Params(false, true)]
        public bool WithNulls { get; set; }

        [Params(false, true)]
        public bool Floats { get; set; }

        [Params(false, true)]
        public bool Mean { get; set; }

        private FixedSizeListArray _input = null!;
        private DoubleArray[] _components = null!;

        [GlobalSetup]
        public void Setup()
        {
            _input = FixedSizeListAggregationReference.Create(Rows, Width, Floats, WithNulls);
            _components = FixedSizeListAggregationReference.Gather(_input);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            FixedSizeListAggregationReference.Dispose(_components);
            _input.Dispose();
        }

        [Benchmark(Baseline = true)]
        public void StridedReference()
        {
            using DoubleArray result = FixedSizeListAggregationReference.Strided(_input, Mean);
        }

        [Benchmark]
        public void GatherThenKernel()
        {
            using DoubleArray result = FixedSizeListAggregationReference.GatherThenKernel(_input, Mean);
        }

        [Benchmark]
        public void GatheredKernelOnly()
        {
            using DoubleArray result = FixedSizeListAggregationReference.GatheredKernel(_components, Mean);
        }

        [Benchmark]
        public void OnePassPrototype()
        {
            using DoubleArray result = Mean ? _input.MeanByComponent() : _input.SumByComponent();
        }

        [Benchmark]
        public int GatherOnly()
        {
            DoubleArray[] components = FixedSizeListAggregationReference.Gather(_input);
            int count = components.Length;
            FixedSizeListAggregationReference.Dispose(components);
            return count;
        }
    }
}
