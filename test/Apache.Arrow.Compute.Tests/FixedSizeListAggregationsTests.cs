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

using System;
using System.Collections.Generic;
using Apache.Arrow.Compute.Testing;
using Apache.Arrow.Types;
using Xunit;

namespace Apache.Arrow.Compute.Tests
{
    public class FixedSizeListAggregationsTests
    {
        public static IEnumerable<object[]> Matrix()
        {
            foreach (int rows in new[] { 1024, 65536 })
            foreach (int width in new[] { 2, 3, 5, 16 })
            foreach (bool floats in new[] { false, true })
            foreach (bool nulls in new[] { false, true })
                yield return new object[] { rows, width, floats, nulls };
        }

        [Theory]
        [MemberData(nameof(Matrix))]
        public void ReferenceAndGather_MatchStudyMatrix(int rows, int width, bool floats, bool nulls)
        {
            using FixedSizeListArray input = FixedSizeListAggregationReference.Create(rows, width, floats, nulls);
            foreach (bool mean in new[] { false, true })
            {
                using DoubleArray expected = FixedSizeListAggregationReference.Strided(input, mean);
                using DoubleArray gathered = FixedSizeListAggregationReference.GatherThenKernel(input, mean);
                Assert.Equal(width, expected.Length);
                Assert.Equal(0, expected.NullCount);
                AssertSame(expected, gathered);
            }
        }

        [Fact]
        public void Reductions_SkipNullParentsAndUseComponentCounts()
        {
            using FixedSizeListArray input = Lists(3, new[] { true, false, true, true },
                1, null, null, 1000, 2000, 3000, 4, 8, null, null, 10, null);
            AssertReductions(input, new double?[] { 5, 18, null }, new double?[] { 2.5, 9, null });
        }

        [Fact]
        public void Reductions_RespectIndependentParentAndChildSlices()
        {
            DoubleArray storage = Doubles(999, 999, 100, 100, 100, 4, null, 6, 1000, 2000, 3000, 7, 8, 9, 999);
            using DoubleArray child = (DoubleArray)storage.SliceShared(2, 12);
            storage.Dispose();
            using var parent = new FixedSizeListArray(new FixedSizeListType(DoubleType.Default, 3), 4,
                child, Bitmap(true, true, false, true), 1);
            using var input = (FixedSizeListArray)parent.SliceShared(1, 3);
            Assert.Equal(1, input.Offset);
            Assert.Equal(2, input.Values.Offset);
            AssertReductions(input, new double?[] { 11, 8, 15 }, new double?[] { 5.5, 8, 7.5 });
        }

        [Theory]
        [InlineData(0)]
        [InlineData(3)]
        public void Reductions_EmptyOrAllNullParentsReturnNull(int rows)
        {
            using FixedSizeListArray input = Lists(2, new bool[rows], new double?[rows * 2]);
            AssertReductions(input, new double?[] { null, null }, new double?[] { null, null });
        }

        [Fact]
        public void Reductions_PromoteFloatBeforeAccumulation()
        {
            var builder = new FixedSizeListArray.Builder(FloatType.Default, 2);
            foreach (float value in new[] { 16777216f, 1f, -16777216f })
            {
                builder.Append();
                ((FloatArray.Builder)builder.ValueBuilder).Append(value).Append(float.MaxValue);
            }
            using FixedSizeListArray input = builder.Build();
            AssertReductions(input, new double?[] { 1, 3 * (double)float.MaxValue },
                new double?[] { 1.0 / 3, (double)float.MaxValue });
        }

        [Fact]
        public void Reductions_NonfiniteValuesFollowIeeeArithmetic()
        {
            using FixedSizeListArray input = Lists(5, new[] { true, true, false },
                double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, -0.0,
                1, 2, 3, double.NegativeInfinity, -0.0,
                999, double.NaN, double.NaN, 999, double.NaN);
            AssertReductions(input,
                new double?[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 0 },
                new double?[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 0 });
        }

        [Fact]
        public void Reductions_DoubleOverflowRemainsInfinity()
        {
            using FixedSizeListArray input = Lists(1, new[] { true, true }, double.MaxValue, double.MaxValue);
            AssertReductions(input, new double?[] { double.PositiveInfinity }, new double?[] { double.PositiveInfinity });
        }

        internal static FixedSizeListArray Lists(int width, bool[] parents, params double?[] values)
        {
            int nullCount = 0;
            foreach (bool valid in parents) if (!valid) nullCount++;
            return new FixedSizeListArray(new FixedSizeListType(DoubleType.Default, width),
                parents.Length, Doubles(values), Bitmap(parents), nullCount);
        }

        internal static DoubleArray Doubles(params double?[] values)
        {
            var builder = new DoubleArray.Builder();
            foreach (double? value in values)
            {
                if (value.HasValue) builder.Append(value.Value);
                else builder.AppendNull();
            }
            return builder.Build();
        }

        internal static ArrowBuffer Bitmap(params bool[] values)
        {
            var builder = new ArrowBuffer.BitmapBuilder();
            foreach (bool value in values) builder.Append(value);
            return builder.Build();
        }

        private static void AssertReductions(FixedSizeListArray input, double?[] sums, double?[] means)
        {
            foreach (bool mean in new[] { false, true })
            {
                using DoubleArray expected = Doubles(mean ? means : sums);
                using DoubleArray reference = FixedSizeListAggregationReference.Strided(input, mean);
                using DoubleArray gathered = FixedSizeListAggregationReference.GatherThenKernel(input, mean);
                AssertSame(expected, reference);
                AssertSame(expected, gathered);
            }
        }

        private static void AssertSame(DoubleArray expected, DoubleArray actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            Assert.Equal(expected.NullCount, actual.NullCount);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected.GetValue(i), actual.GetValue(i));
        }
    }
}
