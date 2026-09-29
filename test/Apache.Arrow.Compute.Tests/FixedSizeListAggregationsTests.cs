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
        public void AllApproaches_MatchStudyMatrix(int rows, int width, bool floats, bool nulls)
        {
            using FixedSizeListArray input = FixedSizeListAggregationReference.Create(rows, width, floats, nulls);
            foreach (bool mean in new[] { false, true })
            {
                using DoubleArray expected = FixedSizeListAggregationReference.Strided(input, mean);
                using DoubleArray gathered = FixedSizeListAggregationReference.GatherThenKernel(input, mean);
                using DoubleArray actual = mean ? input.MeanByComponent() : input.SumByComponent();
                Assert.Equal(width, expected.Length);
                Assert.Equal(0, expected.NullCount);
                AssertSame(expected, gathered);
                AssertSame(expected, actual);
            }
        }

        [Fact]
        public void Reductions_SkipNullParentsAndUseComponentCounts()
        {
            using FixedSizeListArray input = Lists(3, new[] { true, false, true, true },
                1, null, null, 1000, 2000, 3000, 4, 8, null, null, 10, null);
            AssertReductions(input, new double?[] { 5, 18, null }, new double?[] { 2.5, 9, null });
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void Reductions_RespectIndependentParentAndChildSlices(bool floats)
        {
            Array storage = Numeric(floats, 999, 999, 100, 100, 100, 4, null, 6, 1000, 2000, 3000, 7, 8, 9, 999);
            using Array child = storage.SliceShared(2, 12);
            storage.Dispose();
            using var parent = new FixedSizeListArray(new FixedSizeListType(child.Data.DataType, 3), 4,
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
        public void Reductions_ValidZeroIsNotAnEmptyComponent()
        {
            using FixedSizeListArray input = Lists(3, new[] { true, true }, 1, 0, null, -1, 0, null);
            AssertReductions(input, new double?[] { 0, 0, null }, new double?[] { 0, 0, null });
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

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void Reductions_NonfiniteValuesFollowIeeeArithmetic(bool floats)
        {
            Array child = Numeric(floats,
                double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.PositiveInfinity, -0.0,
                1, 2, 3, double.NegativeInfinity, -0.0,
                999, double.NaN, double.NaN, 999, double.NaN);
            using var input = new FixedSizeListArray(new FixedSizeListType(child.Data.DataType, 5), 3,
                child, Bitmap(true, true, false), 1);
            AssertReductions(input,
                new double?[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 0 },
                new double?[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.NaN, 0 });
            using DoubleArray sum = input.SumByComponent();
            Assert.Equal(0L, BitConverter.DoubleToInt64Bits(sum.GetValue(4).Value));
        }

        [Fact]
        public void Reductions_DoubleOverflowRemainsInfinity()
        {
            using FixedSizeListArray input = Lists(1, new[] { true, true }, double.MaxValue, double.MaxValue);
            AssertReductions(input, new double?[] { double.PositiveInfinity }, new double?[] { double.PositiveInfinity });
        }

        [Fact]
        public void NullInput_IsRejected()
        {
            Assert.Throws<ArgumentNullException>(() => FixedSizeListAggregations.SumByComponent(null));
            Assert.Throws<ArgumentNullException>(() => FixedSizeListAggregations.MeanByComponent(null));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(-1)]
        public void InvalidListWidths_AreRejectedAtConstruction(int width)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new FixedSizeListType(DoubleType.Default, width));
        }

        [Fact]
        public void UnaddressableResultWidth_IsRejected()
        {
            using var input = new FixedSizeListArray(new FixedSizeListType(DoubleType.Default, int.MaxValue),
                0, Doubles(), ArrowBuffer.Empty);
            Assert.Throws<ArgumentException>(() => input.SumByComponent());
            Assert.Throws<ArgumentException>(() => input.MeanByComponent());
        }

        [Fact]
        public void UnsupportedChildren_AreRejected()
        {
            using Int32Array child = new Int32Array.Builder().Append(1).Append(2).Build();
            using var input = new FixedSizeListArray(new FixedSizeListType(Int32Type.Default, 2), 1, child, ArrowBuffer.Empty);
            Assert.Throws<NotSupportedException>(() => input.SumByComponent());
            Assert.Throws<NotSupportedException>(() => input.MeanByComponent());
        }

        [Fact]
        public void MismatchedChildType_IsRejected()
        {
            using var input = new FixedSizeListArray(new FixedSizeListType(FloatType.Default, 2), 1, Doubles(1, 2), ArrowBuffer.Empty);
            Assert.Throws<ArgumentException>(() => input.SumByComponent());
            Assert.Throws<ArgumentException>(() => input.MeanByComponent());
        }

        [Theory]
        [InlineData(2, 0, 1, 3)]
        [InlineData(1, 1, 1, 3)]
        [InlineData(1, -1, 0, 2)]
        [InlineData(-1, 0, 0, 2)]
        [InlineData(1, 0, 2, 2)]
        [InlineData(1, int.MaxValue, 0, 2)]
        public void MalformedParentShapes_AreRejected(int rows, int offset, int nullCount, int childLength)
        {
            using var input = new FixedSizeListArray(new FixedSizeListType(DoubleType.Default, 2), rows,
                Doubles(new double?[childLength]), Bitmap(true, false, true), nullCount, offset);
            Assert.Throws<ArgumentException>(() => input.SumByComponent());
            Assert.Throws<ArgumentException>(() => input.MeanByComponent());
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void ShortValidityBitmaps_AreRejected(bool childBitmap)
        {
            ArrowBuffer shortBitmap = new ArrowBuffer(new byte[1]);
            DoubleArray child = childBitmap
                ? new DoubleArray(new ArrowBuffer(new byte[9 * sizeof(double)]), shortBitmap, 9, 1, 0)
                : Doubles(new double?[9]);
            using var input = new FixedSizeListArray(new FixedSizeListType(DoubleType.Default, 1), 9, child,
                childBitmap ? ArrowBuffer.Empty : shortBitmap, childBitmap ? 0 : 1);
            Assert.Throws<ArgumentException>(() => input.SumByComponent());
            Assert.Throws<ArgumentException>(() => input.MeanByComponent());
        }

        [Fact]
        public void MissingValidityBitmap_IsRejected()
        {
            using var input = new FixedSizeListArray(new FixedSizeListType(DoubleType.Default, 2), 1,
                Doubles(1, 2), ArrowBuffer.Empty, 1);
            Assert.Throws<ArgumentException>(() => input.SumByComponent());
            Assert.Throws<ArgumentException>(() => input.MeanByComponent());
        }

        [Fact]
        public void ShortChildValueBuffer_IsRejected()
        {
            var child = new DoubleArray(new ArrowBuffer(new byte[sizeof(double)]), ArrowBuffer.Empty, 2, 0, 0);
            using var input = new FixedSizeListArray(new FixedSizeListType(DoubleType.Default, 2), 1, child, ArrowBuffer.Empty);
            Assert.Throws<ArgumentException>(() => input.SumByComponent());
            Assert.Throws<ArgumentException>(() => input.MeanByComponent());
        }

        [Fact]
        public void ResultOwnsBuffersIndependentlyOfInput()
        {
            FixedSizeListArray input = Lists(2, new[] { true, true }, 1, 2, 3, 4);
            using DoubleArray result = input.SumByComponent();
            Assert.Equal(1, ((DoubleArray)input.Values).GetValue(0));
            input.Dispose();
            Assert.Equal(4, result.GetValue(0));
            Assert.Equal(6, result.GetValue(1));
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

        private static Array Numeric(bool floats, params double?[] values)
        {
            if (!floats) return Doubles(values);
            var builder = new FloatArray.Builder();
            foreach (double? value in values)
            {
                if (value.HasValue) builder.Append((float)value.Value);
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
                using DoubleArray actual = mean ? input.MeanByComponent() : input.SumByComponent();
                AssertSame(expected, reference);
                AssertSame(expected, gathered);
                AssertSame(expected, actual);
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
