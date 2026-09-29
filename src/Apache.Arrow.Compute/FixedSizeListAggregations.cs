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
using Apache.Arrow.Types;

namespace Apache.Arrow.Compute
{
    /// <summary>Experimental component reductions over fixed-size lists with float or double children.</summary>
    /// <remarks>
    /// Null parents are excluded entirely; a null child excludes only its component.
    /// Results have one nullable double per component. Floats are promoted before accumulation.
    /// Each component accumulates in row order with ordinary IEEE arithmetic, without
    /// rejecting NaN/infinities. These scalar kernels are identical on all supported frameworks.
    /// The caller owns the result; input buffers are neither modified nor retained.
    /// </remarks>
    public static class FixedSizeListAggregations
    {
        /// <summary>Sums each component. Empty or all-null components are null, not zero.</summary>
        public static DoubleArray SumByComponent(this FixedSizeListArray array) => Reduce(array, false);

        /// <summary>Means each component using its own valid count. Empty or all-null components are null.</summary>
        public static DoubleArray MeanByComponent(this FixedSizeListArray array) => Reduce(array, true);

        private static DoubleArray Reduce(FixedSizeListArray array, bool mean)
        {
            if (array is null) throw new ArgumentNullException(nameof(array));
            if (!(array.Data.DataType is FixedSizeListType type) ||
                type.ListSize <= 0 || type.ListSize > int.MaxValue / sizeof(double))
                throw new ArgumentException("The fixed-size-list width must fit a positive double result buffer.", nameof(array));
            if (array.Values is DoubleArray doubles && type.ValueDataType.TypeId == ArrowTypeId.Double)
                return Reduce<double, DoubleConverter>(array, doubles, type.ListSize, mean);
            if (array.Values is FloatArray floats && type.ValueDataType.TypeId == ArrowTypeId.Float)
                return Reduce<float, FloatConverter>(array, floats, type.ListSize, mean);
            if (type.ValueDataType.TypeId != array.Values.Data.DataType.TypeId)
                throw new ArgumentException("Declared and actual child types must match.", nameof(array));
            throw new NotSupportedException("Component reductions support only float and double fixed-size-list children.");
        }

        private static DoubleArray Reduce<T, TConverter>(
            FixedSizeListArray array, PrimitiveArray<T> child, int width, bool mean)
            where T : struct, IEquatable<T>
            where TConverter : struct, IConverter<T>
        {
            ValidateValidity(array);
            ValidateValidity(child);
            long end = ((long)array.Offset + array.Length) * width;
            if (end > child.Length)
                throw new ArgumentException("The child does not cover the parent slice.", nameof(array));
            if (((long)child.Offset + child.Length) * System.Runtime.InteropServices.Marshal.SizeOf<T>() > child.ValueBuffer.Length)
                throw new ArgumentException("The child value buffer does not cover its logical slice.", nameof(array));
            if (array.Data.Children == null || array.Data.Children.Length != 1 ||
                !ReferenceEquals(array.Data.Children[0], child.Data))
                throw new ArgumentException("Exactly one matching child is required.", nameof(array));

            var sums = new double[width];
            var counts = new int[width];
            ReadOnlySpan<T> values = child.Values;
            bool parentNoNulls = array.NullCount == 0;
            bool childNoNulls = child.NullCount == 0;
            TConverter converter = default;
            int start = checked(array.Offset * width);
            for (int row = 0; row < array.Length; row++)
            {
                int index = start + row * width;
                if (!parentNoNulls && array.IsNull(row)) continue;
                for (int component = 0; component < width; component++)
                {
                    if (!childNoNulls && child.IsNull(index + component)) continue;
                    sums[component] += converter.ToDouble(values[index + component]);
                    counts[component]++;
                }
            }

            var result = new DoubleArray.Builder().Reserve(width);
            for (int component = 0; component < width; component++)
            {
                if (counts[component] == 0) result.AppendNull();
                else result.Append(mean ? sums[component] / counts[component] : sums[component]);
            }
            return result.Build();
        }

        private static void ValidateValidity(Array array)
        {
            ArrayData data = array.Data;
            if (data.Length < 0 || data.Offset < 0 || data.NullCount < -1 || data.NullCount > data.Length)
                throw new ArgumentException("Invalid array length, offset or null count.", nameof(array));
            long requiredBytes = ((long)data.Offset + data.Length + 7) / 8;
            if ((!array.NullBitmapBuffer.IsEmpty && array.NullBitmapBuffer.Length < requiredBytes) ||
                (data.NullCount > 0 && array.NullBitmapBuffer.IsEmpty))
                throw new ArgumentException("The validity bitmap does not cover the logical slice.", nameof(array));
        }

        private interface IConverter<T>
        {
            double ToDouble(T value);
        }

        private readonly struct DoubleConverter : IConverter<double>
        {
            public double ToDouble(double value) => value;
        }

        private readonly struct FloatConverter : IConverter<float>
        {
            public double ToDouble(float value) => value;
        }
    }
}
