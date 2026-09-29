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

namespace Apache.Arrow.Compute.Testing
{
    internal static class FixedSizeListAggregationReference
    {
        internal static FixedSizeListArray Create(int rows, int width, bool floats, bool withNulls)
        {
            var builder = new FixedSizeListArray.Builder(floats ? (IArrowType)FloatType.Default : DoubleType.Default, width);
            builder.Reserve(rows);
            for (int row = 0; row < rows; row++)
            {
                if (withNulls && row % 11 == 0)
                {
                    builder.AppendNull();
                    continue;
                }
                builder.Append();
                for (int component = 0; component < width; component++)
                {
                    if (withNulls && (row + component) % 7 == 0)
                    {
                        builder.ValueBuilder.AppendNull();
                    }
                    else
                    {
                        double value = ((row * 17 + component * 13) % 101 - 50) / 16.0;
                        if (floats) ((FloatArray.Builder)builder.ValueBuilder).Append((float)value);
                        else ((DoubleArray.Builder)builder.ValueBuilder).Append(value);
                    }
                }
            }
            return builder.Build();
        }

        internal static DoubleArray Strided(FixedSizeListArray array, bool mean)
        {
            int width = ((FixedSizeListType)array.Data.DataType).ListSize;
            var result = new DoubleArray.Builder().Reserve(width);
            for (int component = 0; component < width; component++)
            {
                double sum = 0;
                int count = 0;
                for (int row = 0; row < array.Length; row++)
                {
                    double? value = Read(array, row, component, width);
                    if (value.HasValue)
                    {
                        sum += value.Value;
                        count++;
                    }
                }
                if (count == 0) result.AppendNull();
                else result.Append(mean ? sum / count : sum);
            }
            return result.Build();
        }

        internal static DoubleArray[] Gather(FixedSizeListArray array)
        {
            int width = ((FixedSizeListType)array.Data.DataType).ListSize;
            var result = new DoubleArray[width];
            for (int component = 0; component < width; component++)
            {
                var builder = new DoubleArray.Builder().Reserve(array.Length);
                for (int row = 0; row < array.Length; row++)
                {
                    double? value = Read(array, row, component, width);
                    if (value.HasValue) builder.Append(value.Value);
                    else builder.AppendNull();
                }
                result[component] = builder.Build();
            }
            return result;
        }

        internal static DoubleArray GatheredKernel(DoubleArray[] components, bool mean)
        {
            var result = new DoubleArray.Builder().Reserve(components.Length);
            foreach (DoubleArray component in components)
            {
                double? value = mean ? component.Mean() : component.Sum();
                if (value.HasValue) result.Append(value.Value);
                else result.AppendNull();
            }
            return result.Build();
        }

        internal static DoubleArray GatherThenKernel(FixedSizeListArray array, bool mean)
        {
            DoubleArray[] components = Gather(array);
            try
            {
                return GatheredKernel(components, mean);
            }
            finally
            {
                Dispose(components);
            }
        }

        internal static void Dispose(DoubleArray[] arrays)
        {
            foreach (DoubleArray array in arrays) array.Dispose();
        }

        private static double? Read(FixedSizeListArray array, int row, int component, int width)
        {
            if (array.IsNull(row)) return null;
            int index = checked((array.Offset + row) * width + component);
            // GetValue indexes the child's logical slice, independently of the parent offset.
            if (array.Values is DoubleArray doubles) return doubles.GetValue(index);
            if (array.Values is FloatArray floats) return floats.GetValue(index);
            throw new NotSupportedException("Reference supports only float and double children.");
        }
    }
}
