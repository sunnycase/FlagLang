// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Globalization;
using Nncase;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestBfloat16
{
    [Fact]
    public void TestInfinity()
    {
        var positiveInfinity = (BFloat16)1F / (BFloat16)0F;
        Assert.Equal(BFloat16.Infinity, positiveInfinity);
    }

    [Fact]
    public void TestNegInfinity()
    {
        var negInfinity = (BFloat16)(-1F) / (BFloat16)0F;
        Assert.Equal(BFloat16.NegInfinity, negInfinity);
    }

    [Fact]
    public void TestEpsilon()
    {
        var epsilon = (BFloat16)0.0078125;
        Assert.Equal(BFloat16.Epsilon, epsilon);
    }

    [Fact]
    public void TestNan()
    {
        var nan = (BFloat16)(0F / 0F);
        Assert.Equal(BFloat16.NaN, nan);
    }

    [Fact]
    public void TestCompare()
    {
        var f = 1.234F;
        var a = (BFloat16)1.23F;
        var b = (BFloat16)1.23F;
        var c = (BFloat16)2.34F;

        Assert.True(a == b);
        Assert.True(a != c);
        Assert.True(a < c);
        Assert.True(a <= c);
        Assert.True(c > b);
        Assert.True(c >= b);

        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
        Assert.NotEqual(a, f);
        Assert.True(a.Equals((object)b));
    }

    [Fact]
    public void TestGetHashCode()
    {
        ushort a = 0x1234;
        var b = BFloat16.FromRaw(a);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
    }

    [Fact]
    public void TestToString()
    {
        var a = (BFloat16)1.23F;
        Assert.Equal(((float)a).ToString(), a.ToString());
    }

    [Fact]
    public void TestNumericInterfaceMembers()
    {
        Assert.Equal((BFloat16)2F, BFloat16.Abs((BFloat16)(-2F)));
        Assert.True(BFloat16.IsCanonical(BFloat16.NaN));
        Assert.False(BFloat16.IsComplexNumber((BFloat16)1F));
        Assert.True(BFloat16.IsEvenInteger((BFloat16)2F));
        Assert.True(BFloat16.IsFinite((BFloat16)1F));
        Assert.False(BFloat16.IsImaginaryNumber((BFloat16)1F));
        Assert.True(BFloat16.IsInteger((BFloat16)2F));
        Assert.True(BFloat16.IsNaN(BFloat16.NaN));
        Assert.True(BFloat16.IsNegative((BFloat16)(-1F)));
        Assert.True(BFloat16.IsNormal((BFloat16)1F));
        Assert.True(BFloat16.IsOddInteger((BFloat16)3F));
        Assert.True(BFloat16.IsPositive((BFloat16)1F));
        Assert.True(BFloat16.IsPositiveInfinity(BFloat16.Infinity));
        Assert.True(BFloat16.IsRealNumber((BFloat16)1F));
        Assert.True(BFloat16.IsSubnormal(BFloat16.FromRaw(1)));
        Assert.True(BFloat16.IsZero(BFloat16.Zero));
        Assert.Equal((BFloat16)(-3F), BFloat16.MaxMagnitude((BFloat16)(-3F), (BFloat16)2F));
        Assert.Equal((BFloat16)2F, BFloat16.MinMagnitude((BFloat16)(-3F), (BFloat16)2F));
    }

    [Fact]
    public void TestNumericParsingConversionAndFormatting()
    {
        Assert.Equal((BFloat16)2.5F, BFloat16.Parse("2.5", CultureInfo.InvariantCulture));
        Assert.Equal((BFloat16)2.5F, BFloat16.Parse("2.5".AsSpan(), CultureInfo.InvariantCulture));
        Assert.Equal((BFloat16)3.5F, BFloat16.Parse("3.5", NumberStyles.Float, CultureInfo.InvariantCulture));
        Assert.Equal((BFloat16)3.5F, BFloat16.Parse("3.5".AsSpan(), NumberStyles.Float, CultureInfo.InvariantCulture));

        Assert.True(BFloat16.TryParse("4.5", CultureInfo.InvariantCulture, out var parsedString));
        Assert.Equal((BFloat16)4.5F, parsedString);
        Assert.True(BFloat16.TryParse("5.5".AsSpan(), CultureInfo.InvariantCulture, out var parsedSpan));
        Assert.Equal((BFloat16)5.5F, parsedSpan);
        Assert.True(BFloat16.TryParse("6.5", NumberStyles.Float, CultureInfo.InvariantCulture, out var styledString));
        Assert.Equal((BFloat16)6.5F, styledString);
        Assert.True(BFloat16.TryParse("7.5".AsSpan(), NumberStyles.Float, CultureInfo.InvariantCulture, out var styledSpan));
        Assert.Equal((BFloat16)7.5F, styledSpan);

        Assert.True(BFloat16.TryConvertFromChecked<int>(8, out var fromInt));
        Assert.Equal((BFloat16)8F, fromInt);
        Assert.True(BFloat16.TryConvertFromSaturating<int>(9, out var saturatingInt));
        Assert.Equal((BFloat16)9F, saturatingInt);
        Assert.True(BFloat16.TryConvertFromTruncating<double>(10.75, out var truncatingDouble));
        Assert.Equal((BFloat16)10.75F, truncatingDouble);
        Assert.True(BFloat16.TryConvertToChecked<float>((BFloat16)11F, out var toFloat));
        Assert.Equal(11F, toFloat);
        Assert.True(BFloat16.TryConvertToSaturating<double>((BFloat16)12F, out var toDouble));
        Assert.Equal(12D, toDouble);
        Assert.True(BFloat16.TryConvertToTruncating<int>((BFloat16)13.75F, out var toInt));
        Assert.Equal(13, toInt);

        Span<char> destination = stackalloc char[16];
        Assert.True(((BFloat16)1.5F).TryFormat(destination, out var charsWritten, default, CultureInfo.InvariantCulture));
        Assert.Equal("1.5", destination[..charsWritten].ToString());
        Assert.Equal("1.50", ((BFloat16)1.5F).ToString("0.00", CultureInfo.InvariantCulture));
        Assert.Equal(0, ((BFloat16)1F).CompareTo((object)(BFloat16)1F));
        Assert.Throws<ArgumentException>(() => ((BFloat16)1F).CompareTo(1F));
    }
}
