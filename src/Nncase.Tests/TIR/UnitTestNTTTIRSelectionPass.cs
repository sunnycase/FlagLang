// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Buffers;
using Nncase.IR.Distributed;
using Nncase.IR.Shapes;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Tiling;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestNTTTIRSelectionPass : TestClassBase
{
    public UnitTestNTTTIRSelectionPass()
    {
        CompileOptions.TargetOptions = new NTTTargetOptions();
    }

    [Fact]
    public async Task DependLowersByPreservingVisitedDependenciesAndReturningValue()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4)));
        var dependency = IR.F.Math.Unary(UnaryOp.Neg, input);
        var value = IR.F.Math.Unary(UnaryOp.Abs, input);
        var body = IR.F.Tensors.Depend(dependency, value);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, input));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        var calls = ExprCollector.Collect(lowered.Body).OfType<Call>().ToArray();

        Assert.DoesNotContain(calls, call => call.Target is Nncase.IR.Tensors.Depend);
        Assert.Equal(2, calls.Count(call => call.Target is Nncase.TIR.NTT.Unary));
        Assert.True(CompilerServices.InferenceType(lowered), CompilerServices.Print(lowered));
    }

    [Fact]
    public async Task BroadcastSelectsNttExpandKernel()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, Shape.Scalar));
        var body = IR.F.Tensors.Broadcast(input, new RankedShape(4));
        var function = new Function("main", new IRBlock(body, input));
        Assert.True(CompilerServices.InferenceType(function));

        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        var fields = lowered.Body.Fields.ToArray();
        var expand = Assert.IsType<Call>(fields[0]);

        Assert.IsType<Nncase.TIR.NTT.Expand>(expand.Target);
        Assert.IsType<Return>(fields[^1]);
    }

    [Fact]
    public async Task SequentialRegisterDirectAffineLowersWithoutAddressableIntermediates()
    {
        const int blockSize = 256;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var right = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default);
        var sum = IR.F.Math.Binary(BinaryOp.Add, left, right);
        var scatter = Nncase.IR.F.Affine.Scatter(sum, dest, relation, symbols);
        var body = new Sequential(new Expr[] { left, right, sum, scatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var fields = lowered.Body.Fields.ToArray();
        var allExprs = ExprCollector.Collect(lowered.Body).ToArray();
        var calls = allExprs.OfType<Call>().ToArray();

        Assert.Contains(fields, expr => expr is Nncase.TIR.For { Mode: LoopMode.Serial });
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineGather);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineScatter);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.VectorizedBinary);
        Assert.DoesNotContain(allExprs, expr => expr is Nncase.TIR.Buffer);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.Load);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.Store);
        var ret = Assert.IsType<Return>(fields[^1]);
        Assert.Empty(ret.Values.ToArray());
    }

    [Fact]
    public async Task RegisterDirectAffineLowersAllIndependentScattersInBlockOrder()
    {
        const int blockSize = 256;
        var lhs0 = new Var("lhs0", TensorType.Pointer(DataTypes.Float32));
        var rhs0 = new Var("rhs0", TensorType.Pointer(DataTypes.Float32));
        var dest0 = new Var("dest0", TensorType.Pointer(DataTypes.Float32));
        var lhs1 = new Var("lhs1", TensorType.Pointer(DataTypes.Float32));
        var rhs1 = new Var("rhs1", TensorType.Pointer(DataTypes.Float32));
        var dest1 = new Var("dest1", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left0 = Nncase.IR.F.Affine.Gather(lhs0, relation, symbols, shape, None.Default);
        var right0 = Nncase.IR.F.Affine.Gather(rhs0, relation, symbols, shape, None.Default);
        var scatter0 = Nncase.IR.F.Affine.Scatter(left0 + right0, dest0, relation, symbols);
        var left1 = Nncase.IR.F.Affine.Gather(lhs1, relation, symbols, shape, None.Default);
        var right1 = Nncase.IR.F.Affine.Gather(rhs1, relation, symbols, shape, None.Default);
        var scatter1 = Nncase.IR.F.Affine.Scatter(left1 + right1, dest1, relation, symbols);
        var body = new Sequential(new Expr[] { left0, right0, scatter0, left1, right1, scatter1 });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, lhs0, rhs0, dest0, lhs1, rhs1, dest1));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var fields = lowered.Body.Fields.ToArray();
        var calls = ExprCollector.Collect(lowered.Body).OfType<Call>().ToArray();

        Assert.Equal(2, fields.OfType<Nncase.TIR.For>().Count());
        Assert.Equal(2, calls.Count(call => call.Target is Nncase.TIR.Store));
        Assert.IsType<Return>(fields[^1]);
    }

    [Fact]
    public async Task RegisterDirectAffineRejectsUnrelatedSideEffectField()
    {
        const int blockSize = 256;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var right = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default);
        var scatter = Nncase.IR.F.Affine.Scatter(left + right, dest, relation, symbols);
        var unrelatedStore = T.Store(dest, Dimension.Zero, Tensor.FromScalar(0f));
        var body = new Sequential(new Expr[] { unrelatedStore, left, right, scatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        Assert.Contains("cannot preserve unrelated field", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RegisterDirectAffineRejectsMaskedGatherDefaultValue()
    {
        const int blockSize = 256;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var defaultValue = Const.FromTensor(Tensor.Zeros(DataTypes.Float32, new long[] { blockSize }));
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, defaultValue);
        var scatter = Nncase.IR.F.Affine.Scatter(left, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        Assert.Contains("default is None", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RegisterDirectAffineRejectsProducerConsumerStorageLayoutMismatch()
    {
        const int blockSize = 128;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var right = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default);
        var sum = left + right;
        var scatter = Nncase.IR.F.Affine.Scatter(sum, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        TileDecisionMetadata.Set(left, CreateRegisterDecision("tile_0", "Affine.Gather", shape));
        TileDecisionMetadata.Set(right, CreateRegisterDecision("tile_1", "Affine.Gather", shape));
        TileDecisionMetadata.Set(sum, CreateRegisterDecision("tile_2", "Binary", shape, CreateReverseRegisterStorageLayout(shape)));
        TileDecisionMetadata.Set(scatter, CreateRegisterDecision("tile_3", "Affine.Scatter", shape));

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        Assert.Contains("producer/consumer storage layouts", ex.Message, StringComparison.Ordinal);
        Assert.Contains("Binary", ex.Message, StringComparison.Ordinal);
        Assert.Contains("Affine.Scatter", ex.Message, StringComparison.Ordinal);
        Assert.Contains("ReverseLocal", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ReshapeBitcastAcceptsEquivalentExplicitLayouts()
    {
        var shape = new RankedShape(128);
        var inputType = CreateExplicitSbpDistributedType(shape);
        var outputType = CreateExplicitSbpDistributedType(shape);

        var lowered = await RunReshapeSelectionAsync(CompileOptions, inputType, outputType);
        var calls = ExprCollector.Collect(lowered.Body).OfType<Call>().ToArray();

        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.Reshape);
    }

    [Fact]
    public async Task ReshapeBitcastRejectsExplicitDistributionLayoutMismatch()
    {
        var shape = new RankedShape(128);
        var inputType = CreateExplicitSbpDistributedType(shape);
        var outputType = CreateExplicitSbpDistributedType(shape, distributionKind: "DifferentExplicitDistribution");

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => RunReshapeSelectionAsync(CompileOptions, inputType, outputType));

        Assert.Contains("GenerateReshape", ex.Message, StringComparison.Ordinal);
        Assert.Contains("distribution layouts", ex.Message, StringComparison.Ordinal);
        Assert.Contains("DifferentExplicitDistribution", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ReshapeBitcastRejectsExplicitStorageLayoutMismatch()
    {
        var shape = new RankedShape(128);
        var inputType = CreateExplicitSbpDistributedType(shape);
        var outputType = CreateExplicitSbpDistributedType(shape, storageLayoutFactory: localShape => CreateReverseRegisterStorageLayout(localShape));

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => RunReshapeSelectionAsync(CompileOptions, inputType, outputType));

        Assert.Contains("GenerateReshape", ex.Message, StringComparison.Ordinal);
        Assert.Contains("storage layouts", ex.Message, StringComparison.Ordinal);
        Assert.Contains("ReverseLocal", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task BitcastAcceptsSameSizeExplicitLayouts()
    {
        var shape = new RankedShape(128);
        var inputType = CreateExplicitSbpDistributedType(shape, storageLayoutFactory: localShape => CreateReverseRegisterStorageLayout(localShape));
        var inferredInput = new Var("inferred_input", inputType);
        var inferredBitcast = Nncase.IR.F.Tensors.Bitcast(inferredInput, DataTypes.UInt32);
        var inferredFunction = new Function("inferred", CUDATarget.Kind, new IRBlock(inferredBitcast, inferredInput));
        Assert.True(CompilerServices.InferenceType(inferredFunction), CompilerServices.Print(inferredFunction));
        var inferredType = Assert.IsType<DistributedType>(inferredBitcast.CheckedType);
        Assert.True(inferredType.HasExplicitLayout);
        Assert.Equal(DataTypes.UInt32, inferredType.TensorType.DType);
        Assert.Equal("ReverseLocal", inferredType.StorageLayout.Kind);

        var lowered = await RunBitcastSelectionAsync(
            CompileOptions,
            inputType,
            inputType with { TensorType = new TensorType(DataTypes.UInt32, shape) });
        var calls = ExprCollector.Collect(lowered.Body).OfType<Call>().ToArray();
        var ret = Assert.IsType<Return>(lowered.Body.Fields[^1]);
        var output = Assert.IsType<TIR.Buffer>(ret.Values.ToArray().Single());

        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.Cast);
        Assert.Equal(DataTypes.UInt32, output.ElemType);
        Assert.Equal("ReverseLocal", Assert.IsType<DistributedType>(output.Type).StorageLayout.Kind);
    }

    [Fact]
    public async Task BitcastRejectsElementSizeChangingExplicitLayout()
    {
        var inputShape = new RankedShape(128);
        var outputShape = new RankedShape(512);
        var inputType = CreateExplicitSbpDistributedType(inputShape, storageLayoutFactory: localShape => CreateReverseRegisterStorageLayout(localShape));
        var outputType = inputType with { TensorType = new TensorType(DataTypes.UInt8, outputShape) };
        var inferredInput = new Var("inferred_input", inputType);
        var inferredBitcast = Nncase.IR.F.Tensors.Bitcast(inferredInput, DataTypes.UInt8);
        var inferredFunction = new Function("inferred", CUDATarget.Kind, new IRBlock(inferredBitcast, inferredInput));

        Assert.False(CompilerServices.InferenceType(inferredFunction));
        var invalidType = Assert.IsType<InvalidType>(inferredBitcast.CheckedType);
        Assert.Contains("IR.Tensors.Bitcast type inference", invalidType.Reason, StringComparison.Ordinal);
        Assert.Contains("element sizes differ", invalidType.Reason, StringComparison.Ordinal);
        Assert.Contains("ReverseLocal", invalidType.Reason, StringComparison.Ordinal);

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => RunBitcastSelectionAsync(CompileOptions, inputType, outputType));

        Assert.Contains("GenerateBitcast", ex.Message, StringComparison.Ordinal);
        Assert.Contains("element sizes differ", ex.Message, StringComparison.Ordinal);
        Assert.Contains("InputDType", ex.Message, StringComparison.Ordinal);
        Assert.Contains("OutputDType", ex.Message, StringComparison.Ordinal);
        Assert.Contains("ReverseLocal", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task BitcastRejectsElementSizeChangingExplicitStorageOnlyLayoutWithBitcastDiagnostic()
    {
        var inputShape = new RankedShape(128);
        var outputShape = new RankedShape(512);
        var inputType = CreateExplicitStorageOnlySbpDistributedType(
            inputShape,
            DataTypes.Float32,
            localShape => CreateReverseRegisterStorageLayout(localShape));
        var outputType = CreateExplicitStorageOnlySbpDistributedType(
            outputShape,
            DataTypes.UInt8,
            localShape => CreateReverseRegisterStorageLayout(localShape));
        var inferredInput = new Var("inferred_input", inputType);
        var inferredBitcast = Nncase.IR.F.Tensors.Bitcast(inferredInput, DataTypes.UInt8);
        var inferredFunction = new Function("inferred", CUDATarget.Kind, new IRBlock(inferredBitcast, inferredInput));

        Assert.False(CompilerServices.InferenceType(inferredFunction));
        var invalidType = Assert.IsType<InvalidType>(inferredBitcast.CheckedType);
        AssertBitcastDiagnostic(invalidType.Reason, "IR.Tensors.Bitcast");

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => RunBitcastSelectionAsync(CompileOptions, inputType, outputType));

        AssertBitcastDiagnostic(ex.Message, "GenerateBitcast");
    }

    [Fact]
    public async Task BoxingD2DRejectsExplicitDistributionLayoutMismatch()
    {
        var shape = new RankedShape(128);
        var inputType = CreateExplicitSbpDistributedType(shape);
        var outputType = CreateExplicitSbpDistributedType(shape, distributionKind: "DifferentExplicitDistribution");
        var inferredInput = new Var("inferred_input", inputType);
        var inferredBoxing = Nncase.IR.F.Distributed.Boxing(inferredInput, outputType);
        var inferredFunction = new Function("inferred", CUDATarget.Kind, new IRBlock(inferredBoxing, inferredInput));

        Assert.False(CompilerServices.InferenceType(inferredFunction));
        var invalidType = Assert.IsType<InvalidType>(inferredBoxing.CheckedType);
        AssertD2DTransferDiagnostic(invalidType.Reason, "IR.Distributed.Boxing", "DifferentExplicitDistribution", "explicit distribution layouts");

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => RunD2DTransferSelectionAsync(CompileOptions, inputType, outputType, force: false));

        AssertD2DTransferDiagnostic(ex.Message, "GenerateReshard", "DifferentExplicitDistribution", "explicit distribution layouts");
    }

    [Fact]
    public async Task ForceBoxingD2DRejectsExplicitStorageLayoutMismatch()
    {
        var shape = new RankedShape(128);
        var inputType = CreateExplicitSbpDistributedType(shape);
        var outputType = CreateExplicitSbpDistributedType(shape, storageLayoutFactory: localShape => CreateReverseRegisterStorageLayout(localShape));
        var inferredInput = new Var("inferred_input", inputType);
        var inferredBoxing = Nncase.IR.F.Distributed.ForceBoxing(inferredInput, outputType);
        var inferredFunction = new Function("inferred", CUDATarget.Kind, new IRBlock(inferredBoxing, inferredInput));

        Assert.False(CompilerServices.InferenceType(inferredFunction));
        var invalidType = Assert.IsType<InvalidType>(inferredBoxing.CheckedType);
        AssertD2DTransferDiagnostic(invalidType.Reason, "IR.Distributed.ForceBoxing", "ReverseLocal", "explicit storage layouts");

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => RunD2DTransferSelectionAsync(CompileOptions, inputType, outputType, force: true));

        AssertD2DTransferDiagnostic(ex.Message, "ForceBoxing memcopy", "ReverseLocal", "explicit storage layouts");
    }

    [Fact]
    public async Task BoxingD2DWrapsInvalidInputExplicitStorageLayoutDiagnostic()
    {
        var shape = new RankedShape(128);
        var inputType = CreateExplicitSbpDistributedType(shape, storageLayoutFactory: localShape => CreateUnsupportedNamedPrimitiveStorageLayout(localShape));
        var outputType = CreateExplicitSbpDistributedType(shape);
        var inferredInput = new Var("inferred_input", inputType);
        var inferredBoxing = Nncase.IR.F.Distributed.Boxing(inferredInput, outputType);
        var inferredFunction = new Function("inferred", CUDATarget.Kind, new IRBlock(inferredBoxing, inferredInput));

        Assert.False(CompilerServices.InferenceType(inferredFunction));
        var invalidType = Assert.IsType<InvalidType>(inferredBoxing.CheckedType);
        AssertD2DLayoutVerificationDiagnostic(
            invalidType.Reason,
            "IR.Distributed.Boxing",
            "Swizzle",
            "input layout verification failed",
            "unsupported named primitive xor_swizzle");

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => RunD2DTransferSelectionAsync(CompileOptions, inputType, outputType, force: false));

        AssertD2DLayoutVerificationDiagnostic(
            ex.Message,
            "GenerateReshard",
            "Swizzle",
            "input layout verification failed",
            "unsupported named primitive xor_swizzle");
    }

    [Fact]
    public async Task ForceBoxingD2DWrapsInvalidOutputExplicitDistributionLayoutDiagnostic()
    {
        var shape = new RankedShape(128);
        var inputType = CreateExplicitSbpDistributedType(shape);
        var outputType = CreateExplicitSbpDistributedType(shape, distributionLayoutFactory: CreateBrokenInverseDistributionLayout);
        var inferredInput = new Var("inferred_input", inputType);
        var inferredBoxing = Nncase.IR.F.Distributed.ForceBoxing(inferredInput, outputType);
        var inferredFunction = new Function("inferred", CUDATarget.Kind, new IRBlock(inferredBoxing, inferredInput));

        Assert.False(CompilerServices.InferenceType(inferredFunction));
        var invalidType = Assert.IsType<InvalidType>(inferredBoxing.CheckedType);
        AssertD2DLayoutVerificationDiagnostic(
            invalidType.Reason,
            "IR.Distributed.ForceBoxing",
            "BrokenInverseDistribution",
            "output layout verification failed",
            "must declare inverse OwnerLocalToGlobal, but got BrokenInverse");

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => RunD2DTransferSelectionAsync(CompileOptions, inputType, outputType, force: true));

        AssertD2DLayoutVerificationDiagnostic(
            ex.Message,
            "ForceBoxing memcopy",
            "BrokenInverseDistribution",
            "output layout verification failed",
            "must declare inverse OwnerLocalToGlobal, but got BrokenInverse");
    }

    [Fact]
    public async Task RegisterDirectAffineLowersUnaryCastWhereWithoutAddressableIntermediates()
    {
        const int blockSize = 256;
        var input = new Var("input", TensorType.Pointer(DataTypes.Float32));
        var cond = new Var("cond", TensorType.Pointer(DataTypes.Boolean));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var whereDest = new Var("where_dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var inputTile = Nncase.IR.F.Affine.Gather(input, relation, symbols, shape, None.Default);
        var castSource = IR.F.Tensors.Cast(IR.F.Math.Unary(UnaryOp.Neg, inputTile), DataTypes.Float32);
        var condTile = Nncase.IR.F.Affine.Gather(cond, relation, symbols, shape, None.Default);
        var rhsTile = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default);
        var whereScatter = Nncase.IR.F.Affine.Scatter(IR.F.Tensors.Where(condTile, castSource, rhsTile), whereDest, relation, symbols);
        var body = new Sequential(new Expr[] { inputTile, castSource, condTile, rhsTile, whereScatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, input, cond, rhs, whereDest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));

        AssertRegisterDirectAffineLowering(lowered);
    }

    [Fact]
    public async Task DistributedRegisterDirectAffineUsesTritonBlockedStridedThreadOwnership()
    {
        const int blockSize = 1024;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var domain = Nncase.IR.F.Affine.Dim(0);
        domain.Metadata.Range = new(0, blockSize - 1);
        var programId = new ProgramIdDim(0);
        var programIdSymbol = Nncase.IR.F.Affine.Symbol(0);
        var relation = new AffineRelation(
            [domain],
            [programIdSymbol],
            [(((AffineExpr)programIdSymbol) * (AffineConstant)(long)blockSize) + domain]);
        var symbols = new RankedShape(programId);
        var shape = new RankedShape(blockSize);
        var ndsbp = new IRArray<SBP>(new SBP[] { SBP.S(0) });
        var placement = new Placement([128], "t");
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default, ndsbp, placement);
        var right = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default, ndsbp, placement);
        var sum = IR.F.Math.Binary(BinaryOp.Add, left, right);
        var scatter = Nncase.IR.F.Affine.Scatter(sum, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var printed = CompilerServices.Print(lowered, PrinterFlags.Script);

        AssertRegisterDirectAffineLowering(lowered);
        Assert.Contains("T.Serial(out var d0, (0, 8, 1)", printed, StringComparison.Ordinal);
        Assert.Contains("128 * d0", printed, StringComparison.Ordinal);
        Assert.Contains("tid", printed, StringComparison.Ordinal);
        Assert.DoesNotContain("8 * tid", printed, StringComparison.Ordinal);
    }

    [Fact]
    public async Task BlockLocalSmemSharedGatherLowersWithSyncAndSharedBuffer()
    {
        const int blockSize = 128;
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var firstDest = new Var("first_dest", TensorType.Pointer(DataTypes.Float32));
        var secondDest = new Var("second_dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var tile = Nncase.IR.F.Affine.Gather(source, relation, symbols, shape, None.Default);
        var firstScatter = Nncase.IR.F.Affine.Scatter(tile, firstDest, relation, symbols);
        var secondScatter = Nncase.IR.F.Affine.Scatter(tile, secondDest, relation, symbols);
        var body = new Sequential(new Expr[] { tile, firstScatter, secondScatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, source, firstDest, secondDest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.True(TileDecisionMetadata.TryGet(tile, out var decision));
        Assert.Equal(BufferScope.BlockLocal, decision.Storage.Scope);
        Assert.Equal(PhysicalMemorySpace.SMem, decision.Storage.PhysicalLocation);

        var selected = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var fields = selected.Body.Fields.ToArray();
        Assert.IsType<Nncase.TIR.NTT.AffineGather>(Assert.IsType<Call>(fields[0]).Target);
        Assert.IsType<Nncase.TIR.NTT.SynchronizeThreads>(Assert.IsType<Call>(fields[1]).Target);
        Assert.IsType<Nncase.TIR.For>(fields[2]);
        Assert.IsType<Return>(fields[3]);
        Assert.Equal(2, ExprCollector.Collect(fields[2]).OfType<Call>().Count(call => call.Target is Nncase.TIR.Store));
        var smemBuffer = Assert.Single(ExprCollector.Collect(selected.Body).OfType<Nncase.TIR.Buffer>());
        Assert.Equal(BufferScope.BlockLocal, smemBuffer.Storage.Scope);
        Assert.Equal(PhysicalMemorySpace.SMem, smemBuffer.Storage.PhysicalLocation);

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(selected, new()));
        var calls = ExprCollector.Collect(lowered.Body).OfType<Call>().ToArray();
        Assert.Contains(calls, call => call.Target is Nncase.TIR.NTT.SynchronizeThreads);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineGather);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineScatter);
        Assert.Contains(calls, call => call.Target is BufferStore);
        Assert.Contains(calls, call => call.Target is BufferLoad);
        Assert.True(CompilerServices.InferenceType(lowered), CompilerServices.Print(lowered, PrinterFlags.Script));

        var module = new IRModule(lowered);
        var passManager = CompileSession.CreatePassManager("smem-bufferize");
        passManager.Add<BufferizePass>();
        module = await passManager.RunAsync(module);
        var scheduled = Assert.IsType<PrimFunction>(module.Entry);
        Assert.True(scheduled.SchedResult.IsScheduled);
        Assert.True(scheduled.SchedResult.BlockLocalDataPoolSize >= (ulong)decision.ByteSize);
    }

    [Fact]
    public async Task BlockLocalSmemFusedConsumerUsesSharedStorageLayoutIndex()
    {
        const int blockSize = 128;
        const int threadsPerCta = 32;
        const int elementsPerThread = blockSize / threadsPerCta;
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var firstDest = new Var("first_dest", TensorType.Pointer(DataTypes.Float32));
        var secondDest = new Var("second_dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var ndsbp = new IRArray<SBP>(new SBP[] { SBP.S(0) });
        var placement = new Placement([threadsPerCta], "t");
        var tile = Nncase.IR.F.Affine.Gather(source, relation, symbols, shape, None.Default, ndsbp, placement);
        var firstScatter = Nncase.IR.F.Affine.Scatter(tile, firstDest, relation, symbols);
        var secondScatter = Nncase.IR.F.Affine.Scatter(tile, secondDest, relation, symbols);
        var body = new Sequential(new Expr[] { tile, firstScatter, secondScatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, source, firstDest, secondDest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.True(TileDecisionMetadata.TryGet(tile, out var decision));
        Assert.Equal("TritonBlocked", decision.DistributionLayout?.Kind);
        Assert.Equal("SharedBlock", decision.StorageLayout.Kind);

        var selected = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(selected, new()));
        var fields = lowered.Body.Fields.ToArray();
        var gatherLoop = Assert.IsType<Nncase.TIR.For>(fields[0]);
        var consumerLoop = Assert.IsType<Nncase.TIR.For>(fields[2]);
        var smemStore = Assert.Single(ExprCollector.Collect(gatherLoop.Body).OfType<Call>().Where(call => call.Target is BufferStore));
        var smemLoad = Assert.Single(ExprCollector.Collect(consumerLoop.Body).OfType<Call>().Where(call => call.Target is BufferLoad));
        var storeIndex = GetSingleBufferIndex(smemStore, BufferStore.Indices);
        var loadIndex = GetSingleBufferIndex(smemLoad, BufferLoad.Indices);

        for (long programId = 0; programId <= 1; programId++)
        {
            var storeValues = EvaluateIndexValues(storeIndex, gatherLoop.LoopVar, elementsPerThread, programId, threadId: 7);
            var loadValues = EvaluateIndexValues(loadIndex, consumerLoop.LoopVar, elementsPerThread, programId, threadId: 7);
            Assert.Equal(storeValues, loadValues);
        }

        Assert.Equal(new long[] { 135, 167, 199, 231 }, EvaluateIndexValues(loadIndex, consumerLoop.LoopVar, elementsPerThread, programId: 1, threadId: 7));
        Assert.True(CompilerServices.InferenceType(lowered), CompilerServices.Print(lowered, PrinterFlags.Script));
    }

    [Fact]
    public async Task BlockLocalSmemUnsupportedExplicitLayoutFailsFast()
    {
        const int blockSize = 128;
        const int threadsPerCta = 32;
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var firstDest = new Var("first_dest", TensorType.Pointer(DataTypes.Float32));
        var secondDest = new Var("second_dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var ndsbp = new IRArray<SBP>(new SBP[] { SBP.S(0) });
        var placement = new Placement([threadsPerCta], "t");
        var tile = Nncase.IR.F.Affine.Gather(source, relation, symbols, shape, None.Default, ndsbp, placement);
        var firstScatter = Nncase.IR.F.Affine.Scatter(tile, firstDest, relation, symbols);
        var secondScatter = Nncase.IR.F.Affine.Scatter(tile, secondDest, relation, symbols);
        var body = new Sequential(new Expr[] { tile, firstScatter, secondScatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, source, firstDest, secondDest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var unsupported = CreateUnsupportedTritonBlockedLayout(shape);
        TileDecisionMetadata.Set(tile, CreateBlockLocalSmemDecision(
            "tile_0",
            shape,
            new TileLifetime(0, 2),
            unsupported,
            StorageLayout.SharedBlock(shape, unsupported)));

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        Assert.Contains("UnsupportedLayout", ex.Message, StringComparison.Ordinal);
        Assert.Contains("smem_tile_0", ex.Message, StringComparison.Ordinal);
        Assert.Contains("Add a DistributionLayout evaluator", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task BlockLocalSmemMultipleSharedGathersPreserveNonOverlappingSchedule()
    {
        const int blockSize = 128;
        var firstSource = new Var("first_source", TensorType.Pointer(DataTypes.Float32));
        var secondSource = new Var("second_source", TensorType.Pointer(DataTypes.Float32));
        var firstDest0 = new Var("first_dest_0", TensorType.Pointer(DataTypes.Float32));
        var firstDest1 = new Var("first_dest_1", TensorType.Pointer(DataTypes.Float32));
        var secondDest0 = new Var("second_dest_0", TensorType.Pointer(DataTypes.Float32));
        var secondDest1 = new Var("second_dest_1", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var firstTile = Nncase.IR.F.Affine.Gather(firstSource, relation, symbols, shape, None.Default);
        var firstScatter0 = Nncase.IR.F.Affine.Scatter(firstTile, firstDest0, relation, symbols);
        var firstScatter1 = Nncase.IR.F.Affine.Scatter(firstTile, firstDest1, relation, symbols);
        var secondTile = Nncase.IR.F.Affine.Gather(secondSource, relation, symbols, shape, None.Default);
        var secondScatter0 = Nncase.IR.F.Affine.Scatter(secondTile, secondDest0, relation, symbols);
        var secondScatter1 = Nncase.IR.F.Affine.Scatter(secondTile, secondDest1, relation, symbols);
        var body = new Sequential(new Expr[] { firstTile, firstScatter0, firstScatter1, secondTile, secondScatter0, secondScatter1 });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, firstSource, secondSource, firstDest0, firstDest1, secondDest0, secondDest1));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.True(TileDecisionMetadata.TryGet(firstTile, out var firstDecision));
        Assert.True(TileDecisionMetadata.TryGet(secondTile, out var secondDecision));
        Assert.Equal("smem-slot0@0+512[0,2]", firstDecision.Telemetry.AllocationSlot);
        Assert.Equal("smem-slot0@0+512[3,5]", secondDecision.Telemetry.AllocationSlot);

        var selected = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var fields = selected.Body.Fields.ToArray();
        Assert.IsType<Nncase.TIR.NTT.AffineGather>(Assert.IsType<Call>(fields[0]).Target);
        Assert.IsType<Nncase.TIR.NTT.SynchronizeThreads>(Assert.IsType<Call>(fields[1]).Target);
        Assert.IsType<Nncase.TIR.For>(fields[2]);
        Assert.IsType<Nncase.TIR.NTT.SynchronizeThreads>(Assert.IsType<Call>(fields[3]).Target);
        Assert.IsType<Nncase.TIR.NTT.AffineGather>(Assert.IsType<Call>(fields[4]).Target);
        Assert.IsType<Nncase.TIR.NTT.SynchronizeThreads>(Assert.IsType<Call>(fields[5]).Target);
        Assert.IsType<Nncase.TIR.For>(fields[6]);
        Assert.IsType<Return>(fields[7]);
        Assert.Equal(2, ExprCollector.Collect(fields[2]).OfType<Call>().Count(call => call.Target is Nncase.TIR.Store));
        Assert.Equal(2, ExprCollector.Collect(fields[6]).OfType<Call>().Count(call => call.Target is Nncase.TIR.Store));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(selected, new()));
        var module = new IRModule(lowered);
        var passManager = CompileSession.CreatePassManager("smem-bufferize");
        passManager.Add<BufferizePass>();
        module = await passManager.RunAsync(module);
        var scheduled = Assert.IsType<PrimFunction>(module.Entry);
        Assert.True(scheduled.SchedResult.IsScheduled);
        Assert.Equal((ulong)firstDecision.ByteSize, scheduled.SchedResult.BlockLocalDataPoolSize);
    }

    [Fact]
    public async Task BlockLocalSmemGatherRequiresPerGatherReuse()
    {
        const int blockSize = 128;
        var firstSource = new Var("first_source", TensorType.Pointer(DataTypes.Float32));
        var secondSource = new Var("second_source", TensorType.Pointer(DataTypes.Float32));
        var firstDest = new Var("first_dest", TensorType.Pointer(DataTypes.Float32));
        var secondDest = new Var("second_dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var firstTile = Nncase.IR.F.Affine.Gather(firstSource, relation, symbols, shape, None.Default);
        var firstScatter = Nncase.IR.F.Affine.Scatter(firstTile, firstDest, relation, symbols);
        var secondTile = Nncase.IR.F.Affine.Gather(secondSource, relation, symbols, shape, None.Default);
        var secondScatter = Nncase.IR.F.Affine.Scatter(secondTile, secondDest, relation, symbols);
        var body = new Sequential(new Expr[] { firstTile, firstScatter, secondTile, secondScatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, firstSource, secondSource, firstDest, secondDest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));
        TileDecisionMetadata.Set(firstTile, CreateBlockLocalSmemDecision("tile_0", shape, new TileLifetime(0, 1)));
        TileDecisionMetadata.Set(secondTile, CreateBlockLocalSmemDecision("tile_1", shape, new TileLifetime(2, 3)));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        Assert.Contains("every Affine.Gather", ex.Message, StringComparison.Ordinal);
        Assert.Contains("at least two direct Affine.Scatter consumers", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DirectAffineSelectionRequiresTileDecision()
    {
        const int blockSize = 256;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(left, lhs));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        Assert.Contains("requires a tile decision", ex.Message, StringComparison.Ordinal);
    }

    private static TileDecision CreateBlockLocalSmemDecision(
        string id,
        Shape shape,
        TileLifetime lifetime,
        DistributionLayout? distributionLayout = null,
        StorageLayout? storageLayout = null) =>
        new(
            id,
            "Affine.Gather",
            shape,
            distributionLayout,
            storageLayout ?? StorageLayout.Identity(shape),
            new BufferStorage(BufferUsage.Temp, BufferScope.BlockLocal, PhysicalMemorySpace.SMem),
            lifetime,
            shape[0].FixedValue * DataTypes.Float32.SizeInBytes,
            new TileCapacity(shape[0].FixedValue * DataTypes.Float32.SizeInBytes, 49152, "test"),
            new TileTelemetry(1, shape[0].FixedValue * DataTypes.Float32.SizeInBytes, 0, shape[0].FixedValue * DataTypes.Float32.SizeInBytes, $"{id}@0+512"),
            true,
            "test smem decision");

    private static DistributionLayout CreateUnsupportedTritonBlockedLayout(Shape shape)
    {
        var baseLayout = DistributionLayout.TritonBlocked(
            shape,
            new TritonBlockedLayout(
                SizePerThread: checked((int)(shape[0].FixedValue / 32)),
                ThreadsPerWarp: 32,
                WarpsPerCTA: 1,
                Order: [0],
                CTAsPerCGA: [1],
                CTASplitNum: [1],
                CTAOrder: [0],
                ThreadElementOrder: TritonThreadElementOrder.Strided));
        return baseLayout with { Kind = "UnsupportedLayout" };
    }

    private static TileDecision CreateRegisterDecision(string id, string opKind, Shape shape, StorageLayout? storageLayout = null) =>
        new(
            id,
            opKind,
            shape,
            null,
            storageLayout ?? StorageLayout.Identity(shape),
            new BufferStorage(BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.Register),
            new TileLifetime(0, 0),
            shape[0].FixedValue * DataTypes.Float32.SizeInBytes,
            new TileCapacity(shape[0].FixedValue * DataTypes.Float32.SizeInBytes, 4096, "test"),
            new TileTelemetry(1, shape[0].FixedValue * DataTypes.Float32.SizeInBytes, shape[0].FixedValue * DataTypes.Float32.SizeInBytes, 0, $"{id}@register"),
            false,
            "test register decision");

    private static StorageLayout CreateReverseRegisterStorageLayout(Shape shape)
    {
        var extent = shape[0].FixedValue;
        return new StorageLayout(
            "ReverseLocal",
            shape,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                ["l0"],
                [new IndexMapBinding("p0", IndexExpr.Add(IndexExpr.Const(extent - 1), IndexExpr.Mul(IndexExpr.Const(-1), IndexExpr.Var("l0"))))],
                [$"0<=l0<{extent}"],
                [$"0<=p0<{extent}"],
                Inverse: "LogicalToPhysical"));
    }

    private static StorageLayout CreateUnsupportedNamedPrimitiveStorageLayout(Shape shape)
    {
        var extent = shape[0].FixedValue;
        return new StorageLayout(
            "Swizzle",
            shape,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                ["l0"],
                [new IndexMapBinding("p0", new IndexNamedPrimitive("xor_swizzle", [IndexExpr.Var("l0")]))],
                [$"0<=l0<{extent}"],
                [$"0<=p0<{extent}"],
                Inverse: "LogicalToPhysical"));
    }

    private static DistributionLayout CreateBrokenInverseDistributionLayout(DistributionLayout layout) =>
        layout with
        {
            Kind = "BrokenInverseDistribution",
            GlobalToOwnerLocal = layout.GlobalToOwnerLocal with { Inverse = "BrokenInverse" },
        };

    private static DistributedType CreateExplicitSbpDistributedType(
        Shape shape,
        DataType? dataType = null,
        string? distributionKind = null,
        Func<Shape, StorageLayout>? storageLayoutFactory = null,
        Func<DistributionLayout, DistributionLayout>? distributionLayoutFactory = null)
    {
        var tensorType = new TensorType(dataType ?? DataTypes.Float32, shape);
        var axisPolicies = new IRArray<SBP>(new SBP[] { SBP.S(0) });
        var placement = new Placement([4], "t");
        var distributionLayout = DistributionLayout.FromAxisPolicies(tensorType, axisPolicies, placement);
        if (distributionKind is not null)
        {
            distributionLayout = distributionLayout with { Kind = distributionKind };
        }

        if (distributionLayoutFactory is not null)
        {
            distributionLayout = distributionLayoutFactory(distributionLayout);
        }

        return new DistributedType(
            tensorType,
            axisPolicies,
            placement,
            ExplicitDistributionLayout: distributionLayout,
            ExplicitStorageLayout: storageLayoutFactory?.Invoke(distributionLayout.LocalShape) ?? StorageLayout.Identity(distributionLayout.LocalShape));
    }

    private static DistributedType CreateExplicitStorageOnlySbpDistributedType(
        Shape shape,
        DataType dataType,
        Func<Shape, StorageLayout> storageLayoutFactory)
    {
        var tensorType = new TensorType(dataType, shape);
        var axisPolicies = new IRArray<SBP>(new SBP[] { SBP.S(0) });
        var placement = new Placement([4], "t");
        var distributionLayout = DistributionLayout.FromAxisPolicies(tensorType, axisPolicies, placement);

        return new DistributedType(
            tensorType,
            axisPolicies,
            placement,
            ExplicitStorageLayout: storageLayoutFactory(distributionLayout.LocalShape));
    }

    private static async Task<PrimFunction> RunReshapeSelectionAsync(CompileOptions compileOptions, DistributedType inputType, DistributedType outputType)
    {
        var input = new Var("input", inputType.TensorType);
        var boxed = Nncase.IR.F.Distributed.Boxing(input, inputType);
        var reshape = Nncase.IR.F.Tensors.Reshape(boxed, outputType.TensorType.Shape);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(reshape, input));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));
        reshape.CheckedType = outputType;

        return Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(compileOptions, CUDATarget.Kind).RunAsync(function, new()));
    }

    private static async Task<PrimFunction> RunBitcastSelectionAsync(CompileOptions compileOptions, DistributedType inputType, DistributedType outputType)
    {
        var input = new Var("input", inputType);
        var unary = Nncase.IR.F.Math.Unary(UnaryOp.Neg, input);
        unary.CheckedType = inputType;
        var bitcast = Nncase.IR.F.Tensors.Bitcast(unary, outputType.TensorType.DType);
        bitcast.CheckedType = outputType;
        var function = new Function("main", CUDATarget.Kind, new IRBlock(bitcast, input));

        return Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(compileOptions, CUDATarget.Kind).RunAsync(function, new()));
    }

    private static async Task<PrimFunction> RunD2DTransferSelectionAsync(CompileOptions compileOptions, DistributedType inputType, DistributedType outputType, bool force)
    {
        var input = new Var("input", inputType);
        var transfer = force
            ? Nncase.IR.F.Distributed.ForceBoxing(input, outputType)
            : Nncase.IR.F.Distributed.Boxing(input, outputType);
        transfer.CheckedType = outputType;
        var function = new Function("main", CUDATarget.Kind, new IRBlock(transfer, input));

        return Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(compileOptions, CUDATarget.Kind).RunAsync(function, new()));
    }

    private static void AssertBitcastDiagnostic(string message, string context)
    {
        Assert.Contains(context, message, StringComparison.Ordinal);
        Assert.Contains("InputDType", message, StringComparison.Ordinal);
        Assert.Contains("OutputDType", message, StringComparison.Ordinal);
        Assert.Contains("InputShape", message, StringComparison.Ordinal);
        Assert.Contains("OutputShape", message, StringComparison.Ordinal);
        Assert.Contains("ReverseLocal", message, StringComparison.Ordinal);
        Assert.Contains("element sizes differ", message, StringComparison.Ordinal);
    }

    private static void AssertD2DTransferDiagnostic(string message, string context, string layoutName, string reason)
    {
        Assert.Contains(context, message, StringComparison.Ordinal);
        Assert.Contains("InputDType", message, StringComparison.Ordinal);
        Assert.Contains("OutputDType", message, StringComparison.Ordinal);
        Assert.Contains("InputShape", message, StringComparison.Ordinal);
        Assert.Contains("OutputShape", message, StringComparison.Ordinal);
        Assert.Contains(layoutName, message, StringComparison.Ordinal);
        Assert.Contains(reason, message, StringComparison.Ordinal);
    }

    private static void AssertD2DLayoutVerificationDiagnostic(
        string message,
        string context,
        string layoutName,
        string roleReason,
        string verifierReason)
    {
        Assert.Contains(context, message, StringComparison.Ordinal);
        Assert.Contains("InputDType", message, StringComparison.Ordinal);
        Assert.Contains("OutputDType", message, StringComparison.Ordinal);
        Assert.Contains("InputShape", message, StringComparison.Ordinal);
        Assert.Contains("OutputShape", message, StringComparison.Ordinal);
        Assert.Contains(layoutName, message, StringComparison.Ordinal);
        Assert.Contains("layout verification failed", message, StringComparison.Ordinal);
        Assert.Contains(roleReason, message, StringComparison.Ordinal);
        Assert.Contains(verifierReason, message, StringComparison.Ordinal);
    }

    private static Dimension GetSingleBufferIndex(Call bufferAccess, ParameterInfo indicesParameter)
    {
        var indices = Assert.IsType<Nncase.IR.Tuple>(bufferAccess[indicesParameter]);
        var cast = Assert.IsType<Call>(Assert.Single(indices.Fields.ToArray()));
        Assert.IsType<Nncase.IR.Tensors.Cast>(cast.Target);
        var asTensor = Assert.IsType<Call>(cast.Arguments[0]);
        Assert.IsType<Nncase.IR.Shapes.AsTensor>(asTensor.Target);
        return Assert.IsAssignableFrom<Dimension>(asTensor.Arguments[0]);
    }

    private static long[] EvaluateIndexValues(Dimension dimension, DimVar loopVar, int extent, long programId, long threadId) =>
        Enumerable.Range(0, extent)
            .Select(lane => EvaluateDimension(dimension, loopVar, lane, programId, threadId))
            .ToArray();

    private static long EvaluateDimension(Dimension dim, DimVar loopVar, long lane, long programId, long threadId)
    {
        return dim switch
        {
            DimConst constant => constant.Value,
            DimVar dimVar when ReferenceEquals(dimVar, loopVar) || dimVar.Name == loopVar.Name => lane,
            ThreadIdDim => threadId,
            ProgramIdDim { Axis: 0 } => programId,
            DimSum sum => sum.Bias + sum.Operands.ToArray().Sum(x => EvaluateDimension(x, loopVar, lane, programId, threadId)),
            DimProduct product => product.Scale * product.Operands.ToArray().Aggregate(1L, (acc, x) => acc * EvaluateDimension(x, loopVar, lane, programId, threadId)),
            DimFraction fraction => EvaluateFraction(fraction, loopVar, lane, programId, threadId),
            DimRemainder remainder => EvaluateDimension(remainder.Numerator, loopVar, lane, programId, threadId) % EvaluateDimension(remainder.Denominator, loopVar, lane, programId, threadId),
            DimMin min => min.Operands.ToArray().Min(x => EvaluateDimension(x, loopVar, lane, programId, threadId)),
            DimMax max => max.Operands.ToArray().Max(x => EvaluateDimension(x, loopVar, lane, programId, threadId)),
            _ => throw new NotSupportedException($"Unsupported dimension expression {dim.GetType().Name}: {dim}"),
        };
    }

    private static long EvaluateFraction(DimFraction fraction, DimVar loopVar, long lane, long programId, long threadId)
    {
        var numerator = EvaluateDimension(fraction.Numerator, loopVar, lane, programId, threadId);
        var denominator = EvaluateDimension(fraction.Denominator, loopVar, lane, programId, threadId);
        return fraction.DivMode switch
        {
            DimDivideMode.FloorDiv => numerator / denominator,
            DimDivideMode.CeilDiv => (numerator + denominator - 1) / denominator,
            _ => throw new ArgumentOutOfRangeException(nameof(fraction), $"Unsupported divide mode {fraction.DivMode}."),
        };
    }

    private static void AssertRegisterDirectAffineLowering(PrimFunction lowered)
    {
        var fields = lowered.Body.Fields.ToArray();
        var allExprs = ExprCollector.Collect(lowered.Body).ToArray();
        var calls = allExprs.OfType<Call>().ToArray();

        Assert.Contains(fields, expr => expr is Nncase.TIR.For { Mode: LoopMode.Serial });
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineGather);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineScatter);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.VectorizedBinary);
        Assert.DoesNotContain(allExprs, expr => expr is Nncase.TIR.Buffer);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.Load);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.Store);
        var ret = Assert.IsType<Return>(fields[^1]);
        Assert.Empty(ret.Values.ToArray());
    }
}
