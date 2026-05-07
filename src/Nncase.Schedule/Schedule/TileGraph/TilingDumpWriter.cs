// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Schedule.TileGraph;

internal static class TilingDumpWriter
{
    private const string RootNodeId = "root";

    public static void Dump(TreeSolveResult result, PrimFunction primFunction, long maxAlign)
    {
        if (!DumpScope.Current.IsEnabled(DumpFlags.Tiling))
        {
            return;
        }

        var model = new ScheduleDumpModel(result, primFunction, maxAlign);
        model.Collect();
        WriteText("auto_tiling_summary.md", BuildMarkdown(model));
        WriteText("auto_tiling_schedule.dot", BuildDot(model));
        WriteText("auto_tiling_schedule.svg", BuildSvg(model));
    }

    private static string BuildMarkdown(ScheduleDumpModel model)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# Auto Tiling Schedule: {model.FunctionName}");
        sb.AppendLine();
        sb.AppendLine($"- objective: {model.ObjectiveValue.ToString(CultureInfo.InvariantCulture)}");
        sb.AppendLine($"- max_alignment: {model.MaxAlignment.ToString(CultureInfo.InvariantCulture)}");
        sb.AppendLine($"- memory_hierarchy: {model.MemoryHierarchy}");
        sb.AppendLine($"- loop_count: {model.Loops.Count.ToString(CultureInfo.InvariantCulture)}");
        sb.AppendLine($"- max_loop_depth: {model.MaxLoopDepth.ToString(CultureInfo.InvariantCulture)}");
        sb.AppendLine($"- allocated_buffer_count: {model.Buffers.Count.ToString(CultureInfo.InvariantCulture)}");
        sb.AppendLine($"- call_count: {model.Calls.Count.ToString(CultureInfo.InvariantCulture)}");
        sb.AppendLine();

        sb.AppendLine("## Loop Nest");
        if (model.Loops.Count == 0)
        {
            sb.AppendLine("- none");
        }
        else
        {
            foreach (var loop in model.Loops)
            {
                sb.AppendLine($"{Indent(loop.Depth)}- L{loop.Id}: {loop.Var} {loop.Domain}, mode={loop.Mode}");
            }
        }

        sb.AppendLine();
        sb.AppendLine("## Allocated Buffers");
        if (model.Buffers.Count == 0)
        {
            sb.AppendLine("- none");
        }
        else
        {
            foreach (var buffer in model.Buffers)
            {
                sb.AppendLine(
                    $"- B{buffer.Id}: {buffer.Name} as {buffer.VarName}, open_at={buffer.OpenAt}, " +
                    $"size={buffer.SizeBytes}, offset={buffer.Offset}, storage=({buffer.Storage}), " +
                    $"shape={buffer.Shape}, strides={buffer.Strides}, type={buffer.Type}");
            }
        }

        sb.AppendLine();
        sb.AppendLine("## Solver Buffer Plan");
        if (model.SolverBuffers.Count == 0)
        {
            sb.AppendLine("- none");
        }
        else
        {
            foreach (var buffer in model.SolverBuffers)
            {
                sb.AppendLine(
                    $"- {buffer.NodeBuffer}: store={buffer.StoreLevelDisplayName}(level={buffer.StoreLevel}), create_loop={buffer.CreateLoop}, " +
                    $"size={buffer.SizeBytes}, offset={buffer.Offset}, shape={buffer.Shape}, strides={buffer.Strides}, " +
                    $"liveness={buffer.Liveness}, budget={buffer.BudgetBytes}");
            }
        }

        sb.AppendLine();
        sb.AppendLine("## Calls");
        if (model.Calls.Count == 0)
        {
            sb.AppendLine("- none");
        }
        else
        {
            foreach (var call in model.Calls)
            {
                sb.AppendLine($"{Indent(call.Depth)}- C{call.Id}: {call.OpKind} at {call.OpenAt}");
            }
        }

        return sb.ToString();
    }

    private static string BuildDot(ScheduleDumpModel model)
    {
        var sb = new StringBuilder();
        sb.AppendLine("digraph AutoTilingSchedule {");
        sb.AppendLine("  rankdir=TB;");
        sb.AppendLine("  graph [fontname=\"monospace\"];");
        sb.AppendLine("  node [fontname=\"monospace\", shape=box, style=\"rounded,filled\", fillcolor=\"#ffffff\"];");
        sb.AppendLine("  edge [fontname=\"monospace\"];");
        sb.AppendLine($"  {RootNodeId} [label=\"{DotEscape(model.FunctionName)}\\nloops={model.Loops.Count}, max_depth={model.MaxLoopDepth}, buffers={model.Buffers.Count}\", fillcolor=\"#e8f0fe\"];");

        foreach (var loop in model.Loops)
        {
            sb.AppendLine($"  loop{loop.Id} [label=\"L{loop.Id}: {DotEscape(loop.Var)}\\n{DotEscape(loop.Domain)}\\nmode={DotEscape(loop.Mode)}\", fillcolor=\"#d9ead3\"];");
            sb.AppendLine($"  {loop.ParentNodeId} -> loop{loop.Id} [label=\"contains\"];");
        }

        foreach (var buffer in model.Buffers)
        {
            sb.AppendLine(
                $"  buffer{buffer.Id} [shape=folder, label=\"B{buffer.Id}: {DotEscape(buffer.Name)}\\n" +
                $"var={DotEscape(buffer.VarName)}\\nsize={DotEscape(buffer.SizeBytes)} bytes, offset={DotEscape(buffer.Offset)}\\n" +
                $"{DotEscape(buffer.Storage)}\\nshape={DotEscape(buffer.Shape)}\", fillcolor=\"{BufferColor(buffer.Location)}\"];");
            sb.AppendLine($"  {buffer.ParentNodeId} -> buffer{buffer.Id} [label=\"alloc\"];");
        }

        foreach (var call in model.Calls)
        {
            sb.AppendLine($"  call{call.Id} [shape=ellipse, label=\"C{call.Id}: {DotEscape(call.OpKind)}\", fillcolor=\"#fce5cd\"];");
            sb.AppendLine($"  {call.ParentNodeId} -> call{call.Id} [label=\"exec\"];");
        }

        sb.AppendLine("}");
        return sb.ToString();
    }

    private static string BuildSvg(ScheduleDumpModel model)
    {
        const int Width = 1600;
        const int LeftX = 32;
        const int RightX = 740;
        const int RowHeight = 24;
        var rows = Math.Max(model.Loops.Count + model.Calls.Count + 4, model.Buffers.Count + model.SolverBuffers.Count + 4);
        var height = Math.Max(260, 128 + rows * RowHeight);
        var sb = new StringBuilder();
        sb.AppendLine($"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{Width}\" height=\"{height}\" viewBox=\"0 0 {Width} {height}\">");
        sb.AppendLine("<style>text{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:14px}.title{font-size:20px;font-weight:700}.h{font-size:16px;font-weight:700}.small{font-size:12px}.box{stroke:#444;stroke-width:1;rx:6;ry:6}</style>");
        sb.AppendLine("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>");
        sb.AppendLine($"<text class=\"title\" x=\"{LeftX}\" y=\"32\">{XmlEscape(model.FunctionName)}</text>");
        sb.AppendLine($"<text x=\"{LeftX}\" y=\"56\">objective={model.ObjectiveValue}, loops={model.Loops.Count}, max_depth={model.MaxLoopDepth}, buffers={model.Buffers.Count}, calls={model.Calls.Count}</text>");
        sb.AppendLine($"<text class=\"h\" x=\"{LeftX}\" y=\"92\">Loop Nest / Calls</text>");
        sb.AppendLine($"<text class=\"h\" x=\"{RightX}\" y=\"92\">Buffer Allocations</text>");

        var y = 122;
        foreach (var loop in model.Loops)
        {
            var x = LeftX + loop.Depth * 28;
            sb.AppendLine(Rect(x - 8, y - 16, 660 - loop.Depth * 28, 22, "#d9ead3"));
            sb.AppendLine($"<text x=\"{x}\" y=\"{y}\">L{loop.Id}: {XmlEscape(loop.Var)} {XmlEscape(loop.Domain)} mode={XmlEscape(loop.Mode)}</text>");
            y += RowHeight;
        }

        if (model.Calls.Count > 0)
        {
            y += 8;
            foreach (var call in model.Calls)
            {
                var x = LeftX + call.Depth * 28;
                sb.AppendLine(Rect(x - 8, y - 16, 660 - call.Depth * 28, 22, "#fce5cd"));
                sb.AppendLine($"<text x=\"{x}\" y=\"{y}\">C{call.Id}: {XmlEscape(call.OpKind)}</text>");
                y += RowHeight;
            }
        }

        y = 122;
        foreach (var buffer in model.Buffers)
        {
            sb.AppendLine(Rect(RightX - 8, y - 16, 820, 22, BufferColor(buffer.Location)));
            sb.AppendLine($"<text x=\"{RightX}\" y=\"{y}\">B{buffer.Id}: {XmlEscape(buffer.Name)} size={XmlEscape(buffer.SizeBytes)} offset={XmlEscape(buffer.Offset)} open_at={XmlEscape(buffer.OpenAt)} {XmlEscape(buffer.Location)}</text>");
            y += RowHeight;
        }

        if (model.SolverBuffers.Count > 0)
        {
            y += 16;
            sb.AppendLine($"<text class=\"h\" x=\"{RightX}\" y=\"{y}\">Solver Plan</text>");
            y += RowHeight;
            foreach (var buffer in model.SolverBuffers)
            {
                sb.AppendLine($"<text class=\"small\" x=\"{RightX}\" y=\"{y}\">{XmlEscape(buffer.NodeBuffer)} store={XmlEscape(buffer.StoreLevelDisplayName)}[{buffer.StoreLevel}] create_loop={buffer.CreateLoop} size={buffer.SizeBytes} offset={buffer.Offset} live={XmlEscape(buffer.Liveness)}</text>");
                y += 18;
            }
        }

        sb.AppendLine("</svg>");
        return sb.ToString();
    }

    private static string Rect(int x, int y, int width, int height, string fill) =>
        $"<rect class=\"box\" x=\"{x}\" y=\"{y}\" width=\"{width}\" height=\"{height}\" fill=\"{fill}\"/>";

    private static void WriteText(string fileName, string text)
    {
        using var stream = DumpScope.Current.OpenFile(fileName);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false));
        writer.Write(text);
    }

    private static string Indent(int depth) => new(' ', depth * 2);

    private static string BufferColor(string location) => NormalizeDisplayName(location) switch
    {
        "REGISTER" => "#d9ead3",
        "SMEM" or "SHAREDMEMORY" => "#cfe2f3",
        "TMEM" => "#d9d2e9",
        "GMEM" or "GLOBAL" or "DRAM" => "#f4cccc",
        "L1" or "L2" => "#fff2cc",
        "LOCAL" or "LOCALADDRESSABLE" => "#fce5cd",
        _ => "#eeeeee",
    };

    private static string NormalizeDisplayName(string value)
        => value.Replace("_", string.Empty, StringComparison.Ordinal).Replace("-", string.Empty, StringComparison.Ordinal).ToUpperInvariant();

    private static string DotEscape(string text) => text
        .Replace("\\", "\\\\", StringComparison.Ordinal)
        .Replace("\"", "\\\"", StringComparison.Ordinal)
        .Replace("\r", string.Empty, StringComparison.Ordinal)
        .Replace("\n", "\\n", StringComparison.Ordinal);

    private static string XmlEscape(string text) => text
        .Replace("&", "&amp;", StringComparison.Ordinal)
        .Replace("<", "&lt;", StringComparison.Ordinal)
        .Replace(">", "&gt;", StringComparison.Ordinal)
        .Replace("\"", "&quot;", StringComparison.Ordinal);

    private sealed record LoopDump(
        int Id,
        string ParentNodeId,
        int Depth,
        string Var,
        string Domain,
        string Mode);

    private sealed record BufferDump(
        int Id,
        string ParentNodeId,
        int Depth,
        string VarName,
        string Name,
        string Type,
        string Storage,
        string Location,
        string SizeBytes,
        string Offset,
        string Shape,
        string Strides,
        string OpenAt);

    private sealed record CallDump(
        int Id,
        string ParentNodeId,
        int Depth,
        string OpKind,
        string OpenAt);

    private sealed record SolverBufferDump(
        string NodeBuffer,
        int StoreLevel,
        string StoreLevelDisplayName,
        int CreateLoop,
        long SizeBytes,
        ulong Offset,
        string Shape,
        string Strides,
        string Liveness,
        int BudgetBytes);

    private sealed class ScheduleDumpModel
    {
        private readonly List<string> _loopNodeStack = new();
        private readonly List<string> _loopNameStack = new();
        private int _nextLoopId;
        private int _nextBufferId;
        private int _nextCallId;

        public ScheduleDumpModel(TreeSolveResult result, PrimFunction primFunction, long maxAlign)
        {
            Result = result;
            PrimFunction = primFunction;
            FunctionName = primFunction.Name;
            ObjectiveValue = result.ObjectiveValue;
            MaxAlignment = maxAlign;
        }

        public TreeSolveResult Result { get; }

        public PrimFunction PrimFunction { get; }

        public string FunctionName { get; }

        public long ObjectiveValue { get; }

        public long MaxAlignment { get; }

        public string MemoryHierarchy => FormatMemoryHierarchy(Result.TargetOptions.MemoryHierarchyLevels);

        public List<LoopDump> Loops { get; } = new();

        public List<BufferDump> Buffers { get; } = new();

        public List<CallDump> Calls { get; } = new();

        public List<SolverBufferDump> SolverBuffers { get; } = new();

        public int MaxLoopDepth => Loops.Count == 0 ? 0 : Loops.Max(l => l.Depth + 1);

        private string CurrentParentNodeId => _loopNodeStack.Count == 0 ? RootNodeId : _loopNodeStack[^1];

        private string CurrentOpenAt => _loopNameStack.Count == 0 ? "function_entry" : string.Join("/", _loopNameStack);

        private int CurrentDepth => _loopNameStack.Count;

        public void Collect()
        {
            ValidateMemoryHierarchyLevels();
            CollectSolverBuffers();
            Visit(PrimFunction.Body);
        }

        private void Visit(Expr expr)
        {
            switch (expr)
            {
                case Sequential sequential:
                    foreach (var field in sequential.Fields)
                    {
                        Visit(field);
                    }

                    break;
                case For loop:
                    VisitFor(loop);
                    break;
                case Let let:
                    VisitLet(let);
                    break;
                case Call call:
                    Calls.Add(new CallDump(_nextCallId++, CurrentParentNodeId, CurrentDepth, call.Target.GetType().Name, CurrentOpenAt));
                    break;
            }
        }

        private void VisitFor(For loop)
        {
            var id = _nextLoopId++;
            var nodeId = $"loop{id}";
            var loopName = loop.LoopVar.Name;
            Loops.Add(new LoopDump(id, CurrentParentNodeId, CurrentDepth, loopName, FormatRange(loop.Domain), loop.Mode.ToString()));
            _loopNodeStack.Add(nodeId);
            _loopNameStack.Add(loopName);
            Visit(loop.Body);
            _loopNameStack.RemoveAt(_loopNameStack.Count - 1);
            _loopNodeStack.RemoveAt(_loopNodeStack.Count - 1);
        }

        private void VisitLet(Let let)
        {
            if (let.Expression is Call { Target: Nncase.IR.Buffers.AllocateBufferView } call &&
                call.Arguments[Nncase.IR.Buffers.AllocateBufferView.Buffer.Index] is Nncase.TIR.Buffer buffer)
            {
                var physical = buffer.MemSpan.Buffer;
                Buffers.Add(new BufferDump(
                    _nextBufferId++,
                    CurrentParentNodeId,
                    CurrentDepth,
                    let.Var.Name,
                    buffer.Name,
                    buffer.Type.ToString(),
                    FormatStorage(buffer.Storage),
                    FormatStorageLocation(buffer.Storage),
                    buffer.MemSpan.Size.ToString(),
                    FormatExpr(physical.Start),
                    FormatDimensions(buffer.Dimensions),
                    FormatDimensions(buffer.Strides),
                    CurrentOpenAt));
            }

            Visit(let.Body);
        }

        private void CollectSolverBuffers()
        {
            foreach (var (level, nodeBuffers) in Result.LevelNodeBufferInfos.OrderBy(kv => kv.Key))
            {
                foreach (var (nodeBuffer, info) in nodeBuffers.OrderBy(kv => kv.Key.Node.OpId).ThenBy(kv => kv.Key.Id.ToString()))
                {
                    var createLoop = Result.TileNodeMemo[nodeBuffer.Node].BufferInfoMap[nodeBuffer.Id].GetLastRelatedPos();
                    SolverBuffers.Add(new SolverBufferDump(
                        nodeBuffer.Id.ToString(),
                        level,
                        GetMemoryHierarchyLevel(level).DisplayName,
                        createLoop,
                        info.Size,
                        info.Offset,
                        FormatLongs(info.Shape),
                        FormatLongs(info.Strides),
                        info.Liveness.ToString(),
                        Result.TargetOptions.MemoryCapacities[level]));
                }
            }
        }

        private string FormatRange(Nncase.TIR.Range range) => $"({range.Start}, {range.Stop}, {range.Step})";

        private string FormatExpr(Expr expr) => expr is None ? "none" : expr.ToString();

        private string FormatDimensions(ReadOnlySpan<Dimension> dimensions) => $"[{string.Join(", ", dimensions.ToArray().Select(d => d.ToString()))}]";

        private string FormatLongs(ReadOnlySpan<long> values) => $"[{string.Join(", ", values.ToArray())}]";

        private string FormatStrings(ReadOnlySpan<string> values) => $"[{string.Join(", ", values.ToArray())}]";

        private string FormatStorage(BufferStorage storage)
        {
            var displayLocation = FormatStorageLocation(storage);
            return $"Usage={storage.Usage}, Scope={storage.Scope}, Location={displayLocation}, Hierarchy={storage.Hierarchy}, Alignment={storage.Alignment}";
        }

        private string FormatStorageLocation(BufferStorage storage) => GetStorageDisplayName(storage);

        private string GetStorageDisplayName(BufferStorage storage)
        {
            var level = GetMemoryHierarchyLevel(storage.Hierarchy);
            if (level.PhysicalLocation != storage.PhysicalLocation || level.Scope != storage.Scope)
            {
                throw new InvalidOperationException(
                    $"Buffer storage {storage} refers to hierarchy level {storage.Hierarchy}, but target level is {level}. " +
                    "Storage and target memory hierarchy attributes must agree.");
            }

            return level.DisplayName;
        }

        private MemoryHierarchyLevel GetMemoryHierarchyLevel(int level) =>
            Result.TargetOptions.GetRequired(level, "Auto tiling dump");

        private string FormatMemoryHierarchy(ReadOnlySpan<MemoryHierarchyLevel> values) =>
            $"[{string.Join(", ", values.ToArray().Select(x => x.ToString()))}]";

        private void ValidateMemoryHierarchyLevels()
        {
            var levels = Result.TargetOptions.MemoryHierarchyLevels;
            if (levels.Length == 0)
            {
                throw new InvalidOperationException($"{nameof(INTTTargetOptions.MemoryHierarchyLevels)} must not be empty when dumping auto tiling.");
            }

            for (var i = 0; i < levels.Length; i++)
            {
                if (string.IsNullOrWhiteSpace(levels[i].DisplayName))
                {
                    throw new InvalidOperationException($"{nameof(INTTTargetOptions.MemoryHierarchyLevels)} contains an empty display name at level {i}.");
                }
            }
        }
    }
}
