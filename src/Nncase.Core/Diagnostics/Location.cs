// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Diagnostics;

public abstract record Location
{
}

public record FileLocation(string FilePath, int StartLine, int StartColumn, int EndLine, int EndColumn) : Location
{
    public FileLocation(string filePath, int line, int column)
        : this(filePath, line, column, line, column)
    {
    }
}

public record NameLocation(string Name, Location? ChildLocation) : Location
{
}
