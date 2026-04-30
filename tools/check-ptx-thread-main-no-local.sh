#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <thread-main-or-kernel.ptx>" >&2
  exit 2
fi

ptx_file="$1"
if [[ ! -f "$ptx_file" ]]; then
  echo "PTX file not found: $ptx_file" >&2
  exit 2
fi

awk '
BEGIN {
  found = 0
  in_signature = 0
  in_body = 0
  bad = 0
  local_pattern = "(^|[[:space:]])(\\.local|ld\\.local|st\\.local|cvta\\.local)([^[:alnum:]_.]|$)"
}

/\.visible[[:space:]]+\.func/ && /thread_main/ {
  found = 1
  in_signature = 1
}

in_signature && /\{/ {
  in_signature = 0
  in_body = 1
}

in_body {
  if ($0 ~ local_pattern) {
    print FILENAME ":" FNR ": forbidden local-memory op in thread_main: " $0
    bad = 1
  }

  if ($0 ~ /^}/) {
    in_body = 0
  }
}

END {
  if (!found) {
    print FILENAME ": missing .visible .func thread_main" > "/dev/stderr"
    exit 2
  }

  if (bad) {
    exit 1
  }
}
' "$ptx_file"
