#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
work_dir="$repo_root/dump/ptx-thread-main-no-local-selftest"
checker="$script_dir/check-ptx-thread-main-no-local.sh"

rm -rf "$work_dir"
mkdir -p "$work_dir"

pass_file="$work_dir/wrapper-local-thread-main-clean.ptx"
cat > "$pass_file" <<'PTX'
.visible .entry native_entry()
{
        .local .align 8 .b8 wrapper_storage[16];
        ret;
}

.visible .func thread_main()
{
        add.u32 %r1, %r2, %r3;
        ret;
}
PTX

"$checker" "$pass_file"

for op in "ld.local.u32 %r1, [%rd1];" "st.local.u64 [%rd1], %rd2;" "cvta.local.u64 %rd1, %rd2;" "cvta.to.local.u64 %rd1, %rd2;"; do
  case_name="${op%% *}"
  fail_file="$work_dir/${case_name//./-}.ptx"
  cat > "$fail_file" <<PTX
.visible .func thread_main()
{
        $op
        ret;
}
PTX
  if "$checker" "$fail_file" >/dev/null 2>&1; then
    echo "expected $checker to reject $op in thread_main" >&2
    exit 1
  fi
done

echo "PTX thread_main local-memory self-test passed: $work_dir"
