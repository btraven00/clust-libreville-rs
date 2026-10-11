#!/usr/bin/env bash
# bioc-export.sh OUTDIR: the R package as a standalone source package, ready for
# R CMD build / check / BiocCheck and for its own git repository (Bioconductor
# wants the package at a repository's root): fixture symlinks dereferenced, the
# core crate copied into src/rust/detleiden-core, crates.io dependencies
# vendored into src/rust/vendor.tar.xz (offline build, as Bioc/CRAN require).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
out=$1
rm -rf "$out"
mkdir -p "$out"
cp -rL "$here"/. "$out"
rm -rf "$out"/src/rust/target "$out"/src/*.o "$out"/src/*.so "$out"/src/Makevars "$out"/bioc-export.sh
core="$out/src/rust/detleiden-core"
mkdir -p "$core"
cp -r "$root/src" "$root/LICENSE" "$core/"
# the library only: no benches, examples or dev-dependencies
python3 - "$root/Cargo.toml" "$core/Cargo.toml" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
for sec in (r"\[dev-dependencies\]", r"\[\[bench\]\]", r"\[profile\.release\]"):
    s = re.sub(sec + r"\n(?:[^\[\n][^\n]*\n|\n)*", "", s)
open(sys.argv[2], "w").write(s)
PY
sed -i "s|path = '../../..'|path = 'detleiden-core'|" "$out/src/rust/Cargo.toml"
(cd "$out/src/rust" && cargo generate-lockfile -q)
(cd "$out" && Rscript -e 'rextendr::vendor_crates()')
echo "exported to $out"
