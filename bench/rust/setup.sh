#!/bin/sh
# Recreate vendor/ (git-ignored) from the crates.io tarball + patches.
#   vendor/stock-trace = leiden-rs 0.8.1 + trace.patch   (LEIDEN_TRACE=1 prints per-level hashes/timings)
#   vendor/seqagg      = leiden-rs 0.8.1 + seqagg.patch + trace.patch
set -e
cd "$(dirname "$0")"
crate=$(ls ~/.cargo/registry/cache/*/leiden-rs-0.8.1.crate | head -1)
rm -rf vendor && mkdir -p vendor/stock-trace vendor/seqagg
tar xzf "$crate" -C vendor/stock-trace --strip-components=1
tar xzf "$crate" -C vendor/seqagg --strip-components=1
(cd vendor/stock-trace && patch -s -p1 < ../../trace.patch)
(cd vendor/seqagg && patch -s -p1 < ../../seqagg.patch && patch -s -p1 < ../../trace.patch)
# distinct versions so cargo can hold both path copies next to the crates.io one
sed -i 's/^version = "0.8.1"$/version = "0.8.1-trace"/' vendor/stock-trace/Cargo.toml
sed -i 's/^version = "0.8.1"$/version = "0.8.1-seqagg"/' vendor/seqagg/Cargo.toml
