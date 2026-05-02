---
name: gcov_coverage_build
description: How to correctly build software with gcov code coverage instrumentation — ensuring .gcno and .gcda files are generated in the right location for verification.
version: 1.0.0
required_tools: ['sandbox_sh']
---

# Building with gcov Code Coverage Instrumentation

Build software with gcov instrumentation so that compile-time (`.gcno`) and runtime (`.gcda`) coverage data files end up in the correct directory for verification.

## When to Use This Skill

- A task asks you to compile software **with gcov** or **code coverage instrumentation**.
- The task mentions `.gcno` files, `.gcda` files, or `--coverage` flags.
- The task requires running `gcov`, `lcov`, or `gcovr` on the built binary.

## The #1 Failure Mode: Wrong .gcda File Location

gcov coverage files are **tied to the build directory path**:
- `.gcno` files — created at **compile time** in the build directory
- `.gcda` files — created at **runtime** in the **same directory** as `.gcno` files

The binary records the absolute path to the build directory at compile time. When executed, it writes `.gcda` files to that recorded path — **regardless of where the binary itself is located**.

**This means:** If you build in `/tmp/build/` and install to `/app/pkg/`, the binary writes `.gcda` files to `/tmp/build/`, not `/app/pkg/`. Verifiers checking `/app/pkg/` for `.gcda` files will fail.

## Rule 1: Build In-Place in the Target Directory

Always configure and compile **directly in the directory where coverage files are expected**.

### Correct

```bash
mkdir -p /app/sqlite
cd /app/sqlite

# Extract source directly into the target directory
tar -xzf /app/vendor/sqlite-src.tar.gz --strip-components=1

# Configure with gcov and build in-place
./configure --gcov
make -j$(nproc)

# Symlink to PATH (don't copy — the binary must stay where it was built)
ln -sf /app/sqlite/sqlite3 /usr/local/bin/sqlite3
```

### Wrong

```bash
# WRONG: Separate build directory
mkdir /tmp/build && cd /tmp/build
/app/sqlite-src/configure --gcov
make -j$(nproc)
make install DESTDIR=/app/sqlite
# .gcno stays in /tmp/build, .gcda generated in /tmp/build → verification FAILS
```

```bash
# WRONG: Copying the binary away from the build directory
cp /app/sqlite/sqlite3 /usr/local/bin/sqlite3
# The copy works, but .gcda still gets written to /app/sqlite/ (good)
# However if you DELETE or clean /app/sqlite/ after copying, .gcda has nowhere to go
```

## Rule 2: Use the Right Configure Flags

Different projects support different ways to enable gcov. Try in this order:

| Method | Example | Notes |
|--------|---------|-------|
| Built-in flag | `./configure --gcov` | SQLite, some autoconf projects |
| Enable flag | `./configure --enable-gcov` | Common autoconf pattern |
| Manual CFLAGS | `CFLAGS="-fprofile-arcs -ftest-coverage" LDFLAGS="--coverage" ./configure` | Universal fallback |
| Shorthand | `CFLAGS="--coverage" LDFLAGS="--coverage" ./configure` | GCC shorthand for the above |

For CMake projects:
```bash
cmake -DCMAKE_C_FLAGS="--coverage" -DCMAKE_EXE_LINKER_FLAGS="--coverage" ..
```

### Important: Both Compile AND Link Flags

gcov requires flags at **both** compile and link stages:
- **Compile:** `-fprofile-arcs -ftest-coverage` (or `--coverage`)
- **Link:** `-lgcov` or `--coverage`

If you only set `CFLAGS` without `LDFLAGS`, linking may fail or `.gcda` files won't be generated.

## Rule 3: Make the Binary Available on PATH Without Moving It

After building, the binary must be on PATH but must **stay in the build directory** (so `.gcda` files go to the right place).

### Preferred: Symlink

```bash
ln -sf /app/sqlite/sqlite3 /usr/local/bin/sqlite3
```

### Also acceptable: Add build directory to PATH

```bash
export PATH="/app/sqlite:$PATH"
echo 'export PATH="/app/sqlite:$PATH"' >> /etc/profile.d/sqlite.sh
```

### Avoid: Copying the binary elsewhere

Copying works functionally, but the copied binary still writes `.gcda` to the original build directory. If that directory is cleaned up, coverage data is lost.

## Rule 4: Verify Coverage Files

After building, always verify that coverage instrumentation is working:

```bash
# 1. Check .gcno files exist in the build directory (compile-time)
echo "=== .gcno files (compile-time) ==="
find /app/sqlite -name "*.gcno" | head -5
# Must find files — if empty, gcov flags weren't applied correctly

# 2. Run the binary to generate .gcda files (runtime)
sqlite3 :memory: "SELECT 1;"

# 3. Check .gcda files appear alongside .gcno files
echo "=== .gcda files (runtime) ==="
find /app/sqlite -name "*.gcda" | head -5
# Must find files — if empty, build path is wrong or linking missed --coverage

# 4. Verify the binary contains gcov symbols
nm /app/sqlite/sqlite3 | grep -i gcov | head -3
# Should show __gcov symbols
```

## Rule 5: Use Pre-Vendored Sources When Available

If the task says source is pre-vendored (e.g., "source snapshot is pre-vendored at /app/vendor/..."), **always use it** instead of downloading from the network:

```bash
cd /app/sqlite
tar -xzf /app/vendor/sqlite-fossil-release.tar.gz --strip-components=1
```

This is faster, more reliable, and the task expects you to use it.

## Quick Reference

1. Extract source **into** the target directory (use `--strip-components=1`)
2. Configure and build **in-place** — never use a separate build directory
3. Use `--gcov`, `--enable-gcov`, or manual `CFLAGS/LDFLAGS` with `--coverage`
4. Symlink (don't copy) the binary to PATH
5. Verify: `.gcno` exists after build, `.gcda` exists after running the binary
