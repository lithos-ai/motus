---
name: source_package_build
description: How to correctly extract, build, and install software from source packages — preserving directory structure, building from the right working directory, ensuring binaries are accessible, handling legacy C/C++ compilation, and building from web-downloaded archives.
version: 2.0.0
required_tools: ['sandbox_sh']
---


# Building from Source Packages

Extract, build, and install software from source packages correctly. The most common failure mode is **destroying the expected directory layout** during extraction. This skill teaches you how to avoid that and get a clean, verifiable build.

## When to Use This Skill

- You need to download and build a package from source (e.g., via `apt-get source` or from the web).
- A task requires source code to exist in a specific directory with its original directory structure intact.
- You need to compile software using `make`, `dpkg-buildpackage`, `./configure && make`, or similar tools.
- You need to ensure a compiled binary is available on PATH after building.
- You need to build **legacy C/C++ software** (1990s–2000s era) on a modern Linux system.
- The task involves downloading source archives from the internet (FTP sites, archive.org, mirrors).

## Core Principle

> **Preserve the source directory as a subdirectory of the target.** When source extraction creates `package-1.2.3/`, that directory must appear INSIDE the target directory, not replace it.

---

## Rule 1: Preserve Directory Structure During Extraction

This is the single most important rule. When extracting source into a target directory (e.g., `/app/`), the extracted source directory must end up **inside** the target.

### ✅ Correct Approaches

**Best: Extract directly inside the target directory**

```bash
cd /app && apt-get source foo
# Result: /app/foo-1.2.3/  (source dir is INSIDE /app)
```

**Also correct: Copy the entire source directory into the target**

```bash
cp -a /tmp/foo-1.2.3 /app/foo-1.2.3
# Result: /app/foo-1.2.3/  (preserves the directory)
```

**Also correct: Move the source directory into the target**

```bash
mv /tmp/foo-1.2.3 /app/
# Result: /app/foo-1.2.3/
```

### ❌ Wrong Approaches

**WRONG: Copying contents with wildcard (flattens structure)**

```bash
cp -a /tmp/foo-1.2.3/* /app/
# Result: /app/Makefile, /app/src/, etc. — the foo-1.2.3 directory is GONE
# The target now IS the source tree, instead of CONTAINING it
```

**WRONG: Replacing the target with the source directory**

```bash
rm -rf /app && cp -a /tmp/foo-1.2.3 /app
# Result: /app/ is now the source tree itself
# Anything else that was in /app/ is destroyed
# Pattern /app/foo-*/ will NOT match because the directory level is missing
```

**WRONG: Extracting a tarball without checking what it creates**

```bash
cd /app && tar xf foo-1.2.3.tar.gz
# This MIGHT be correct (if the tarball contains a foo-1.2.3/ root dir)
# But some tarballs extract files directly — always verify after!
```

---

## Rule 2: Verify Layout Before Building

After extraction, **always** verify the directory structure before proceeding to the build step.

```bash
ls -la /app/
```

### What you SHOULD see

```
drwxr-xr-x  8 root root 4096 Jan  1 00:00 foo-1.2.3
-rw-r--r--  1 root root 1234 Jan  1 00:00 foo_1.2.3.dsc
-rw-r--r--  1 root root 5678 Jan  1 00:00 foo_1.2.3.orig.tar.gz
```

The key item is `foo-1.2.3/` as a **subdirectory** inside the target.

### What indicates a problem

```
-rw-r--r--  1 root root  1234 Jan  1 00:00 Makefile
drwxr-xr-x  3 root root  4096 Jan  1 00:00 src
-rw-r--r--  1 root root   567 Jan  1 00:00 configure
```

If you see raw source files (Makefile, src/, configure) directly in the target instead of inside a version-named subdirectory, the structure was flattened. **Stop and fix it** before building.

### Quick pattern check

```bash
# Verify a package-version directory exists inside the target
ls -d /app/foo-*/
# Should output something like: /app/foo-1.2.3/
```

---

## Rule 3: Build from the Correct Working Directory

Build commands must run **inside** the source directory, not above it.

### ✅ Correct

```bash
cd /app/foo-1.2.3 && ./configure && make
```

```bash
cd /app/foo-1.2.3 && dpkg-buildpackage -us -uc
```

```bash
cd /app/foo-1.2.3 && make -j$(nproc)
```

### ❌ Wrong

```bash
cd /app && make
# /app/ doesn't have a Makefile — the Makefile is in /app/foo-1.2.3/
```

```bash
cd /app && dpkg-buildpackage -us -uc
# Wrong directory level
```

### Tip: Find the right directory dynamically

If you don't know the exact version number:

```bash
# Use a glob to find the source directory
cd /app/foo-*/ && make
```

Or discover it first:

```bash
SRC_DIR=$(ls -d /app/foo-*/ | head -1)
cd "$SRC_DIR" && ./configure && make
```

---

## Rule 4: Using `apt-get source` Correctly

`apt-get source` is the standard way to fetch Debian/Ubuntu source packages. It requires source repositories to be enabled.

### Step-by-step

```bash
# 1. Enable source repositories
sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list
apt-get update

# 2. Install build dependencies (if needed)
apt-get build-dep -y packagename

# 3. Navigate to the target directory FIRST, then fetch the source
cd /app
apt-get source packagename

# 4. Verify the layout
ls -la /app/
# Should show: packagename-X.Y.Z/ plus .dsc, .orig.tar.gz, .diff.gz files

# 5. Enter the source directory and build
cd /app/packagename-*/
dpkg-buildpackage -us -uc -b
# Or: ./configure && make
```

### Important notes

- `apt-get source` extracts into the **current working directory** — that's why you must `cd` to the target first.
- It creates a directory named `packagename-version/` automatically.
- It also downloads `.dsc`, `.orig.tar.gz`, and sometimes `.diff.gz` or `.debian.tar.xz` files alongside the directory.
- You do NOT need to be root to download source, but you do need root for `apt-get update` and `apt-get build-dep`.

---

## Rule 5: Ensure the Binary Is on PATH

After a successful build, the compiled binary may not be accessible from the shell. Verify and fix this.

### Check if the binary exists and where

```bash
# Look for the binary in common locations
find /app/packagename-*/ -type f -executable -name "packagename"

# Or after 'make install':
which packagename
```

### Common approaches to make the binary accessible

**Option A: `make install` (installs to system paths, usually /usr/local/bin/)**

```bash
cd /app/packagename-*/ && make install
which packagename   # should show /usr/local/bin/packagename
```

**Option B: Copy the binary to a PATH directory**

```bash
cp /app/packagename-*/src/packagename /usr/local/bin/
chmod +x /usr/local/bin/packagename
```

**Option C: Add the build directory to PATH**

```bash
export PATH="/app/packagename-*/src:$PATH"
```

**Option D: Install built `.deb` packages (if using dpkg-buildpackage)**

```bash
# After dpkg-buildpackage, .deb files appear one level up from the source dir
ls /app/*.deb
dpkg -i /app/packagename_*.deb
```

### Verify the binary works

```bash
packagename --version
# or
packagename --help
```

---

## Error Handling

### Source repo not available

```
E: Unable to find a source package for packagename
```

**Fix:** Ensure source repos are enabled and updated:

```bash
# Check if deb-src lines exist
grep deb-src /etc/apt/sources.list

# If missing or commented out, enable them
sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list
apt-get update
```

On newer Ubuntu systems using `.sources` format:

```bash
# Check for the new format
ls /etc/apt/sources.list.d/*.sources
# Enable source by setting Types to include deb-src
sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/*.sources
apt-get update
```

### Missing build dependencies

```
dpkg-checkbuilddeps: error: Unmet build dependencies: libfoo-dev libbar-dev
```

**Fix:**

```bash
apt-get build-dep -y packagename
# Or install specific missing packages:
apt-get install -y libfoo-dev libbar-dev
```

### Build fails with compiler errors

1. Check you're in the correct directory (`pwd` should show the source dir).
2. Ensure build dependencies are installed.
3. Read the error output — the most common issues are missing `-dev` packages.

### Binary not found after build

```bash
# Search for any executables that were built
find /app/packagename-*/ -type f -executable | head -20

# Check if 'make install' would help
cd /app/packagename-*/ && make install

# Or look for built .deb packages
ls /app/*.deb
```

---

## Complete Example: Building `sl` from Source

```bash
# Enable source repositories
sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list
apt-get update

# Install build dependencies
apt-get build-dep -y sl

# Extract source into target directory
cd /app
apt-get source sl

# Verify directory structure
ls -la /app/
# drwxr-xr-x  2 root root 4096 ... sl-5.02/
# -rw-r--r--  1 root root  ... sl_5.02.dsc
# -rw-r--r--  1 root root  ... sl_5.02.orig.tar.gz
# ...

# Confirm the source directory exists at the right level
ls -d /app/sl-*/
# /app/sl-5.02/

# Enter source directory and build
cd /app/sl-*/
make

# Verify the binary was built
ls -la sl
# -rwxr-xr-x  1 root root 18432 ... sl

# Install so it's on PATH
make install
# Or: cp sl /usr/local/bin/

# Final verification
which sl
sl --help
```

---

## Rule 6: Building from Web-Downloaded Archives

When source is not available via `apt-get source` (e.g., legacy software, custom packages, old versions), you must download archives directly from the internet.

### Downloading Source Archives

Try these sources in order:
1. **Official project website/FTP** (e.g., `https://www.povray.org/ftp/pub/...`)
2. **archive.org** (Wayback Machine snapshots of old FTP servers)
3. **Mirror sites** (e.g., `ftp.mirrorservice.org`)
4. **SourceForge** (for older open-source projects)

```bash
# Install download tools
apt-get update && apt-get install -y wget curl

# Download with retries
wget --tries=3 --timeout=30 https://example.org/package-src.tar.gz
```

### Handling .TAR.Z (compress format) Archives

Older software (pre-2000) often uses `.TAR.Z` format (Unix compress). This requires `ncompress`:

```bash
apt-get install -y ncompress

# Decompress .Z files first, then extract .TAR
uncompress ARCHIVE1.TAR.Z ARCHIVE2.TAR.Z
tar -xf ARCHIVE1.TAR
tar -xf ARCHIVE2.TAR
```

### Multi-Archive Distributions (CRITICAL)

Many legacy packages split their source into **multiple archives** (e.g., separate archives for source code, documentation, and scene/data files). **You MUST download and extract ALL of them**, not just the source code archive.

Example: A package with 3 archives:
- `PKGSRC.TAR.Z` — source code and build files
- `PKGDOC.TAR.Z` — documentation and include files
- `PKGSCN.TAR.Z` — sample data/scene files

Download **all three**. The verifier often checks for files from the documentation or data archives.

### Preserving ALL Distribution Files (CRITICAL — #1 Failure Mode)

When organizing extracted files into the target directory, you MUST preserve **every file from the distribution** — not just the source code. This includes:

- `*.doc` — documentation files
- `*.diz` — distribution information (e.g., `file_id.diz`)
- `*.cat` — catalog files
- `README`, `LICENSE`, `COPYING`, `CHANGELOG`
- `*.inc` — include/header files (often in doc subdirs)
- Any other metadata files at the top level of the extracted archive

```bash
# After extracting archives to /app, organize into target directory
mkdir -p /app/package-1.2.3

# Move ALL directories
mv source/ povdoc/ povscn/ machine/ /app/package-1.2.3/

# CRITICAL: Also move ALL loose files (docs, metadata, catalogs)
for file in *.doc *.diz *.cat *.txt README* LICENSE* COPYING*; do
    [ -f "$file" ] && mv "$file" /app/package-1.2.3/
done
```

**Why this matters:** Verifiers commonly validate that the source tree contains authentic distribution files using MD5 hash checks. Missing files like `file_id.diz` will cause verification failure even if the build itself succeeds.

### Verification After Organization

```bash
# Check that ALL expected files are present in the target
ls -la /app/package-1.2.3/
# Should see: source dirs AND loose distribution files (*.doc, *.diz, etc.)

# Count files to make sure nothing was lost
find /app/package-1.2.3/ -type f | wc -l
```

---

## Rule 7: Building Legacy C/C++ Code on Modern Systems

Software from the 1990s–early 2000s often won't compile on modern systems without patches. Here are the most common fixes.

### Compiler Flags

```bash
# Use C89 standard mode (most legacy C code predates C99)
sed -i 's/CFLAGS=.*-c/CFLAGS=\t\t-c -std=c89 -Wno-error/' Makefile

# Ensure gcc is used (some old Makefiles default to cc)
sed -i 's/CC =.*/CC = gcc/' Makefile
```

### Common Compilation Fixes

**1. Missing `#include <stdlib.h>`** — Modern compilers require explicit declaration of `malloc`, `free`, `exit`, `atoi`, etc.

```bash
# Add stdlib.h to all .c files that don't have it
for file in *.c; do
    if ! grep -q "#include <stdlib.h>" "$file"; then
        sed -i '1i#include <stdlib.h>' "$file"
    fi
done
```

**2. `void main()` → `int main()`** — Old code often declares `main` as returning void, which modern compilers reject.

```bash
# In the config/frame header, change the main return type
sed -i 's/#define MAIN_RETURN_TYPE void/#define MAIN_RETURN_TYPE int/' frame.h
```

**3. Old-style `malloc`/`calloc` declarations** — Legacy code sometimes declares `char *malloc()` which conflicts with modern `<stdlib.h>`.

```bash
# Comment out old-style malloc declarations
sed -i 's/char \*malloc.*PARAMS.*/\/\* & \*\//' config.h
```

**4. Missing function declarations** — Add forward declarations for functions defined in platform-specific files.

```bash
# Add missing function declarations to config header
echo "void platform_init_PROGRAM();" >> config.h
```

### Deterministic Builds (When Output Comparison Is Required)

If the task verifies output by comparing against a reference (e.g., image rendering, data processing), the build MUST produce **deterministic, reproducible output**. The most common source of non-determinism is the system `rand()` function.

**Fix: Replace system RNG with a deterministic implementation.**

Check if the source has platform-specific RNG implementations (often in `ibm.c`, `unix.c`, or similar). If the Unix build doesn't include one, add a deterministic linear congruential generator:

```c
// Deterministic RNG (portable, reproducible across platforms)
static unsigned long int rng_next = 1;

int pov_rand(void)
{
    rng_next = rng_next * 1103515245L + 12345L;
    return ((int)(rng_next / 0x10000L) & 0x7FFF);
}

void pov_srand(seed)
int seed;
{
    rng_next = (unsigned long int)seed;
}
```

Append this to the platform-specific source file (e.g., `unix.c`) using:

```bash
cat >> unix.c << 'DETRNG'
// ... (paste the deterministic RNG code above)
DETRNG
```

**Why this matters:** Without a deterministic RNG, renders or other outputs may vary slightly between runs, causing output comparison tests (e.g., SSIM image similarity) to fail intermittently.

### Platform File Setup

Legacy software often has platform-specific files in a `machine/` directory. For Unix/Linux builds:

```bash
# Copy Unix platform files to the source directory
cp machine/unix/* source/

# Rename platform config header
cd source/
mv unixconf.h config.h
```

---

## Quick Reference Checklist

Before moving from extraction to building, verify:

- [ ] Source directory exists **inside** the target (e.g., `/app/foo-1.2.3/` not `/app/Makefile`)
- [ ] `ls -d /app/packagename-*/` matches the expected pattern
- [ ] **ALL distribution files are preserved** — not just source code, but also `*.doc`, `*.diz`, `README`, etc.
- [ ] You have `cd`-ed into the source directory before running build commands
- [ ] Build dependencies are installed (`apt-get build-dep -y packagename`)

After building, verify:

- [ ] The binary was compiled successfully (check exit code)
- [ ] The binary is accessible (via `which`, direct path, or `.deb` install)
- [ ] The binary runs correctly (`--version` or `--help`)
- [ ] If output comparison is used: verify output is deterministic (run twice, compare results)
