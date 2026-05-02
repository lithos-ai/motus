---
name: dna_primer_design
description: How to correctly design PCR primers for Gibson Assembly, Golden Gate Assembly, and related molecular cloning techniques — covering overhang-template overlap detection, effective annealing region calculation, robust Tm computation, and boundary ambiguity handling.
version: 2.0.0
required_tools: ['sandbox_sh']
---

# DNA Primer Design for Molecular Cloning

Design PCR primers for Gibson Assembly, Golden Gate Assembly, and related cloning techniques with correct melting temperature (Tm) calculations that account for overhang-template overlap — the #1 source of primer design errors.

## When to Use This Skill

- Designing primers for Gibson Assembly (overlapping homologous ends)
- Designing primers for Golden Gate Assembly (Type IIS restriction enzyme-based)
- Designing primers for In-Fusion cloning or similar overlap-based methods
- Any primer design where overhangs (non-annealing 5' tails) are added to primers
- When a task specifies Tm constraints for primer binding regions
- When inserting a sequence into a vector or joining DNA fragments

## Core Concept: The Overhang-Template Overlap Problem

This is the **#1 failure mode** in primer design for assembly cloning. Understanding it is critical.

When you add an overhang to a primer (e.g., for Gibson Assembly homology arms), some or all of those overhang bases may **match the template sequence adjacent to the binding site**. This means the primer anneals over a **longer region than you designed**, producing a higher Tm than expected.

### Visual Example

```
Template (5'→3'):  ...TTACGGTAATGCCTGAACCC...
                          ↑ binding starts here (position 7)

Designed binding region:   ATGCCTGAA        (9 nt, Tm ≈ 28°C)
Add Gibson overhang:  ggta·ATGCCTGAA       (overhang·binding)

But look at the template upstream of the binding site:
Template:          ...GGTAATGCCTGAA...
                      ^^^^                 ← overhang matches!

Effective annealing region: GGTAATGCCTGAA  (13 nt, Tm ≈ 44°C)
                            ↑ 4 extra bases from overhang match
```

**The Tm you report must be for the EFFECTIVE annealing region (13 nt), not the designed binding region (9 nt).**

If the task says "Tm must be 58–72°C" and your effective Tm is 76°C because of overhang overlap, the primer **fails the constraint** even though the designed binding portion alone was fine.

---

## Tool Configuration

| Tool | Purpose | Example |
|------|---------|---------|
| `sandbox_sh` | Run Python/Biopython and CLI tools (e.g., `oligotm`) for Tm calculations, sequence manipulation, primer verification | `sandbox_sh("oligotm -tp 1 -sc 1 -mv 50 -dv 2 -n 0.8 -d 500 ATGCCTGAA")` |

All computation MUST be done via `sandbox_sh`. Never estimate Tm by hand or use the basic formula (4°C per GC + 2°C per AT) — it is not accurate enough. Use the Tm calculation tool specified in the task instructions.

---

## Step-by-Step Workflow

### Phase 1: Understand the Task Requirements

Before designing anything, extract these parameters:

1. **Template sequence(s)** — the DNA the primers will bind to
2. **Target region** — what portion to amplify or where to insert
3. **Tm constraints** — acceptable range (e.g., 58–72°C) and max ΔTm between primers
4. **Tm calculation tool** — check if the task specifies a ground-truth tool (e.g., `oligotm` with specific flags). If it does, you MUST use that tool for all Tm calculations and install it immediately. See the **Tm Calculation Reference** section below.
5. **Overhang requirements** — Gibson homology arms, Golden Gate sites, etc.
6. **Salt/DNA concentration** — for Tm calculation (defaults vary by tool)
7. **Primer length constraints** — if specified (typical: 18–25 nt binding region)

### Phase 2: Design Initial Binding Regions

```python
# Use sandbox_sh to run this
# IMPORTANT: Use the Tm tool from Phase 1 (e.g., oligotm or Tm_NN)
from Bio.Seq import Seq

template = "YOUR_TEMPLATE_SEQUENCE"

# Forward primer: take 18-25 nt from the start of the target region
# Start with 20 nt and adjust
fwd_binding = template[start:start+20]

# Reverse primer: reverse complement of 18-25 nt from the end of target region
rev_binding = str(Seq(template[end-20:end]).reverse_complement())

# Calculate Tm using the task-specified tool (see Tm Calculation Reference)
fwd_tm = calc_tm(fwd_binding)  # use your calc_tm function from Phase 1
rev_tm = calc_tm(rev_binding)

print(f"Fwd binding: {fwd_binding} (Tm={fwd_tm:.1f}°C)")
print(f"Rev binding: {rev_binding} (Tm={rev_tm:.1f}°C)")
```

**Aim for Tm values in the MIDDLE of the acceptable range**, not near the edges. This provides buffer for overhang overlap (Phase 3) and boundary interpretation ambiguity (Phase 4).

### Phase 3: Add Overhangs and Check for Template Overlap (CRITICAL)

This is the most important step. After deciding on overhang sequences, you MUST check whether they extend the effective annealing region.

```python
def find_effective_annealing(template, binding_start, binding_end, overhang, direction='forward'):
    """
    Determine the effective annealing region after adding an overhang.

    For forward primers: check if overhang matches template UPSTREAM of binding_start
    For reverse primers: work on reverse complement; check UPSTREAM of binding on revcomp

    Returns: (effective_annealing_seq, num_extra_bases)
    """
    binding_seq = template[binding_start:binding_end]

    if direction == 'forward':
        # Check how many overhang bases (from 3' end of overhang) match
        # the template immediately upstream of the binding site
        extra = 0
        for i in range(1, len(overhang) + 1):
            # overhang[-i] is the i-th base from the 3' end of the overhang
            template_pos = binding_start - i
            if template_pos < 0:
                break
            if overhang[-i].upper() == template[template_pos].upper():
                extra += 1
            else:
                break
        effective = template[binding_start - extra : binding_end]

    elif direction == 'reverse':
        # For reverse primers, the "template" is the reverse complement
        from Bio.Seq import Seq
        rc_template = str(Seq(template).reverse_complement())
        # binding on rc_template
        rc_binding_start = len(template) - binding_end
        rc_binding_end = len(template) - binding_start

        extra = 0
        for i in range(1, len(overhang) + 1):
            rc_pos = rc_binding_start - i
            if rc_pos < 0:
                break
            if overhang[-i].upper() == rc_template[rc_pos].upper():
                extra += 1
            else:
                break
        effective = rc_template[rc_binding_start - extra : rc_binding_end]

    return effective, extra
```

**After finding the effective annealing region, recalculate Tm:**

```python
effective_seq, extra_bases = find_effective_annealing(
    template, binding_start, binding_end, overhang, 'forward'
)
# Use the task-specified Tm tool (see Tm Calculation Reference)
effective_tm = calc_tm(effective_seq)

print(f"Designed binding: {binding_seq} ({len(binding_seq)} nt)")
print(f"Overhang: {overhang}")
print(f"Extra matching bases: {extra_bases}")
print(f"Effective annealing: {effective_seq} ({len(effective_seq)} nt)")
print(f"Effective Tm: {effective_tm:.1f}°C")
```

**If the effective Tm exceeds the constraint range:**
- **SHORTEN the designed binding region** from its 3' end (forward) or 5' end (reverse) to compensate
- Recalculate and verify again
- Iterate until the effective Tm is within range

### Phase 4: Handle Boundary Ambiguity (Insert Sequences)

When designing primers for sequence insertion, the boundary between "overhang" and "binding region" can be ambiguous. Different tools or evaluators may interpret the boundary differently.

**Procedure:**

```python
# Test Tm under multiple boundary interpretations
# If the designed binding region is template[start:end], test:
for offset in range(-5, 6):  # ±5 bases
    test_start = max(0, start + offset) if direction == 'forward' else start
    test_end = end if direction == 'forward' else min(len(template), end + offset)
    test_region = template[test_start:test_end]
    if len(test_region) < 10:
        continue
    tm = calc_tm(test_region)  # use your task-specified Tm function
    in_range = "✓" if tm_min <= tm <= tm_max else "✗"
    print(f"  offset={offset:+d}: {test_region} ({len(test_region)} nt) Tm={tm:.1f}°C {in_range}")
```

**Goal:** Find primer designs where Tm is within the acceptable range for ALL reasonable boundary interpretations (±3–5 bases). This means aiming for Tm values **well within the middle** of the range.

### Phase 5: Assemble Full Primers, Tm Verification, and Dual-Tm Strategy

After constructing the full primer sequences (overhang + binding region), run verification.

#### Tm Verification Strategy

**First, check the task instructions.** Many tasks explicitly state how Tm should be computed (e.g., "computed with respect to only the part of the primers that anneal to the input template"). If the task is explicit, follow it exactly — no dual interpretation needed.

**When the task is ambiguous** about whether Tm refers to binding-region-only or the effective annealing region (including overhang overlap), design primers that satisfy the constraint under **BOTH interpretations** to be safe:

1. **Binding-region-only Tm:** Tm of just the designed binding region
2. **Effective-annealing Tm:** Tm of binding region + any overhang bases that match the template (Phase 3)

```python
# Use the Tm tool specified by the task (oligotm, Tm_NN, etc.)
# This example uses oligotm — adapt to your task's tool
binding_region = "ATGCCTGAACCCTTAG"  # designed binding portion only
effective_region = "GGTAATGCCTGAACCCTTAG"  # including overhang overlap

tm_binding_only = calc_tm(binding_region)   # use your task's Tm function
tm_effective = calc_tm(effective_region)

print(f"Binding-region-only Tm: {tm_binding_only:.1f}°C")
print(f"Effective-annealing Tm: {tm_effective:.1f}°C")

# If task is ambiguous, BOTH must fall within the constraint range
tm_min, tm_max = 58.0, 72.0
binding_ok = tm_min <= tm_binding_only <= tm_max
effective_ok = tm_min <= tm_effective <= tm_max
print(f"Binding-only in range: {'PASS' if binding_ok else 'FAIL'}")
print(f"Effective in range:    {'PASS' if effective_ok else 'FAIL'}")
```

**If only one interpretation passes:**
- If binding-only Tm is too low but effective Tm is in range → extend the binding region
- If effective Tm is too high but binding-only is in range → shorten the binding region
- The sweet spot is usually a binding region Tm in the **lower-middle** of the range

See the **Verification Checklist** section below. Run the full verification script before submitting.

### Phase 6: MANDATORY — End-to-End Assembly Simulation

> **⚠️ CRITICAL: This phase is NON-NEGOTIABLE. You MUST complete it BEFORE writing any output file. Never submit primers without a passing assembly simulation.**

Off-by-one and off-by-two errors in junction positioning and primer coordinate reasoning are **extremely common** and are **nearly impossible to catch without computational verification**. This phase exists because informal or mental reasoning about enzymatic processes and sequence positions is unreliable. You MUST write and run code to simulate the complete molecular workflow.

#### Step 6a: Partition Consistency Check

After choosing junction positions on the target/output sequence, verify computationally that consecutive junctions tile the sequence without gaps or overlaps.

**For Golden Gate Assembly**, check that each pair of consecutive junction positions satisfies:

```
junction_start[i] + 4 + fragment_body_length[i] == junction_start[i+1]
```

(cyclically, so the last fragment wraps back to the first junction for circular constructs)

```python
# Example: Verify junction partition consistency
junctions = [0, 150, 300, 450]  # 4-nt junction overhang start positions
target_len = 600  # total target sequence length (for circular constructs)

print("=== PARTITION CONSISTENCY CHECK ===")
for i in range(len(junctions)):
    j_start = junctions[i]
    j_next = junctions[(i + 1) % len(junctions)]
    if i + 1 < len(junctions):
        fragment_span = j_next - j_start
    else:
        fragment_span = target_len - j_start + junctions[0]  # wrap-around for circular

    overhang_at_start = target_seq[j_start:j_start+4]
    overhang_at_end = target_seq[j_next:j_next+4] if j_next + 4 <= target_len else target_seq[j_next:] + target_seq[:4-(target_len-j_next)]

    print(f"Fragment {i+1}: junction[{i}]={j_start} -> junction[{(i+1)%len(junctions)}]={j_next}")
    print(f"  Span: {fragment_span} nt")
    print(f"  Start overhang (4nt): {overhang_at_start}")
    print(f"  End overhang (4nt):   {overhang_at_end}")

# Verify no gaps or overlaps: the fragments must tile exactly
total_covered = sum(
    (junctions[(i+1) % len(junctions)] - junctions[i]) % target_len
    for i in range(len(junctions))
)
assert total_covered == target_len, f"PARTITION ERROR: fragments cover {total_covered} nt but target is {target_len} nt"
print(f"\n✓ Partition covers exactly {target_len} nt — no gaps, no overlaps")
```

Print every junction position and fragment span so any misalignment is immediately obvious. **Do not proceed if this check fails.**

#### Step 6b: Full Assembly Simulation in Python (MANDATORY)

Write and run a complete Python script that simulates the entire molecular workflow. The script in `scripts/simulate_assembly.py` provides a reusable template — adapt it to your specific assembly. The simulation must perform these steps in order:

**1. Simulate PCR for each primer pair:**
- Find where each primer's 3'-end binding region matches the template
- Extract the full PCR product: forward primer sequence (5'→3') + template region between primers + reverse complement of reverse primer sequence

**2. Simulate Type IIS enzyme digestion (Golden Gate):**
- Locate the enzyme recognition site (e.g., BsaI: `GGTCTC`) on each strand of each PCR product
- Apply the correct cut offsets (BsaI cuts 1 nt downstream on the recognition strand, 5 nt upstream on the complementary strand, leaving 4-nt sticky ends)
- Extract the digested fragment with its 4-nt overhangs on each end

**3. Assemble fragments by matching sticky ends:**
- For each fragment, identify the 4-nt overhang at each end
- Match complementary overhangs between fragments
- Concatenate fragments in the correct order

**4. Compare assembled sequence to target:**
- Align the assembled sequence character-by-character against the expected target sequence
- Report the result

```python
# === CORE ASSEMBLY SIMULATION LOGIC ===
# Adapt this to your specific assembly; see scripts/simulate_assembly.py for full template

from Bio.Seq import Seq

def simulate_pcr(fwd_primer, rev_primer, template):
    """
    Simulate PCR: find where primers bind, extract the product.
    fwd_primer and rev_primer are full primer sequences (5'->3').
    Returns the double-stranded PCR product (top strand, 5'->3').
    """
    template = template.upper()
    fwd = fwd_primer.upper()
    rev = rev_primer.upper()

    # Find forward primer binding (3' end of primer matches template)
    # Try progressively shorter 3' suffixes until we find a match
    fwd_bind_pos = None
    for bind_len in range(len(fwd), 14, -1):  # at least 15 nt binding
        suffix = fwd[-bind_len:]
        pos = template.find(suffix)
        if pos >= 0:
            fwd_bind_pos = pos
            fwd_bind_len = bind_len
            break

    # Find reverse primer binding (3' end of rev primer matches reverse complement of template)
    rc_template = str(Seq(template).reverse_complement())
    rev_bind_pos = None
    for bind_len in range(len(rev), 14, -1):
        suffix = rev[-bind_len:]
        pos = rc_template.find(suffix)
        if pos >= 0:
            # Convert rc_template position back to template coordinates
            rev_bind_pos = len(template) - pos  # this is the 3' end on the template
            rev_bind_len = bind_len
            break

    if fwd_bind_pos is None or rev_bind_pos is None:
        raise ValueError(f"Could not find primer binding sites on template")

    # PCR product = fwd_primer + template between primers + rc(rev_primer)
    # Top strand: fwd_primer_full + template[fwd_bind_pos+fwd_bind_len : rev_bind_pos-rev_bind_len] + rc(rev_primer_full)
    # Simplified: just take template region and prepend/append primer overhangs
    product = fwd + template[fwd_bind_pos + fwd_bind_len : len(template) - (len(rc_template) - rc_template.find(rev[-rev_bind_len:]) - rev_bind_len)] + str(Seq(rev).reverse_complement())

    return product

def simulate_bsai_digestion(pcr_product):
    """
    Find BsaI sites (GGTCTC) and cut: 1 nt downstream on top strand,
    5 nt upstream on bottom strand -> 4-nt 5' overhang.
    Returns: (left_overhang_4nt, fragment_body, right_overhang_4nt)
    """
    seq = pcr_product.upper()
    rc = str(Seq(seq).reverse_complement())

    # Find GGTCTC on top strand (cuts downstream)
    # Find GGTCTC on bottom strand = GAGACC on top strand (cuts upstream on top strand)

    fwd_site = seq.find("GGTCTC")
    rev_site_rc = rc.find("GGTCTC")  # BsaI on bottom strand

    # Forward BsaI: GGTCTC(N1)↓ ... cut at position fwd_site + 7 on top strand
    # The 4-nt overhang starts at fwd_site + 7
    if fwd_site >= 0:
        cut_top_left = fwd_site + 7      # 1 nt after recognition site end
        cut_bottom_left = fwd_site + 11  # 5 nt after recognition site end on bottom = top pos
        left_overhang = seq[cut_top_left:cut_top_left + 4]

    # Reverse BsaI (on bottom strand): similar logic for right end
    if rev_site_rc >= 0:
        # Convert to top-strand coordinates
        cut_top_right = len(seq) - rev_site_rc - 7
        right_overhang = seq[cut_top_right:cut_top_right + 4]

    # Extract the digested fragment between the two cuts
    fragment = seq[cut_top_left:cut_top_right + 4]

    return left_overhang, fragment, right_overhang

def assemble_and_compare(fragments_with_overhangs, target_sequence):
    """
    Assemble fragments by matching complementary 4-nt sticky ends.
    Compare assembled sequence against target.
    """
    # Sort/order fragments by matching overhangs
    assembled = fragments_with_overhangs[0]['body']

    for i in range(1, len(fragments_with_overhangs)):
        prev_right = fragments_with_overhangs[i-1]['right_overhang']
        curr_left = fragments_with_overhangs[i]['left_overhang']

        if prev_right != curr_left:
            print(f"ERROR: Overhang mismatch between fragment {i} and {i+1}!")
            print(f"  Fragment {i} right overhang: {prev_right}")
            print(f"  Fragment {i+1} left overhang: {curr_left}")
            return False

        # Append (overhangs overlap, so don't double-count)
        assembled += fragments_with_overhangs[i]['body']

    # Character-by-character comparison
    target = target_sequence.upper()
    assembled = assembled.upper()

    if len(assembled) != len(target):
        print(f"LENGTH MISMATCH: assembled={len(assembled)}, target={len(target)}")

    mismatches = []
    for pos in range(min(len(assembled), len(target))):
        if assembled[pos] != target[pos]:
            mismatches.append((pos, target[pos], assembled[pos]))

    if not mismatches and len(assembled) == len(target):
        print("✓ ASSEMBLY SIMULATION PASSED — assembled sequence matches target exactly")
        return True
    else:
        print(f"✗ ASSEMBLY SIMULATION FAILED — {len(mismatches)} mismatches found:")
        for pos, expected, actual in mismatches[:20]:  # show first 20
            print(f"  Position {pos}: expected '{expected}', got '{actual}'")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more mismatches")
        return False
```

> **IMPORTANT:** The code above is a **template** showing the logic. You MUST adapt it to your specific assembly (number of fragments, enzyme used, linear vs. circular construct, etc.). The full reusable script is at `scripts/simulate_assembly.py`.

#### Step 6c: Interpret Simulation Results

- **If PASS:** Proceed to write the output file (Phase 7).
- **If FAIL:**
  1. Identify which junction or primer caused the mismatch based on the position reported
  2. Go back to the relevant Phase (2, 3, or the partition check in 6a) and fix the error
  3. Re-run the entire simulation from the beginning
  4. Do NOT submit until the simulation passes

#### Key Principle

> **Never rely on informal or mental reasoning about enzymatic processes or sequence positions.** Always write and run code to simulate the complete molecular workflow. Off-by-one and off-by-two errors in positional reasoning are extremely common and nearly impossible to catch without computational verification. If you think "I'm pretty sure this is right" — you're wrong. Run the simulation.

### Phase 7: Write Output File

Only after the assembly simulation in Phase 6 passes, write the final output file (FASTA, CSV, or whatever format the task requires).

---

## Tm Calculation Reference

### Step 1: Check the task for a specified Tm tool (CRITICAL)

Many tasks specify an exact Tm calculation tool as ground truth. **You MUST use whatever tool the task specifies.** Using a different tool (even a "better" one) will produce different values and cause verification failures.

Common specification: *"The output of primer3's `oligotm` tool should be considered the ground truth for melting temperatures with the following flags: `-tp 1 -sc 1 -mv 50 -dv 2 -n 0.8 -d 500`"*

### Using `oligotm` (primer3) — use when task specifies it

```bash
# Install primer3 (includes oligotm)
apt-get update -qq && apt-get install -y -qq primer3 2>/dev/null
# Verify it's available
which oligotm
```

```python
import subprocess

def calc_tm_oligotm(seq, mv=50, dv=2, n=0.8, d=500, tp=1, sc=1):
    """Calculate Tm using primer3's oligotm tool.
    Adjust flags to match whatever the task specifies."""
    result = subprocess.run(
        ["oligotm", "-mv", str(mv), "-dv", str(dv), "-n", str(n),
         "-d", str(d), "-tp", str(tp), "-sc", str(sc), seq],
        text=True, capture_output=True
    )
    return float(result.stdout.strip())

# Example usage
tm = calc_tm_oligotm("ATGCCTGAACCC")
print(f"Tm = {tm:.2f}°C")
```

**⚠️ `oligotm` and Biopython's `Tm_NN` give DIFFERENT results for the same sequence.** A primer pair that passes ΔTm ≤ 5°C under one method may fail under the other. Always verify with the SAME tool the verifier uses.

### Using Biopython's `Tm_NN` — fallback when no tool is specified

```python
from Bio.Seq import Seq
from Bio.SeqUtils.MeltingTemp import Tm_NN

# Standard conditions (adjust if task specifies different values)
tm = Tm_NN(
    Seq("ATGCCTGAACCC"),
    Na=50,         # mM sodium — adjust if specified
    dnac1=250,     # nM DNA concentration — adjust if specified
)
print(f"Tm = {tm:.1f}°C")
```

**Common pitfalls:**
- ⛔ Do NOT use `Tm_Wallace` (the 2+4 rule) — it's only for rough estimates of short oligos
- ⛔ Do NOT use `Tm_GC` for primers — it's for long DNA duplexes
- ⛔ Do NOT estimate Tm mentally — always compute with code
- ⛔ Do NOT use Biopython's `Tm_NN` when the task specifies `oligotm` — they give different values
- ✅ Use the Tm tool and parameters specified in the task instructions
- ✅ Always verify your final Tm values using the exact same tool and flags the task specifies

---

## Golden Gate-Specific Considerations

For Golden Gate Assembly with Type IIS enzymes (BsaI, BpiI, etc.):

1. **Add the enzyme recognition site + spacer to the overhang:**
   - BsaI: `GGTCTC(N1)↓` — recognize GGTCTC, cut 1 nt downstream
   - BpiI: `GAAGAC(N2)↓` — recognize GAAGAC, cut 2 nt downstream

2. **The 4-nt fusion site** (sticky end) is part of your overhang — it must be complementary between adjacent fragments

3. **Check that the binding region does NOT contain the enzyme recognition site:**
   ```python
   from Bio.Seq import Seq
   binding = "YOUR_BINDING_REGION"
   enzyme_sites = ["GGTCTC", "GAGACC",  # BsaI + reverse complement
                    "GAAGAC", "GTCTTC"]  # BpiI + reverse complement
   for site in enzyme_sites:
       if site in binding.upper():
           print(f"WARNING: {site} found in binding region — must redesign!")
   ```

4. **Overhang-template overlap still applies!** The enzyme site + spacer + fusion site overhang can match the template. Always run the overlap check.

---

## Error Handling

### Tm too high after overhang overlap
- Shorten the designed binding region from the end opposite the overhang
- For forward primers: trim from the 3' end
- For reverse primers: trim from the 5' end
- Recalculate effective Tm after shortening

### Tm too low
- Extend the binding region (add more template-matching bases)
- Verify extension doesn't create secondary structure issues

### ΔTm between forward and reverse too large
- Adjust the shorter-Tm primer's binding region first (extend it)
- If that's not possible, shorten the higher-Tm primer
- Iterate until both Tm values are within range AND ΔTm is within the limit

### Primer has secondary structure (hairpins, self-dimers)
```python
from Bio.SeqUtils.MeltingTemp import Tm_NN
from Bio.Seq import Seq

primer = "YOUR_FULL_PRIMER"
# Check for self-complementarity
seq = Seq(primer)
rc = seq.reverse_complement()
# Look for runs of complementarity ≥ 4 nt at the 3' end
# (3' end dimers are most problematic for PCR)
```

### Dependencies not available
```bash
# Install primer3 (for oligotm)
apt-get update -qq && apt-get install -y -qq primer3 2>/dev/null
# Install Biopython (for sequence manipulation)
pip install biopython
```

### Assembly simulation fails
- Check the mismatch positions reported by the simulation
- If mismatches cluster at a junction boundary: the junction overhang position is wrong (off-by-one or off-by-two)
- If mismatches span an entire fragment: the wrong template region is being amplified (primer binding coordinates are wrong)
- Fix the root cause, then re-run the ENTIRE simulation from scratch — do not patch

---

## Verification Checklist (MANDATORY — Run Before Submitting)

**Every single item below must pass before you write the output file. No exceptions.**

- [ ] **Run end-to-end assembly simulation** — Simulate PCR → enzyme digestion → ligation → compare assembled sequence to target (Phase 6b). Result must be PASS.
- [ ] **Verify partition consistency** — Consecutive junctions tile the target sequence with no gaps and no overlaps (Phase 6a).
- [ ] **Verify all Tm values using the task-specified tool** — If the task specifies `oligotm` or another tool, verify EVERY primer's Tm using that exact tool and flags. Do NOT rely on a different tool's values.
- [ ] **Overhang-template overlap checked** — Every primer's overhang has been tested for template match extension (Phase 3).
- [ ] **Effective Tm in range for all primers** — After accounting for overhang overlap.
- [ ] **ΔTm within limit** — Between forward and reverse primers of each pair.
- [ ] **Boundary robustness verified** — Tm holds under ±3–5 base boundary shifts (Phase 4).
- [ ] **No enzyme sites in binding regions** — For Golden Gate assemblies.
- [ ] **Check output file format** — Correct FASTA headers (if applicable), no blank lines, correct primer count, sequences in the expected orientation.
- [ ] **DO NOT submit primers until the assembly simulation confirms the assembled sequence matches the target exactly.**

Adapt the verification logic below to use your task's Tm tool (see **Tm Calculation Reference**):

```python
import subprocess
from Bio.Seq import Seq

# Define calc_tm using the task-specified tool
# Example with oligotm (adapt flags to match your task):
def calc_tm(seq):
    result = subprocess.run(
        ["oligotm", "-mv", "50", "-dv", "2", "-n", "0.8",
         "-d", "500", "-tp", "1", "-sc", "1", seq],
        text=True, capture_output=True
    )
    return float(result.stdout.strip())

def verify_primer_pair(template, fwd_annealing, rev_annealing,
                       tm_min, tm_max, delta_tm_max):
    """Verify Tm constraints for a primer pair's annealing regions."""
    fwd_tm = calc_tm(fwd_annealing)
    rev_tm = calc_tm(rev_annealing)
    delta = abs(fwd_tm - rev_tm)

    print(f"Fwd annealing: {fwd_annealing} ({len(fwd_annealing)} nt) Tm={fwd_tm:.2f}°C")
    print(f"Rev annealing: {rev_annealing} ({len(rev_annealing)} nt) Tm={rev_tm:.2f}°C")
    print(f"ΔTm = {delta:.2f}°C")

    ok = True
    if not (tm_min <= fwd_tm <= tm_max):
        print(f"FAIL: Fwd Tm {fwd_tm:.2f} outside [{tm_min}, {tm_max}]")
        ok = False
    if not (tm_min <= rev_tm <= tm_max):
        print(f"FAIL: Rev Tm {rev_tm:.2f} outside [{tm_min}, {tm_max}]")
        ok = False
    if delta > delta_tm_max:
        print(f"FAIL: ΔTm {delta:.2f} > {delta_tm_max}")
        ok = False
    if ok:
        print("ALL Tm CHECKS PASSED")
    return ok
```

### Quick Verification One-Liner

```python
# Quick check using task-specified Tm tool
annealing_seq = "ATGCCTGAACCCTTAG"
tm = calc_tm(annealing_seq)  # uses your calc_tm from above
print(f"Annealing: {annealing_seq} ({len(annealing_seq)} nt), Tm = {tm:.2f}°C")
```

---

## Complete Example: Gibson Assembly Primer Design

**Task:** Design primers to amplify GeneX and add Gibson Assembly overlaps with a vector. Tm must be 58–72°C, ΔTm ≤ 5°C.

```python
from Bio.Seq import Seq
import subprocess

# Define calc_tm using the task-specified tool
# (Use oligotm if task specifies it, otherwise Tm_NN — see Tm Calculation Reference)
def calc_tm(seq):
    result = subprocess.run(
        ["oligotm", "-mv", "50", "-dv", "2", "-n", "0.8",
         "-d", "500", "-tp", "1", "-sc", "1", seq],
        text=True, capture_output=True
    )
    return float(result.stdout.strip())

# Template containing GeneX
template = "AGTCCGATCGAATTCATGAAAGCGATTGTCGAACTTGACCTGCAGCCCGGGTTATAAGGATCC"
gene_start = 16  # ATG start codon

# Vector overlaps (from task specification)
fwd_vector_overlap = "GCTAGCGAATTC"  # 12 nt upstream vector homology

# Step 1: Design initial binding regions (aim for middle of 58-72°C → ~65°C)
for length in range(18, 26):
    fwd_bind = template[gene_start:gene_start+length]
    tm = calc_tm(fwd_bind)
    print(f"Fwd {length}nt: {fwd_bind} Tm={tm:.1f}°C")

# Pick length that gives ~65°C, e.g., 20 nt
fwd_bind = template[gene_start:gene_start+20]

# Step 2: Check overhang-template overlap for forward primer
extra = 0
for i in range(1, len(fwd_vector_overlap) + 1):
    pos = gene_start - i
    if pos < 0: break
    if fwd_vector_overlap[-i].upper() == template[pos].upper():
        extra += 1
    else: break

effective = template[gene_start - extra : gene_start + 20]
effective_tm = calc_tm(effective)
print(f"Effective annealing: {effective} Tm={effective_tm:.1f}°C")

# Step 3: If effective Tm > 72°C, shorten binding region
if effective_tm > 72:
    for length in range(19, 14, -1):
        effective = template[gene_start - extra : gene_start + length]
        tm = calc_tm(effective)
        if tm <= 72:
            print(f"Shortened to {length}nt binding → effective Tm={tm:.1f}°C ✓")
            fwd_bind = template[gene_start:gene_start+length]
            break

# Step 4-7: Repeat for reverse, verify ΔTm, run assembly simulation, write output
```

---

## Key Rules Summary

1. **ALWAYS check overhang-template overlap** — the effective annealing region may be longer than the designed binding region
2. **ALWAYS compute Tm on the EFFECTIVE annealing region** — not just the designed binding portion
3. **Use the Tm tool the task specifies** (e.g., `oligotm` from primer3) — install it, use it for ALL Tm calculations, and verify final values with it. Fall back to Biopython's `Tm_NN` only when no tool is specified. Never estimate by hand.
4. **Aim for Tm in the MIDDLE of the range** — this provides buffer for overlap effects and boundary ambiguity
5. **Test boundary robustness for inserts** — verify Tm holds under ±3–5 base boundary shifts
6. **Run the full verification checklist before submitting** — no exceptions
7. **If effective Tm is out of range, SHORTEN the binding region** — don't just report the wrong Tm
8. **For Golden Gate: check for enzyme sites in the binding region** — BsaI/BpiI sites will destroy the construct
9. **ALWAYS run a computational end-to-end assembly simulation before submitting** — Never submit primers based only on manual or mental verification. Simulate PCR, enzyme digestion, fragment ligation, and character-by-character comparison against the target sequence. Off-by-one and off-by-two errors in junction positioning are extremely common and invisible without code-based verification.
10. **Use the exact Tm calculation tool specified in the task** — If the task says to use `oligotm` with specific flags, use exactly that. Do NOT substitute Biopython's `Tm_NN` or any other method. Different tools give different values, and the verifier uses the task-specified tool.
