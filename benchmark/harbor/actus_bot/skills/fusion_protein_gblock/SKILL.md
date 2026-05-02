---
name: fusion_protein_gblock
description: How to design fusion protein gBlock DNA sequences — identifying protein components (epitope tags, fluorescent proteins, binding tags), retrieving sequences from PDB, assembling with linkers, and codon-optimizing with GC content constraints.
version: 1.0.0
required_tools: ['sandbox_sh']
---

# Fusion Protein gBlock Design

Design synthetic DNA (gBlock) sequences encoding multi-component fusion proteins. This skill covers component identification, sequence retrieval, assembly, and codon optimization.

## When to Use This Skill

- A task asks you to design a **gBlock** or **synthetic gene** encoding a fusion protein.
- The task involves combining multiple protein domains (tags, fluorescent proteins, enzymes, binding domains).
- The task mentions **FRET**, **epitope tags**, **protein binding**, or **codon optimization**.
- The task provides PDB IDs, antibody sequences, or molecule structures to identify protein components.

## Overview: Fusion Protein Design Workflow

```
Phase 1: Read task inputs (antibody FASTA, PDB IDs, plasmid files, molecule info)
Phase 2: Identify each component (antibody binder, FRET donor/acceptor, binding tags, enzymes)
Phase 3: Retrieve protein sequences from PDB API
Phase 4: Assemble fusion protein with linkers
Phase 5: Reverse-translate and codon-optimize the DNA sequence
Phase 6: Verify all constraints (GC content, length, correct sequences)
```

---

## Phase 1: Identify Protein Components

### Antibody Binder Identification (CRITICAL — #1 Failure Mode)

When the task provides an antibody FASTA and asks for an "antibody binder", you must identify **what epitope the antibody recognizes**, then use that epitope tag as the binder.

**Common epitope tags and their antibodies:**

| Epitope Tag | Sequence | Recognized By | Notes |
|-------------|----------|---------------|-------|
| **FLAG** | `DYKDDDDK` | Anti-FLAG M2 antibody | Most common; look for M2 in antibody name/title |
| **HA** | `YPYDVPDYA` | Anti-HA antibody | From influenza hemagglutinin |
| **Myc** | `EQKLISEEDL` | Anti-Myc 9E10 antibody | c-Myc epitope |
| **V5** | `GKPIPNPLLGLD` | Anti-V5 antibody | From simian virus 5 |
| **His** | `HHHHHH` | Anti-His antibody | 6xHis tag |

**How to identify the antibody:**

```python
# Method 1: BLAST antibody sequence against PDB (most reliable)
from rcsbapi.search import SeqSimilarityQuery
from rcsbapi.data import DataQuery

# Read the antibody FASTA
from Bio import SeqIO
for rec in SeqIO.parse("antibody.fasta", "fasta"):
    hit = next(iter(
        SeqSimilarityQuery(str(rec.seq), sequence_type="protein")(
            "polymer_entity", rows=1
        )
    ), None)
    if hit:
        entry_id = hit.split("_", 1)[0]
        q = DataQuery(input_type="entries", input_ids=[entry_id],
                      return_data_list=["struct.title"])
        res = q.exec()
        title = res["data"]["entries"][0]["struct"]["title"]
        print(f"{rec.id}: {title}")
        # Look for keywords: "FLAG", "anti-FLAG", "M2", "HA", "Myc", etc.
```

**Key rule**: The "antibody binder" is the **epitope tag** (e.g., FLAG = `DYKDDDDK`), NOT the target protein of the antibody. If the antibody is anti-FLAG, use the FLAG peptide sequence as the binder component.

### Molecule Binding Protein Identification

When the task specifies a molecule (often via SMILES string) and asks for a "molecule binder":

| Molecule | Common Name | Binding Protein | PDB Example |
|----------|-------------|-----------------|-------------|
| O6-Benzylguanine | BG | **SNAP-tag** | 3KZY |
| O2-Benzylcytosine | BC | **CLIP-tag** | — |
| Chloroalkane ligand | HaloTag ligand | **HaloTag** | — |

**To identify from SMILES:**

```python
# Query PubChem to identify the molecule
import urllib.parse, requests
base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
smiles = "Nc3nc(OCc1ccccc1)c2nc[nH]c2n3"  # example
cids = requests.get(
    f"{base}/compound/smiles/{urllib.parse.quote(smiles)}/cids/TXT"
).text.strip().split()
for cid in cids:
    synonyms = requests.get(f"{base}/compound/cid/{cid}/synonyms/TXT").text
    print(synonyms[:500])
    # Look for: "benzylguanine" → SNAP-tag, "benzylcytosine" → CLIP-tag
```

**Shortcut**: If the SMILES contains a guanine ring with a benzyl group, it's benzylguanine → use SNAP-tag.

### FRET Donor/Acceptor Selection

For FRET tasks, you need a donor (matched by excitation wavelength) and acceptor (matched by emission wavelength) from the available PDB IDs.

**Use FPbase API to find matching fluorescent proteins:**

```python
import requests, csv, time

def fpbase_peaks(pdb_id):
    """Get excitation/emission peaks for a fluorescent protein by PDB ID."""
    try:
        r = requests.get(
            f"https://www.fpbase.org/api/proteins/?pdb__contains={pdb_id}&format=json",
            timeout=15)
        arr = r.json()
        if not isinstance(arr, list) or not arr:
            return None, None
        state = arr[0]["states"][0]
        return state["ex_max"], state["em_max"]
    except:
        return None, None

# Query all PDB IDs and find donor/acceptor
with open("pdb_ids.txt") as f:
    pdb_ids = [line.strip() for line in f if line.strip()]

for pid in pdb_ids:
    ex, em = fpbase_peaks(pid)
    if ex or em:
        print(f"{pid}: excitation={ex}nm, emission={em}nm")
    time.sleep(1)  # rate limit

# Match: donor excitation == filter excitation, acceptor emission == filter emission
```

---

## Phase 2: Retrieve Protein Sequences

### From PDB API

```python
import requests

def get_pdb_sequence(pdb_id):
    """Get protein sequence from RCSB PDB."""
    r = requests.get(f"https://www.rcsb.org/fasta/entry/{pdb_id}/download", timeout=20)
    lines = [l.strip() for l in r.text.split("\n") if l.strip()]
    # Return first chain sequence (skip header line starting with >)
    sequences = [l for l in lines if not l.startswith(">")]
    return sequences[0] if sequences else None

def get_pdb_title(pdb_id):
    """Get structure title from RCSB PDB."""
    r = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=15)
    return r.json()["struct"]["title"]
```

### From GenBank Plasmid Files

```python
from Bio import SeqIO

for record in SeqIO.parse("plasmid.gb", "genbank"):
    for feat in record.features:
        if feat.type == "CDS":
            protein_seq = feat.qualifiers["translation"][0]
            gene_name = feat.qualifiers.get("gene", ["unknown"])[0]
            print(f"{gene_name}: {protein_seq}")
```

---

## Phase 3: Assemble the Fusion Protein

### Remove N-terminal Methionine

Most tasks require removing the initiator methionine from each protein component:

```python
def strip_met(seq):
    return seq[1:] if seq.startswith("M") else seq
```

### GS Linkers

Use glycine-serine (GS) linkers between components. Common patterns:
- `(GGGGS)` x 2-3 repeats = 10-15 amino acids (most common)
- Total linker length must match task constraints (typically 5-20 aa)

```python
gs_linker = "GGGGS" * 3  # 15 aa — adjust repeats to meet length constraints
```

### Assembly Order

Follow the task-specified order exactly. Common FRET fusion protein order:

```
[antibody binder] - [linker] - [FRET donor] - [linker] - [target protein] - [linker] - [FRET acceptor] - [linker] - [molecule binder]
```

```python
flag = "DYKDDDDK"
donor = strip_met(donor_seq)
dhfr = strip_met(dhfr_seq)
acceptor = strip_met(acceptor_seq)
snap = strip_met(snap_seq)
linker = "GGGGS" * 3

fusion = f"{flag}{linker}{donor}{linker}{dhfr}{linker}{acceptor}{linker}{snap}"
```

---

## Phase 4: Reverse-Translate and Codon-Optimize

Use **DnaChisel** for codon optimization with GC content constraints:

```python
pip install dnachisel biopython
```

```python
from dnachisel import (
    DnaOptimizationProblem, CodonOptimize,
    EnforceGCContent, EnforceTranslation,
)
from dnachisel.biotools import reverse_translate

# Reverse-translate protein to DNA
dna = reverse_translate(fusion, randomize_codons=False)

# Optimize with constraints
problem = DnaOptimizationProblem(
    sequence=dna,
    constraints=[
        EnforceTranslation(translation=fusion),
        EnforceGCContent(mini=0.30, maxi=0.70, window=50),
    ],
    objectives=[
        CodonOptimize(species="e_coli"),
    ],
)
problem.resolve_constraints()
problem.optimize()

# Write output
with open("gblock.txt", "w") as f:
    f.write(str(problem.sequence))
```

---

## Phase 5: Verify All Constraints

```python
from Bio.Seq import Seq

gblock = open("gblock.txt").read().strip()

# 1. DNA only
assert all(c in "ATCGatcg" for c in gblock), "Non-DNA characters found"

# 2. Length
assert len(gblock) <= 3000, f"Too long: {len(gblock)} nt"

# 3. Translate and check protein order
protein = str(Seq(gblock).translate())
assert flag in protein, "FLAG tag not found in translation"

# 4. GC content in 50-nt windows
for i in range(0, len(gblock) - 49):
    window = gblock[i:i+50].lower()
    gc = sum(c in "gc" for c in window)
    gc_pct = gc / 50
    assert 0.30 <= gc_pct <= 0.70, f"GC={gc_pct:.0%} at position {i}"

# 5. No start/stop codons in output (task-dependent)
print(f"Length: {len(gblock)} nt")
print(f"Protein: {len(protein)} aa")
print("All checks passed")
```

---

## Dependencies

```bash
pip install biopython dnachisel pandas requests rcsb-api
```

**Common pitfall**: `rdkit-pypi` does NOT exist on PyPI. If you need RDKit, install via `pip install rdkit` or `conda install -c conda-forge rdkit`. But for identifying molecules from SMILES, PubChem API (shown above) is usually sufficient.

## Rules

1. **ALWAYS identify the antibody by BLASTing its sequence against PDB** — don't guess from PDB IDs alone
2. **The "antibody binder" is the epitope tag** (e.g., FLAG = DYKDDDDK), NOT some other protein
3. **Remove N-terminal methionine** from every protein component (unless the task says otherwise)
4. **Use FPbase API** to match fluorescent proteins to excitation/emission wavelengths
5. **Use DnaChisel** for codon optimization with GC content constraints — don't manually optimize codons
6. **Follow the exact component order** specified in the task
7. **Check that linkers are ONLY between components** — no linkers at N/C termini of the fusion
8. **Verify the final translation** matches all expected protein sequences before writing output
