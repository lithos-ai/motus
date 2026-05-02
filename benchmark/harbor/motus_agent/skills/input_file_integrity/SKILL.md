---
name: input_file_integrity
description: How to preserve provided input files (databases, datasets, configs) during optimization tasks — treating them as read-only, optimizing queries/code instead of modifying the environment, and safely backing up and restoring files when modification is unavoidable.
version: 1.0.0
---

# Input File Integrity & Query Optimization

## Core Principle: Treat Provided Files as Read-Only

When working on optimization tasks (SQL queries, code, configs), **never modify the provided input files** (databases, datasets, config files) unless the task explicitly requires it.

### Database Query Optimization

For SQL/database query optimization tasks specifically:

1. **Optimize the QUERY, not the database.** Your goal is to rewrite the query to be faster. Do NOT:
   - Create indexes on the database
   - Modify the schema
   - Alter any tables
   - Change database settings/pragmas that persist

2. **The query rewrite alone must achieve the performance target.** Common optimization techniques that don't modify the database:
   - Replace correlated subqueries with CTEs (Common Table Expressions)
   - Use window functions (ROW_NUMBER, RANK, etc.) instead of nested queries
   - Restructure JOINs for better query plans
   - Use `EXPLAIN QUERY PLAN` to verify the optimizer's approach

3. **Workflow:**
   - Read the original query and schema
   - Run the original query to capture baseline timing AND reference output
   - Analyze with `EXPLAIN QUERY PLAN` to find bottlenecks
   - Rewrite the query (CTEs, window functions, better JOINs)
   - Verify correctness by diffing output against reference
   - Write the optimized query to the output file
   - Do NOT modify the database file at all

### If You Must Modify a Provided File

In rare cases where the task requires modifying a provided file:

1. **Before modifying:** Create a backup with `cp original original.bak` and record the hash: `sha256sum original`
2. **After your work is done:** Restore the original: `cp original.bak original`
3. **Verify restoration:** Run `sha256sum original` and confirm it matches the original hash
4. **Only then** delete the backup: `rm original.bak`

**CRITICAL:** Never delete a backup file without first restoring from it. The sequence is always: backup → modify → work → restore → verify → delete backup.

### Post-Task Verification Checklist

Before declaring a task complete:
- [ ] All provided input files are unmodified (or restored to original state)
- [ ] Output files are written to the correct location
- [ ] Temporary files are cleaned up
- [ ] Solution correctness is verified (diff against reference output)
