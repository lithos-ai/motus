---
name: git_push_deploy_setup
description: How to set up a robust git push-to-deploy pipeline with redundant safe.directory configuration and correct ownership handling to survive package reinstallation and multi-user access.
version: 1.0.0
required_tools: ['sandbox_sh']
---

# Git Push-to-Deploy Setup

Procedural guide for setting up a git bare repository with a post-receive hook that deploys to a web root on push. Addresses two critical failure patterns: **safe.directory config being wiped by package reinstallation**, and **ownership mismatches between users**.

## When to Use This Skill

- You need to set up a git push-to-deploy pipeline (push to a bare repo → content auto-deploys to a web root).
- You are configuring a post-receive hook that checks out files to a web-accessible directory.
- You encounter `fatal: unsafe repository` errors after package updates or user switches.
- Git commands fail with ownership/permission errors in a multi-user environment.
- You need a deploy pipeline that survives system package reinstallation (e.g., `apt-get install git` wiping `/etc/gitconfig`).

---

## ⚠️ Critical Rules

> **NEVER rely on a single level of safe.directory configuration.**
>
> Package reinstallation (e.g., `apt-get install --reinstall git`) wipes `/etc/gitconfig`, destroying system-level safe.directory entries. Always configure safe.directory at **multiple levels** AND embed it directly in hook scripts.

> **NEVER run git commands as root on a repo owned by another user.**
>
> Use `su - <user> -c '...'` to run git commands as the repo owner. Running as root triggers safe.directory checks that may fail unpredictably.

> **After ALL setup is complete, always run a final `chown -R` to fix ownership drift.**
>
> Setup steps executed as root or other users may leave files with wrong ownership. A final `chown` sweep fixes this.

> **After verifying the pipeline works, clean up ONLY the temporary test clone.**
>
> Do NOT remove deployed content from the web root, do NOT delete git refs, do NOT reset the master branch. The deployed content will be overwritten by subsequent pushes — this is the expected behavior of the pipeline.

---

## Step-by-Step Workflow

### Phase 1 — Create the Bare Repository

Choose a single user to own the repo and all deploy operations (e.g., `git`, `www-data`, or a dedicated deploy user).

```bash
# Create the bare repo
mkdir -p /path/to/repo.git
cd /path/to/repo.git
git init --bare

# Set ownership to the chosen user
chown -R <deploy-user>:<deploy-user> /path/to/repo.git
```

### Phase 2 — Configure safe.directory at Multiple Levels (Critical)

Set safe.directory at **three levels** for redundancy. If any one level is wiped, the others still protect you.

```bash
# Level 1: System-wide (survives until package reinstall wipes /etc/gitconfig)
git config --system --add safe.directory '/path/to/repo.git'

# Level 2: Global level for each user that will interact with the repo
su - <deploy-user> -c "git config --global --add safe.directory '/path/to/repo.git'"
su - root -c "git config --global --add safe.directory '/path/to/repo.git'"

# Level 3: Repo-level config (most resilient — lives inside the repo itself)
git config -f /path/to/repo.git/config safe.directory '/path/to/repo.git'
```

Also add the web root path if the work tree is a separate directory:

```bash
git config --system --add safe.directory '/path/to/webroot'
su - <deploy-user> -c "git config --global --add safe.directory '/path/to/webroot'"
git config -f /path/to/repo.git/config safe.directory '/path/to/webroot'
```

**Verify all levels are set:**

```bash
git config --system --get-all safe.directory
su - <deploy-user> -c "git config --global --get-all safe.directory"
git config -f /path/to/repo.git/config --get-all safe.directory
```

### Phase 3 — Create the Post-Receive Hook (Self-Contained)

The hook **must** embed its own `safe.directory` override so it works even if system/global gitconfig is wiped.

```bash
cat > /path/to/repo.git/hooks/post-receive << 'HOOK'
#!/bin/bash
# Self-contained safe.directory protection — survives /etc/gitconfig wipe
git config safe.directory '*'
GIT_WORK_TREE=/path/to/webroot git checkout -f master
HOOK

chmod +x /path/to/repo.git/hooks/post-receive
```

**Why `git config safe.directory '*'` inside the hook?**
- `git config` without `--system` or `--global` writes to the repo's own config.
- This sets `safe.directory = *` in the repo config, which allows git to operate on any directory.
- It runs every time the hook fires, so even if someone resets the repo config, the next push re-establishes it.

### Phase 4 — Prepare the Web Root

```bash
# Create the web root if it doesn't exist
mkdir -p /path/to/webroot

# Set ownership to the deploy user so the hook can write to it
chown -R <deploy-user>:<deploy-user> /path/to/webroot
```

### Phase 5 — Final Ownership Sweep

After ALL configuration is done, fix any ownership drift from commands run as root:

```bash
chown -R <deploy-user>:<deploy-user> /path/to/repo.git
chown -R <deploy-user>:<deploy-user> /path/to/webroot
```

This is the **last step** before testing. Do not run further setup commands as root after this.

### Phase 6 — Test the Pipeline End-to-End

```bash
# Clone the bare repo to a temp directory
cd /tmp
git clone /path/to/repo.git test-clone
cd test-clone

# Configure git identity for the commit
git config user.email "test@test.com"
git config user.name "Test"

# Create test content and push
echo "<h1>Hello Deploy</h1>" > hello.html
git add hello.html
git commit -m "Test deploy"
git push origin master

# Verify the file was deployed to the web root
cat /path/to/webroot/hello.html
# Should output: <h1>Hello Deploy</h1>
```

### Phase 7 — Post-Verification Cleanup

After confirming the pipeline works:

```bash
# ONLY remove the temporary clone directory
rm -rf /tmp/test-clone
```

**Do NOT do any of the following:**
- ❌ `rm /path/to/webroot/hello.html` — leave deployed content in place
- ❌ `git update-ref -d refs/heads/master` — leave all git refs intact
- ❌ Reset or delete the master branch — it should point at the test commit
- ❌ Remove or modify the post-receive hook

The deployed test content will be overwritten by subsequent pushes through the pipeline — this is the correct and expected behavior.

---

## Error Handling

| Problem | Likely Cause | Fix |
|---|---|---|
| `fatal: unsafe repository ('/path/to/repo.git' is owned by someone else)` | safe.directory not set for the current user, or config was wiped | Re-run Phase 2 to set safe.directory at all three levels |
| Hook runs but files don't appear in webroot | Webroot not writable by the hook user | `chown -R <deploy-user>:<deploy-user> /path/to/webroot` |
| `fatal: unsafe repository` after `apt-get install git` | Package reinstall wiped `/etc/gitconfig` | The hook's embedded `git config safe.directory '*'` should self-heal on next push; re-run Phase 2 for interactive use |
| `error: pathspec 'master' did not match any file(s)` | First push used a different branch name (e.g., `main`) | Update the hook to use `main` instead of `master`, or push with `git push origin main:master` |
| Permission denied writing to hook file | Running as wrong user | Use `sudo` to write the hook, then `chown` it to the deploy user |
| `remote: fatal: This operation must be run in a work tree` | `GIT_WORK_TREE` not set in the hook | Ensure the hook uses `GIT_WORK_TREE=/path/to/webroot git checkout -f master` |

## Recovery: Rebuilding safe.directory After Config Wipe

If `/etc/gitconfig` was wiped (e.g., by package reinstallation), run this recovery sequence:

```bash
# Restore system-level config
git config --system --add safe.directory '/path/to/repo.git'
git config --system --add safe.directory '/path/to/webroot'

# Restore global-level config for the deploy user
su - <deploy-user> -c "git config --global --add safe.directory '/path/to/repo.git'"
su - <deploy-user> -c "git config --global --add safe.directory '/path/to/webroot'"

# Repo-level config should still be intact (lives inside the repo)
git config -f /path/to/repo.git/config --get-all safe.directory
```

---

## Quick Reference: Setup Checklist

Before considering the pipeline complete, verify:

- [ ] Bare repo created and owned by the deploy user
- [ ] safe.directory set at system, global, AND repo levels for the repo path
- [ ] safe.directory set at all levels for the webroot path (if different from repo)
- [ ] Post-receive hook contains `git config safe.directory '*'` as its first operational line
- [ ] Post-receive hook is executable (`chmod +x`)
- [ ] Webroot exists and is writable by the deploy user
- [ ] Final `chown -R` sweep completed after all setup
- [ ] End-to-end test push succeeded and content appeared in webroot
- [ ] Only the temp clone was removed; deployed content and refs left intact
