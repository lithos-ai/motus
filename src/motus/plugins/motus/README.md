# Plugin for Claude Code, Codex, and Cursor

The `/motus` skill lets you build, configure, and deploy agents from inside your coding agent.

## Install the plugin

```sh
curl -fsSL https://www.lithosai.com/motus/install.sh | sh
```

This installs the `/motus` skill into Claude Code, Codex, and Cursor.

## Usage

```
/motus                                              # guided agent creation
/motus I need a customer support agent with PII redaction
/motus deploy                                       # auto-detect and guided deploy
/motus deploy agent:my_agent                        # specify entry point
/motus deploy my-project agent:my_agent             # direct cloud deploy
```

## Alternative installation methods

### Claude Code plugin marketplace

```sh
claude plugin marketplace add lithos-ai/motus && claude plugin install motus
```

Or from within Claude Code:

```
/plugin marketplace add lithos-ai/motus
/plugin install motus
```

### Enable auto-updates

1. Open Claude Code and run `/plugin`
2. Go to **Marketplaces** tab
3. Select **motus-marketplace**
4. Choose **Enable auto-update**

### npx skills (installs for all coding agents)

```sh
npx skills add lithos-ai/motus
```

## Team setup for Claude Code

Add to your project's `.claude/settings.json` for automatic availability:

```json
{
  "extraKnownMarketplaces": {
    "motus-marketplace": {
      "source": { "source": "github", "repo": "lithos-ai/motus" }
    }
  },
  "enabledPlugins": {
    "motus@motus-marketplace": true
  }
}
```
