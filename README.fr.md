<!-- # Motus -->

<!-- TODO: commit logo to assets/ and replace with repo-relative or raw.githubusercontent path -->
<p align="center">
  <img alt="Motus" src="assets/motus.png" />
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
  <a href="https://github.com/lithos-ai/motus/releases"><img alt="Release" src="https://img.shields.io/github/v/release/lithos-ai/motus" /></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg" /></a>
  <a href="https://join.slack.com/t/lithosaicommunity/shared_invite/zt-3uf2cykza-P9VETbJAUx7WKjwxMk~06Q"><img alt="Slack" src="https://img.shields.io/badge/Slack-community-purple?logo=slack" /></a>
  <!-- TODO: add CI badge once URL is live -->
  <!-- <a href="https://github.com/lithos-ai/motus/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/lithos-ai/motus/ci.yml?branch=main" /></a> -->
</p>

<h3 align="center">
  Capacites plus elevees. Cout plus faible. Agents plus rapides.<br/>
  Deploiement auto-gere ou cloud, serving d'agents en une commande. Meme code, n'importe quelle echelle.
</h3>

<p align="center">
  <a href="https://www.lithosai.com/">LithosAI</a> &middot;
  <a href="https://console.lithosai.cloud/">Cloud</a> &middot;
  <a href="https://docs.motus.lithosai.com/">Docs</a> &middot;
  <a href="https://docs.motus.lithosai.com/getting-started/quickstart">Quickstart</a> &middot;
  <a href="https://github.com/lithos-ai/motus/tree/main/examples">Examples</a> &middot;
  <a href="https://docs.motus.lithosai.com/contributing/development-setup">Contributing</a> &middot;
  <a href="https://join.slack.com/t/lithosaicommunity/shared_invite/zt-3uf2cykza-P9VETbJAUx7WKjwxMk~06Q">Slack</a>
</p>

## A propos

Motus est un projet open source de serving d'agents qui permet des capacites plus elevees, un cout plus faible et des agents plus rapides. Alors que construire des agents n'a jamais ete aussi simple, Motus adopte une approche sans framework et fournit l'infrastructure necessaire a un serving d'agents efficace. Deploiement simple en environnements auto-geres et cloud, a n'importe quelle echelle.

Bonjour de motus codex

## Utiliser avec votre agent de code

Le moyen le plus rapide de demarrer est de laisser votre agent de code gerer la construction, le serving et le deploiement avec Motus.

Motus fonctionne nativement avec tout agent de code (par ex. Claude Code, Codex ou Cursor). Installez le plugin et le CLI en une commande :

```sh
curl -fsSL https://www.lithosai.com/motus/install.sh | sh
```

Puis utilisez-le directement dans votre workflow :

```
/motus                          # activer les competences Motus

build your agent                # commencer a construire votre agent

/motus serve                    # servir en local

/motus deploy                   # deployer dans le cloud
```

Voir [`plugins/motus/README.md`](plugins/motus/README.md) pour les installations du marketplace et plus de details.

## Servir et deployer n'importe quel agent

Installez Motus pour servir des agents en local et les deployer sur [Motus Cloud](https://console.lithosai.cloud/). Motus prend en charge des agents construits avec :

* Motus
*  OpenAI Agents SDK
*  Anthropic SDK
*  Google ADK
*  Plain Python

### Installer Motus dans votre projet

Avec uv :

```bash
uv add lithosai-motus
```

Ou avec pip :

```bash
pip install lithosai-motus
```

### Servir en local et deployer dans le cloud

```bash
# Servir en local
motus serve start myapp:agent --port 8000

# Discuter avec votre agent local
motus serve chat http://localhost:8000 "Hello!"

# Deployer sur Motus Cloud
motus deploy --name myapp myapp:agent

# Discuter avec votre agent deployee
motus serve chat https://myapp.lithosai.com "Hello!"
```

## Construire avec Motus

Motus est propulse par un runtime de serving qui convertit automatiquement le code Python en workflows paralleles et resilients. Tout est concu pour etre simple, intuitif et personnalisable.

### Construire un agent

```python
from motus.agent import ReActAgent
from motus.models import OpenAIChatClient
from motus.runtime import resolve
from motus.tools import tool

@tool  # define a simple tool
async def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# define a ReAct agent
agent = ReActAgent(client=OpenAIChatClient(), model_name="gpt-4o", tools=[search])
print(resolve(agent("Hello World!")))
```

Commencez simplement, et explorez la [documentation des agents](https://docs.motus.lithosai.com/concepts/agents) pour des usages plus avances.

### Construire un workflow

Exemple : recuperer un article, le resumer, extraire des hashtags en parallele, puis publier :

```python
from motus.runtime import resolve
from motus.runtime.agent_task import agent_task

@agent_task # wrap functions as tasks in your workflow
async def summarize(article): ... # just a normal function

@agent_task
async def extract(article): ... # extract hashtags

@agent_task(retries=3, timeout=10.0) # augment tasks with retries and timeouts
async def fetch(url): ...

@agent_task
async def publish(summary, hashtags): ... # publish on LinkedIn

# Your logic becomes your code directly:
article = fetch("https://www.lithosai.com")
summary = summarize(article)            # Motus infers the dependency graph from data flow.
hashtags = extract(article)             # Both depend on `article`, run in parallel.
post = publish(summary, hashtags)       # Waits for both upstream tasks.

print(resolve(post)) # get final result
```

Pas de DAG explicites, juste Python. Motus s'appuie sur les decorateurs `@agent_task` pour transformer des fonctions Python en taches asynchrones.
Motus se place sous vos agents et fournit planification, parallellisme, cache, resilience, observabilite et tracing. [En savoir plus sur le runtime Motus](https://docs.motus.lithosai.com/concepts/workflow).

### Exemples

Executer les exemples inclus :

```bash
# Basic ReAct agent — interactive console chat
uv run python examples/agent.py

# Task graph demo — parallelism, dependency tracking, multi-return
uv run python examples/runtime/task_graph_demo.py
```

En savoir plus avec nos [exemples complets](examples/).

### Fonctionnalites de Motus

#### Commencer simplement

| | |
|---|---|
| **[Agents](https://docs.motus.lithosai.com/concepts/agents)** | `ReActAgent` execute la boucle de raisonnement, le dispatch d'outils et l'etat de conversation. Memoire multi-tour, sortie structuree via Pydantic, et garde-fous d'entree/sortie. Tout est integre. Un agent fonctionnel en moins de 10 lignes. |
| **[Tools](https://docs.motus.lithosai.com/concepts/tools)** | Ecrivez une fonction, obtenez un outil. Exposez des methodes de classe avec `@tools`, encapsulez un serveur MCP avec `get_mcp()`, imbriquez un autre agent avec `as_tool()`, ou executez du code non fiable dans un sandbox Docker. Tout se compose via la meme interface `tools=[...]`. Utilitaires integres : skills, `bash`, operations fichiers, `glob` / `grep`, suivi des taches. |
| **[Task-graph runtime](https://docs.motus.lithosai.com/concepts/workflow)** | `@agent_task` transforme toute fonction en noeud d'un graphe de dependances avec execution parallele automatique, futurs multi-retours, operateurs non bloquants. Retries, timeouts et backoff sont declaratifs sur la tache et surchargeables par appel avec `.policy()`. |
| **[Observability & debugging](https://docs.motus.lithosai.com/guides/tracing)** | Chaque appel LLM, invocation d'outil et dependance de tache est trace automatiquement. Viewer HTML interactif, export Jaeger, ou dashboard cloud. Active avec une seule variable d'environnement. |
| **[Multi-provider models](https://docs.motus.lithosai.com/concepts/models)** | Client unifie pour OpenAI, Anthropic, Gemini et OpenRouter. Changez de fournisseur en une ligne, la logique d'agent reste identique. Les modeles locaux (Ollama, vLLM, SGLang) fonctionnent via `base_url`. |
| **[Local serving](https://docs.motus.lithosai.com/guides/serving)** | `motus serve` expose tout agent comme API HTTP a session en local. Testez toute la pile de serving avant de deploiement cloud. |

#### Aller plus loin

| | |
|---|---|
| **[Memory](https://docs.motus.lithosai.com/concepts/memory)** | Solutions de memoire fournies : `basic` (append-only), `compact` (auto-resume quand le budget de tokens devient faible). Sauvegarde/restauration de session integree. |
| **[Guardrails](https://docs.motus.lithosai.com/guides/guardrails)** | Validation des entrees et sorties sur les agents et les outils individuels. Declarez les parametres qui comptent — retournez un dict pour modifier, levez une exception pour bloquer. Les garde-fous de sortie structuree correspondent aux champs des modeles Pydantic. |
| **[Multi-agent composition](https://docs.motus.lithosai.com/guides/multi-agent)** | `agent.as_tool()` encapsule tout agent comme outil. Le superviseur ne sait pas s'il appelle une fonction ou un autre agent — l'interface est identique. `fork()` cree des branches de conversation independantes. |
| **[MCP integration](https://docs.motus.lithosai.com/guides/mcp-integration)** | Connectez tout serveur compatible MCP avec `get_mcp()`. Local via stdio, distant via HTTP, ou dans un conteneur Docker. Filtrez et renommez des outils avec `prefix`, `blocklist` et des garde-fous. |
| **[Docker sandboxes](https://docs.motus.lithosai.com/concepts/tools)** | Executez du code non fiable dans des conteneurs isoles. Montez des volumes, exposez des ports, executez shell et Python — attachez-les a tout agent comme fournisseur d'outils. |
| **[Prompt caching](https://docs.motus.lithosai.com/concepts/models)** | Mise en cache des prompts via `CachePolicy` — `STATIC` (system + tools) ou `AUTO` (+ prefixe de conversation). Reduit la latence et le cout sur les longues conversations. |
| **SDK compatibility** | Compatible drop-in pour [OpenAI Agents SDK](https://docs.motus.lithosai.com/integrations/openai-agents), [Anthropic SDK](https://docs.motus.lithosai.com/integrations/anthropic-sdk), et [Google ADK](https://docs.motus.lithosai.com/integrations/google-adk). Changez l'import, gardez votre code. |
| **[Human-in-the-loop](https://docs.motus.lithosai.com/guides/human-in-the-loop)** | Support integre pour approbation interactive, clarification et feedback pendant l'execution de l'agent. Mettez l'agent en pause, demandez une entree humaine, et reprenez. Fonctionne en serving local et en deploiement cloud. |

---

## Contribuer

Voir le **[Guide de contribution](https://docs.motus.lithosai.com/contributing/development-setup)** pour demarrer, ou venez dire bonjour sur [Slack](https://join.slack.com/t/lithosaicommunity/shared_invite/zt-3uf2cykza-P9VETbJAUx7WKjwxMk~06Q). Construisons ensemble !

## Licence

Apache 2.0 — voir [LICENSE](LICENSE).
