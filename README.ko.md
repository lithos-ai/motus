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
  더 높은 성능. 더 낮은 비용. 더 빠른 에이전트.<br/>
  셀프 매니지드 또는 클라우드 배포, 한 번의 명령으로 에이전트 서빙. 동일한 코드로 모든 규모에 대응.
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

## 소개

Motus는 더 높은 성능, 더 낮은 비용, 더 빠른 에이전트를 가능하게 하는 오픈소스 에이전트 서빙 프로젝트입니다. 에이전트 구축이 그 어느 때보다 쉬워진 지금, Motus는 노-프레임워크 접근을 통해 효율적인 에이전트 서빙에 필요한 인프라를 제공합니다. 셀프 매니지드와 클라우드 환경 어디서든, 어떤 규모에서도 손쉽게 배포할 수 있습니다.

## 코딩 에이전트와 함께 사용하기

가장 빠른 시작 방법은 코딩 에이전트가 Motus의 빌드, 서빙, 배포를 처리하도록 맡기는 것입니다.

Motus는 어떤 코딩 에이전트(예: Claude Code, Codex, Cursor)와도 바로 동작합니다. 플러그인과 CLI를 한 번에 설치하세요:

```sh
curl -fsSL https://www.lithosai.com/motus/install.sh | sh
```

이후 워크플로에서 바로 사용합니다:

```
/motus                          # Motus 스킬 활성화

build your agent                # 에이전트 구축 시작

/motus serve                    # 로컬 서빙

/motus deploy                   # 클라우드 배포
```

마켓플레이스 설치 및 자세한 내용은 [`plugins/motus/README.md`](plugins/motus/README.md)를 참고하세요.

## 모든 에이전트를 서빙하고 배포하기

Motus를 설치해 로컬에서 에이전트를 서빙하고 [Motus Cloud](https://console.lithosai.cloud/)로 배포할 수 있습니다. Motus는 다음으로 구축된 에이전트를 지원합니다:

* Motus
* OpenAI Agents SDK
* Anthropic SDK
* Google ADK
* Plain Python

### 프로젝트에 Motus 설치

uv 사용:

```bash
uv add lithosai-motus
```

pip 사용:

```bash
pip install lithosai-motus
```

### 로컬에서 서빙하고 클라우드에 배포

```bash
# 로컬 서빙
motus serve start myapp:agent --port 8000

# 로컬 에이전트와 채팅
motus serve chat http://localhost:8000 "Hello!"

# Motus Cloud에 배포
motus deploy --name myapp myapp:agent

# 배포된 에이전트와 채팅
motus serve chat https://myapp.lithosai.com "Hello!"
```

## Motus로 구축하기

Motus는 Python 코드를 자동으로 병렬적이고 탄력적인 워크플로로 변환하는 서빙 런타임으로 구동됩니다. 모든 것이 단순하고 직관적이며, 커스터마이즈 가능하도록 설계되었습니다.

### 에이전트 구축

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

먼저 간단히 시작하고, 더 고급 사용법은 [agents documentation](https://docs.motus.lithosai.com/concepts/agents)을 참고하세요.

### 워크플로 구축

예: 아티클을 가져와 요약하고, 해시태그를 병렬로 추출한 뒤 게시:

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

명시적 DAG 없이 Python만으로 충분합니다. Motus는 `@agent_task` 데코레이터로 Python 함수를 비동기 태스크로 변환합니다.
Motus는 에이전트 아래에서 스케줄링, 병렬성, 캐시, 복원력, 관측성, 트레이싱을 제공합니다. [Motus 런타임에 대해 더 알아보기](https://docs.motus.lithosai.com/concepts/workflow).

### 예제

포함된 예제를 실행하세요:

```bash
# Basic ReAct agent - interactive console chat
uv run python examples/agent.py

# Task graph demo - parallelism, dependency tracking, multi-return
uv run python examples/runtime/task_graph_demo.py
```

[comprehensive examples](examples/)에서 더 많은 내용을 확인하세요.

### Motus 기능

#### 간단하게 시작

| | |
|---|---|
| **[Agents](https://docs.motus.lithosai.com/concepts/agents)** | `ReActAgent`가 추론 루프, 도구 디스패치, 대화 상태를 실행합니다. 다중 턴 메모리, Pydantic 기반 구조화 출력, 입출력 가드레일이 기본 제공됩니다. 10줄 이내로 동작하는 에이전트를 만들 수 있습니다. |
| **[Tools](https://docs.motus.lithosai.com/concepts/tools)** | 함수를 작성하면 도구가 됩니다. `@tools`로 클래스 메서드를 공개하고, `get_mcp()`로 MCP 서버를 래핑하거나, `as_tool()`로 다른 에이전트를 중첩할 수 있습니다. Docker 샌드박스에서 신뢰할 수 없는 코드를 실행할 수도 있습니다. 모든 것은 동일한 `tools=[...]` 인터페이스로 합성됩니다. 기본 유틸리티: skills, `bash`, 파일 작업, `glob` / `grep`, TODO 추적. |
| **[Task-graph runtime](https://docs.motus.lithosai.com/concepts/workflow)** | `@agent_task`가 어떤 함수든 의존 그래프의 노드로 변환하여 자동 병렬 실행, 다중 반환 future, 논블로킹 연산자를 제공합니다. 재시도, 타임아웃, 백오프는 태스크에 선언적으로 지정하고, 호출부에서 `.policy()`로 재정의할 수 있습니다. |
| **[Observability & debugging](https://docs.motus.lithosai.com/guides/tracing)** | LLM 호출, 도구 실행, 태스크 의존 관계를 모두 자동으로 추적합니다. 인터랙티브 HTML 뷰어, Jaeger 내보내기, 클라우드 대시보드를 지원합니다. 환경 변수 하나로 활성화됩니다. |
| **[Multi-provider models](https://docs.motus.lithosai.com/concepts/models)** | OpenAI, Anthropic, Gemini, OpenRouter를 위한 통합 클라이언트. 한 줄만 바꾸면 제공자를 전환해도 에이전트 로직은 그대로 유지됩니다. 로컬 모델(Ollama, vLLM, SGLang)은 `base_url`로 동작합니다. |
| **[Local serving](https://docs.motus.lithosai.com/guides/serving)** | `motus serve`는 어떤 에이전트든 세션 기반 HTTP API로 로컬에 노출합니다. 클라우드에 배포하기 전에 전체 서빙 스택을 테스트할 수 있습니다. |

#### 더 깊게

| | |
|---|---|
| **[Memory](https://docs.motus.lithosai.com/concepts/memory)** | 제공되는 메모리 솔루션: `basic`(append-only), `compact`(토큰 예산이 줄어들면 자동 요약). 세션 저장/복원도 내장되어 있습니다. |
| **[Guardrails](https://docs.motus.lithosai.com/guides/guardrails)** | 에이전트와 개별 도구의 입력/출력을 검증합니다. 중요한 파라미터를 선언하고, dict를 반환해 수정하거나, 예외를 발생시켜 차단할 수 있습니다. 구조화 출력 가드레일은 Pydantic 모델의 필드와 매칭됩니다. |
| **[Multi-agent composition](https://docs.motus.lithosai.com/guides/multi-agent)** | `agent.as_tool()`로 에이전트를 도구화합니다. 슈퍼바이저는 함수인지 다른 에이전트인지 구분하지 않으며 인터페이스는 동일합니다. `fork()`로 대화를 독립적인 브랜치로 분기할 수 있습니다. |
| **[MCP integration](https://docs.motus.lithosai.com/guides/mcp-integration)** | `get_mcp()`로 MCP 호환 서버에 연결합니다. stdio 기반 로컬, HTTP 기반 원격, Docker 컨테이너 내부 모두 지원합니다. `prefix`, `blocklist`, 가드레일로 도구를 필터/리네임할 수 있습니다. |
| **[Docker sandboxes](https://docs.motus.lithosai.com/concepts/tools)** | 신뢰할 수 없는 코드를 격리된 컨테이너에서 실행합니다. 볼륨 마운트, 포트 노출, 셸/Python 실행을 제공하며, 어떤 에이전트에도 도구 제공자로 연결할 수 있습니다. |
| **[Prompt caching](https://docs.motus.lithosai.com/concepts/models)** | `CachePolicy`를 통한 프롬프트 캐싱 - `STATIC`(system + tools) 또는 `AUTO`(대화 프리픽스 추가). 긴 대화의 지연과 비용을 줄입니다. |
| **SDK compatibility** | [OpenAI Agents SDK](https://docs.motus.lithosai.com/integrations/openai-agents), [Anthropic SDK](https://docs.motus.lithosai.com/integrations/anthropic-sdk), [Google ADK](https://docs.motus.lithosai.com/integrations/google-adk)와 호환. import만 바꾸면 코드는 그대로 유지됩니다. |
| **[Human-in-the-loop](https://docs.motus.lithosai.com/guides/human-in-the-loop)** | 에이전트 실행 중 대화형 승인, 확인, 피드백을 기본 지원합니다. 에이전트를 일시 정지하고 사람의 입력을 받아 재개할 수 있습니다. 로컬 서빙과 클라우드 배포 모두에서 동작합니다. |

---

## 기여하기

자세한 내용은 **[Contributing Guide](https://docs.motus.lithosai.com/contributing/development-setup)**를 참고하거나 [Slack](https://join.slack.com/t/lithosaicommunity/shared_invite/zt-3uf2cykza-P9VETbJAUx7WKjwxMk~06Q)에서 인사해 주세요. 함께 만들어 가요!

## 라이선스

Apache 2.0 - [LICENSE](LICENSE)를 참고하세요.
