// Data is injected via script tag in HTML: spans, minTime, totalDuration

// Configure marked.js for safe markdown rendering
if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
}

function renderMarkdown(text) {
    try {
        if (typeof marked !== 'undefined') {
            return marked.parse(text);
        }
    } catch (e) {
        // Fall back to plain text on any error
    }
    return escapeHtml(text);
}

let selectedSpanId = null;
let currentView = 'task'; // 'task' or 'agent'
let currentPage = 'traces'; // 'traces', 'spans', 'logs', 'metrics', 'agents'
let currentMetricsTab = 'cost'; // 'cost', 'latency', 'agent'
const collapsedSpans = new Set();

// Timeline viewport state (continuous zoom/pan)
let viewStart = null;     // left edge of visible time range (null = minTime)
let viewDuration = null;  // visible time range width (null = totalDuration)
let hideMagicTasks = true; // Filter out magic_task spans by default
const collapsedAgents = new Set(); // Track collapsed agents in agent view
const expandedDetailElements = new Set(); // Track expanded elements in detail panel

// SVG Icons — 14px, thin stroke, currentColor
const ICON = {
    reasoning:'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a8 8 0 0 0-8 8c0 3.4 2.1 6.3 5 7.4V20a1 1 0 0 0 1 1h4a1 1 0 0 0 1-1v-2.6c2.9-1.1 5-4 5-7.4a8 8 0 0 0-8-8z"/><path d="M9 22h6"/></svg>',
    agent:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>',
    model:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 18V5"/><path d="M15 13a4.17 4.17 0 0 1-3-4 4.17 4.17 0 0 1-3 4"/><path d="M17.598 6.5A3 3 0 1 0 12 5a3 3 0 1 0-5.598 1.5"/><path d="M17.997 5.125a4 4 0 0 1 2.526 5.77"/><path d="M18 18a4 4 0 0 0 2-7.464"/><path d="M19.967 17.483A4 4 0 1 1 12 18a4 4 0 1 1-7.967-.517"/><path d="M6 18a4 4 0 0 1-2-7.464"/><path d="M6.003 5.125a4 4 0 0 0-2.526 5.77"/></svg>',
    tool:     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M11.42 15.17 17.25 21A2.652 2.652 0 0 0 21 17.25l-5.877-5.877M11.42 15.17l2.496-3.03c.317-.384.74-.626 1.208-.766M11.42 15.17l-4.655 5.653a2.548 2.548 0 1 1-3.586-3.586l6.837-5.63m5.108-.233c.55-.164 1.163-.188 1.743-.14a4.5 4.5 0 0 0 4.486-6.336l-3.276 3.277a3.004 3.004 0 0 1-2.25-2.25l3.276-3.276a4.5 4.5 0 0 0-6.336 4.486c.091 1.076-.071 2.264-.904 2.95l-.102.085m-1.745 1.437L5.909 7.5H4.5L2.25 3.75l1.5-1.5L7.5 4.5v1.409l4.26 4.26m-1.745 1.437 1.745-1.437m6.615 8.206L15.75 15.75M4.867 19.125h.008v.008h-.008v-.008Z"/></svg>',
    magic:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.64 3.64-1.28-1.28a1.21 1.21 0 0 0-1.72 0L2.36 18.64a1.21 1.21 0 0 0 0 1.72l1.28 1.28a1.2 1.2 0 0 0 1.72 0L21.64 5.36a1.2 1.2 0 0 0 0-1.72"/><path d="m14 7 3 3"/><path d="M5 6v4"/><path d="M19 14v4"/><path d="M10 2v2"/><path d="M7 8H3"/><path d="M21 16h-4"/><path d="M11 3H9"/></svg>',
    input:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2.992 16.342a2 2 0 0 1 .094 1.167l-1.065 3.29a1 1 0 0 0 1.236 1.168l3.413-.998a2 2 0 0 1 1.099.092 10 10 0 1 0-4.777-4.719"/><path d="m9 12 2 2 4-4"/></svg>',
    response: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2.992 16.342a2 2 0 0 1 .094 1.167l-1.065 3.29a1 1 0 0 0 1.236 1.168l3.413-.998a2 2 0 0 1 1.099.092 10 10 0 1 0-4.777-4.719"/><path d="M8 12h.01"/><path d="M12 12h.01"/><path d="M16 12h.01"/></svg>',
    traces:   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 5h13"/><path d="M13 12h8"/><path d="M13 19h8"/><path d="M3 10a2 2 0 0 0 2 2h3"/><path d="M3 5v12a2 2 0 0 0 2 2h3"/></svg>',
    spans:    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M21 7.5H3"/><path d="M21 12H3"/><path d="M21 16.5H3"/></svg>',
    logs:     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 5h1"/><path d="M3 12h1"/><path d="M3 19h1"/><path d="M8 5h1"/><path d="M8 12h1"/><path d="M8 19h1"/><path d="M13 5h8"/><path d="M13 12h8"/><path d="M13 19h8"/></svg>',
    metrics:  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12c.552 0 1.005-.449.95-.998a10 10 0 0 0-8.953-8.951c-.55-.055-.998.398-.998.95v8a1 1 0 0 0 1 1z"/><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/></svg>',
    agents:   '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/><path d="M8 21v-1a2 2 0 0 1 2 -2h4a2 2 0 0 1 2 2v1"/><path d="M15 5a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/><path d="M17 10h2a2 2 0 0 1 2 2v1"/><path d="M5 5a2 2 0 1 0 4 0a2 2 0 0 0 -4 0"/><path d="M3 13v-1a2 2 0 0 1 2 -2h2"/></svg>',
    default:  '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.801 10A10 10 0 1 1 17 3.335"/><path d="m9 11 3 3L22 4"/></svg>',
    warn:     '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 1.5L1.5 13.5h13z"/><line x1="8" y1="6" x2="8" y2="9"/><circle cx="8" cy="11.5" r="0.7" fill="currentColor" stroke="none"/></svg>',
    error:    '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><line x1="4" y1="4" x2="12" y2="12"/><line x1="12" y1="4" x2="4" y2="12"/></svg>',
    check:    '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="3,8.5 6.5,12 13,4"/></svg>',
};
const expandedAgentNested = new Set(); // Track expanded agent nested sections

// Resizable divider state
let detailPanelWidth = null; // null = use CSS default
const DETAIL_MIN_WIDTH = 280;
const DETAIL_MAX_WIDTH = 700;
const TIMELINE_MIN_WIDTH = 300;

// Task view layered layout constants
const ROW_HEIGHT = 36;
let spanListPanelWidth = 280;
let detailOverlayWidth = 440;
const SPAN_LIST_MIN_WIDTH = 180;
const SPAN_LIST_MAX_WIDTH = 500;
const DETAIL_OVERLAY_MIN_WIDTH = 280;
const DETAIL_OVERLAY_MAX_WIDTH = 700;

// Build ordered list of visible spans (respecting collapsed state)
function buildVisibleSpans() {
    traceData.rebuild();
    const childrenMap = traceData.getChildrenMap();

    // Initialize collapse state for NEW parent spans
    childrenMap.forEach((_, parentId) => {
        if (!initializedSpans.has(parentId)) {
            collapsedSpans.add(parentId);
            initializedSpans.add(parentId);
        }
    });

    const result = [];

    function walk(span, level) {
        const children = traceData.getChildren(span.spanId);
        const skip = hideMagicTasks && isMagicSpan(span.meta);
        if (!skip) {
            result.push({ span, level, hasChildren: children.length > 0, childCount: children.length });
        }
        if (skip || !collapsedSpans.has(span.spanId)) {
            children.forEach(child => walk(child, skip ? level : level + 1));
        }
    }

    traceData.getRoots().forEach(span => walk(span, 0));

    result.forEach((item, idx) => { item.rowIndex = idx; });
    return { visibleSpans: result };
}

// Compute nice tick interval for time axis (1-2-5 sequence)
function niceTickInterval(viewDur) {
    if (viewDur <= 0) return 1000; // guard: 1ms minimum
    const rough = viewDur / 8; // aim for ~8 ticks
    const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
    const residual = rough / magnitude;
    let nice;
    if (residual <= 1.5) nice = 1;
    else if (residual <= 3.5) nice = 2;
    else if (residual <= 7.5) nice = 5;
    else nice = 10;
    return Math.max(nice * magnitude, 1); // never below 1µs to prevent infinite loops
}

function formatDuration(microseconds) {
    if (microseconds < 1000) return microseconds.toFixed(0) + 'µs';
    if (microseconds < 1000000) return (microseconds / 1000).toFixed(2) + 'ms';
    return (microseconds / 1000000).toFixed(2) + 's';
}

function formatAbsoluteTime(span, isEnd = false) {
    const startUs = span.meta?.start_us || span.startTime;
    if (!startUs) return '—';
    const ms = isEnd ? startUs / 1000 + span.duration / 1000 : startUs / 1000;
    const d = new Date(ms);
    const pad = (n, len = 2) => String(n).padStart(len, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

// ── Shared UI helpers ────────────────────────────────────────────

// Single-click / double-click discriminator (200ms delay)
function attachClickHandlers(el, { onSingleClick, onDoubleClick }) {
    let timer = null;
    el.addEventListener('click', (e) => {
        if (timer) { clearTimeout(timer); timer = null; return; }
        timer = setTimeout(() => { timer = null; onSingleClick(e); }, 200);
    });
    el.addEventListener('dblclick', (e) => {
        if (timer) { clearTimeout(timer); timer = null; }
        onDoubleClick(e);
    });
}

// ── Span classification helpers ──────────────────────────────────
// Classify spans by task_type category.

function isModelSpan(meta) {
    return (meta?.task_type || '') === 'model_call';
}

function isToolSpan(meta) {
    return (meta?.task_type || '') === 'tool_call';
}

function isAgentSpan(meta) {
    return (meta?.task_type || '') === 'agent_call';
}

function isMagicSpan(meta) {
    return (meta?.task_type || '') === 'magic_task';
}

function getSpanClass(meta) {
    if (meta && meta.error) return 'error';
    if (isAgentSpan(meta)) return 'agent';
    if (isModelSpan(meta)) return 'model';
    if (isToolSpan(meta)) return 'tool';
    if (isMagicSpan(meta)) return 'magic';
    return 'default';
}

function getSpanIcon(meta) {
    if (isAgentSpan(meta)) return ICON.agent;
    if (isModelSpan(meta)) return ICON.model;
    if (isToolSpan(meta)) return ICON.tool;
    if (isMagicSpan(meta)) return ICON.magic;
    return ICON.default;
}

function getSpanDisplayName(span) {
    // For agent spans, show agent_id
    if (isAgentSpan(span.meta) && span.meta?.agent_id) {
        return span.meta.agent_id;
    }
    // For tool_call spans, show tool name from meta
    if (isToolSpan(span.meta) && span.meta?.tool_input_meta) {
        const toolMeta = span.meta.tool_input_meta;
        const toolName = toolMeta.name || toolMeta.function?.name;
        if (toolName) {
            return toolName;
        }
    }
    return span.operationName;
}

// Track which spans we've already initialized collapse state for
const initializedSpans = new Set();
const initializedAgents = new Set();

// ── Cached span hierarchy ────────────────────────────────────
// Rebuilt lazily when spans change (SSE upsert / init).

const traceData = {
    _spanMap: new Map(),
    _childrenMap: new Map(),
    _rootSpans: [],
    _dirty: true,

    rebuild() {
        if (!this._dirty) return;
        this._spanMap.clear();
        this._childrenMap.clear();
        this._rootSpans = [];
        spans.forEach(span => {
            this._spanMap.set(span.spanId, span);
            if (!span.parentSpanId) {
                this._rootSpans.push(span);
            } else {
                if (!this._childrenMap.has(span.parentSpanId)) {
                    this._childrenMap.set(span.parentSpanId, []);
                }
                this._childrenMap.get(span.parentSpanId).push(span);
            }
        });
        // Sort roots and children by startTime so timeline order is chronological
        const byStart = (a, b) => a.startTime - b.startTime;
        this._rootSpans.sort(byStart);
        this._childrenMap.forEach(children => children.sort(byStart));
        this._dirty = false;
    },

    invalidate() { this._dirty = true; },
    getSpan(id) { this.rebuild(); return this._spanMap.get(id); },
    getChildren(id) { this.rebuild(); return this._childrenMap.get(id) || []; },
    getChildrenMap() { this.rebuild(); return this._childrenMap; },
    getRoots() { this.rebuild(); return this._rootSpans; },
};

function computeTraceStats() {
    let totalCost = 0, hasCost = false, totalErrors = 0;
    spans.forEach(span => {
        if (span.tags?.['model.cost_usd'] !== undefined) {
            totalCost += span.tags['model.cost_usd'];
            hasCost = true;
        }
        if (span.meta?.error) totalErrors++;
    });
    return { totalCost, hasCost, totalErrors };
}

function applyTraceStats() {
    const { totalCost, hasCost, totalErrors } = computeTraceStats();
    const totalSpansEl = document.getElementById('totalSpans');
    const totalDurationEl = document.getElementById('totalDuration');
    if (totalSpansEl) totalSpansEl.textContent = spans.length;
    if (totalDurationEl) totalDurationEl.textContent = formatDuration(totalDuration);

    const costEl = document.getElementById('totalCost');
    const costStatEl = document.getElementById('totalCostStat');
    if (hasCost) {
        if (costEl) costEl.textContent = '$' + totalCost.toFixed(5);
        if (costStatEl) costStatEl.style.display = 'flex';
    } else {
        if (costEl) costEl.textContent = '';
        if (costStatEl) costStatEl.style.display = 'none';
    }
    const errEl = document.getElementById('totalErrors');
    const errStatEl = document.getElementById('totalErrorsStat');
    if (totalErrors > 0) {
        if (errEl) errEl.textContent = totalErrors;
        if (errStatEl) errStatEl.style.display = 'flex';
    } else {
        if (errEl) errEl.textContent = '0';
        if (errStatEl) errStatEl.style.display = 'none';
    }
}

// --- Resizable Panel Divider ---

function ensureDivider() {
    const container = document.querySelector('.container');
    const detailPanel = document.getElementById('detailPanel');
    if (!container || !detailPanel) return;

    // Don't add divider on non-trace pages or task view (has own resize handles)
    if (container.classList.contains('single-panel')) return;
    if (container.classList.contains('task-view')) return;

    // Check if divider already exists
    let divider = container.querySelector('.panel-divider');
    if (!divider) {
        divider = document.createElement('div');
        divider.className = 'panel-divider';
        container.insertBefore(divider, detailPanel);

        initDragResize(divider, {
            getStartWidth: () => document.getElementById('detailPanel').getBoundingClientRect().width,
            onMove(dx, startWidth) {
                const cont = document.querySelector('.container');
                let newWidth = startWidth - dx;
                newWidth = Math.max(DETAIL_MIN_WIDTH, Math.min(DETAIL_MAX_WIDTH, newWidth));
                const contRect = cont.getBoundingClientRect();
                const availableWidth = contRect.width - 6 - 32;
                if (contRect.width - 6 - newWidth - 32 < TIMELINE_MIN_WIDTH) {
                    newWidth = availableWidth - TIMELINE_MIN_WIDTH;
                }
                newWidth = Math.max(DETAIL_MIN_WIDTH, newWidth);
                detailPanelWidth = newWidth;
                applyDetailPanelWidth(newWidth);
            },
        });
        divider.addEventListener('dblclick', function() {
            detailPanelWidth = null;
            applyDetailPanelWidth(null);
        });
    }

    // Reapply saved width
    if (detailPanelWidth !== null) {
        applyDetailPanelWidth(detailPanelWidth);
    }
}

function applyDetailPanelWidth(width) {
    const container = document.querySelector('.container');
    const detailPanel = document.getElementById('detailPanel');
    if (!container || !detailPanel) return;

    if (container.classList.contains('agent-view')) {
        // Grid mode
        if (width === null) {
            container.style.gridTemplateColumns = '';
        } else {
            container.style.gridTemplateColumns = `1fr 6px ${width}px`;
        }
        detailPanel.style.width = '';
    } else {
        // Flex mode
        container.style.gridTemplateColumns = '';
        if (width === null) {
            detailPanel.style.width = '';
        } else {
            detailPanel.style.width = width + 'px';
        }
    }
}

function initDragResize(handle, { getStartWidth, onMove }) {
    function startDrag(e) {
        e.preventDefault();
        handle.classList.add('dragging');
        document.body.classList.add('resizing-panels');

        const startX = e.touches ? e.touches[0].clientX : e.clientX;
        const startWidth = getStartWidth();

        function onPointerMove(e) {
            if (e.cancelable) e.preventDefault();
            const currentX = e.touches ? e.touches[0].clientX : e.clientX;
            onMove(currentX - startX, startWidth);
        }

        function onPointerUp() {
            handle.classList.remove('dragging');
            document.body.classList.remove('resizing-panels');
            document.removeEventListener('mousemove', onPointerMove);
            document.removeEventListener('mouseup', onPointerUp);
            document.removeEventListener('touchmove', onPointerMove);
            document.removeEventListener('touchend', onPointerUp);
            document.removeEventListener('touchcancel', onPointerUp);
        }

        document.addEventListener('mousemove', onPointerMove);
        document.addEventListener('mouseup', onPointerUp);
        document.addEventListener('touchmove', onPointerMove, { passive: false });
        document.addEventListener('touchend', onPointerUp);
        document.addEventListener('touchcancel', onPointerUp);
    }

    handle.addEventListener('mousedown', startDrag);
    handle.addEventListener('touchstart', startDrag, { passive: false });
}


// Enforce constraints on window resize
window.addEventListener('resize', function() {
    if (detailPanelWidth !== null) {
        const container = document.querySelector('.container');
        if (container && !container.classList.contains('single-panel') && !container.classList.contains('task-view')) {
            const containerRect = container.getBoundingClientRect();
            const availableWidth = containerRect.width - 6 - 32;
            if (availableWidth - detailPanelWidth < TIMELINE_MIN_WIDTH) {
                detailPanelWidth = Math.max(DETAIL_MIN_WIDTH, availableWidth - TIMELINE_MIN_WIDTH);
                applyDetailPanelWidth(detailPanelWidth);
            }
        }
    }
});

// Switch between task view and agent view
function toggleMagicTaskFilter() {
    hideMagicTasks = document.getElementById('filterMagicTasks')?.checked ?? true;
    if (currentView === 'task') {
        renderTaskTimeline();
    } else {
        renderTimeline();
    }
}

function toggleTheme() {
    const isLight = document.body.classList.toggle('light-theme');
    // Swap icon visibility
    document.getElementById('themeIconSun').style.display = isLight ? 'none' : '';
    document.getElementById('themeIconMoon').style.display = isLight ? '' : 'none';
    // Swap highlight.js stylesheet
    const hljsLink = document.getElementById('hljsTheme');
    if (hljsLink) {
        hljsLink.href = isLight
            ? 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github.min.css'
            : 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css';
    }
}

function switchView(view) {
    if (currentView === view) return;
    currentView = view;

    // Update button states
    document.getElementById('taskViewBtn').classList.toggle('active', view === 'task');
    document.getElementById('agentViewBtn').classList.toggle('active', view === 'agent');

    // Clear selection
    selectedSpanId = null;

    // Re-render the traces page with the new view
    renderTracesPage();
}

// Switch between metrics tabs
function switchMetricsTab(tab) {
    if (currentMetricsTab === tab) return;
    currentMetricsTab = tab;

    // Update tab button states
    document.querySelectorAll('.metrics-tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Re-render metrics page with new tab
    renderMetricsPage();
}

// Group spans by agent for agent-centric view
function groupSpansByAgent() {
    traceData.rebuild();
    const agentGroups = new Map();

    spans.forEach(span => {
        if (isAgentSpan(span.meta) && span.meta?.agent_id) {
            const agentId = span.meta.agent_id;
            const objectId = span.meta.object_id || 'unknown';
            const key = `${agentId}|${objectId}`;

            if (!agentGroups.has(key)) {
                agentGroups.set(key, {
                    agentId: agentId,
                    objectId: objectId,
                    key: key,
                    executions: [],
                    nestedTasks: []
                });
            }

            const group = agentGroups.get(key);
            group.executions.push(span);
            collectNestedTasks(span, group.nestedTasks);
        }
    });

    return { agentGroups };
}

// Recursively collect model and tool spans from descendants
function collectNestedTasks(parentSpan, nestedTasks) {
    const children = traceData.getChildren(parentSpan.spanId);
    children.forEach(child => {
        if (isModelSpan(child.meta) || isToolSpan(child.meta)) {
            nestedTasks.push({
                span: child,
                parentExecution: parentSpan
            });
        }
        // Recursively collect from deeper children
        collectNestedTasks(child, nestedTasks);
    });
}

// Render agent-centric graph view - each execution is a separate node
function renderAgentTimeline() {
    const panel = document.getElementById('timelinePanel');
    panel.innerHTML = '';

    // Ensure container has agent-view class for grid layout
    const container = document.querySelector('.container');
    container.classList.add('agent-view');

    traceData.rebuild();

    // Find all AgentBase._execute spans with agent_id (each one becomes a node)
    const executions = [];
    spans.forEach(span => {
        if (isAgentSpan(span.meta) && span.meta?.agent_id) {
            const nestedTasks = [];
            collectNestedTasks(span, nestedTasks);

            executions.push({
                span: span,
                agentId: span.meta.agent_id,
                objectId: span.meta.object_id || 'unknown',
                nestedTasks: nestedTasks
            });
        }
    });

    if (executions.length === 0) {
        panel.innerHTML = `
            <div class="agent-graph-empty">
                <div class="agent-graph-empty-icon">${ICON.agents}</div>
                <div class="agent-graph-empty-text">No agent executions found</div>
                <div class="agent-graph-empty-subtext">Agent view shows AgentBase._execute calls with agent_id tags</div>
            </div>
        `;
        // Hide nested panel when no executions
        const nestedPanel = document.getElementById('agentNestedSpansPanel');
        if (nestedPanel) {
            nestedPanel.style.display = 'none';
        }
        return;
    }

    // Sort executions by start time (left to right)
    executions.sort((a, b) => a.span.startTime - b.span.startTime);

    // Create graph container (just for the timeline nodes, not the nested panel)
    const graphContainer = document.createElement('div');
    graphContainer.className = 'agent-graph-container';

    // Create timeline row with nodes
    const timeline = document.createElement('div');
    timeline.className = 'agent-graph-timeline';

    executions.forEach((exec, index) => {
        // Create execution node
        const node = createExecutionNode(exec, index);
        timeline.appendChild(node);

        // Add arrow connector after each node (except the last)
        if (index < executions.length - 1) {
            const connector = createNodeConnector();
            timeline.appendChild(connector);
        }
    });

    graphContainer.appendChild(timeline);
    panel.appendChild(graphContainer);

    // Ensure nested panel exists and is visible (it's a sibling of timelinePanel now)
    let nestedPanel = document.getElementById('agentNestedSpansPanel');
    if (!nestedPanel) {
        nestedPanel = document.createElement('div');
        nestedPanel.id = 'agentNestedSpansPanel';
        nestedPanel.className = 'agent-nested-spans-panel';
        const container = document.querySelector('.container');
        const detailPanel = document.getElementById('detailPanel');
        container.insertBefore(nestedPanel, detailPanel);
    }
    nestedPanel.style.display = 'flex';
    nestedPanel.innerHTML = `
        <div class="agent-nested-empty">
            <span>Select an agent node to view its model and tool calls</span>
        </div>
    `;
    ensureDivider();

    // Update stats for agent view
    applyTraceStats();
    const uniqueAgents = new Set(executions.map(e => e.agentId)).size;
    const totalSpansEl = document.getElementById('totalSpans');
    if (totalSpansEl) totalSpansEl.textContent = `${executions.length} executions (${uniqueAgents} agents)`;

}

// Create an execution node element (one per AgentBase._execute call)
function createExecutionNode(exec, index) {
    const node = document.createElement('div');
    node.className = 'agent-node';
    node.dataset.spanId = exec.span.spanId;
    node.dataset.execIndex = index;

    // Calculate stats for this execution
    const duration = exec.span.duration;
    const modelCalls = exec.nestedTasks.filter(t => isModelSpan(t.span.meta)).length;
    const toolCalls = exec.nestedTasks.filter(t => isToolSpan(t.span.meta)).length;
    const hasError = exec.span.meta?.error || exec.nestedTasks.some(t => t.span.meta?.error);

    // Calculate cost for this execution
    let execCost = 0;
    exec.nestedTasks.forEach(t => {
        if (t.span.tags && t.span.tags['model.cost_usd']) {
            execCost += t.span.tags['model.cost_usd'];
        }
    });

    if (hasError) {
        node.classList.add('has-error');
    }

    const errorBadge = hasError ? '<span class="agent-node-error-badge">ERROR</span>' : '';
    const shortObjectId = String(exec.objectId).slice(-4);

    node.innerHTML = `
        <div class="agent-node-header">
            <div class="agent-node-title">
                <div class="agent-node-icon">${ICON.agent}</div>
                <span class="agent-node-name">${exec.agentId}</span>
                ${errorBadge}
            </div>
            <div class="agent-node-subtitle">#${shortObjectId}</div>
        </div>
        <div class="agent-node-body">
            <div class="agent-node-stats">
                <div class="agent-node-stat">
                    <span class="agent-node-stat-label">LLM</span>
                    <span class="agent-node-stat-value highlight-green">${modelCalls}</span>
                </div>
                <div class="agent-node-stat">
                    <span class="agent-node-stat-label">Tools</span>
                    <span class="agent-node-stat-value highlight-purple">${toolCalls}</span>
                </div>
            </div>
        </div>
        <div class="agent-node-footer">
            <div class="agent-node-duration">⏱ ${formatDuration(duration)}</div>
            ${execCost > 0 ? `<div class="agent-node-cost">$${execCost.toFixed(4)}</div>` : ''}
        </div>
    `;

    // Click handler to select this execution
    node.addEventListener('click', () => {
        selectExecutionNode(exec, node);
    });

    return node;
}

// Create an arrow connector between nodes
function createNodeConnector() {
    const connector = document.createElement('div');
    connector.className = 'agent-node-connector';
    connector.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="5" y1="12" x2="19" y2="12"></line>
            <polyline points="12 5 19 12 12 19"></polyline>
        </svg>
    `;
    return connector;
}

// Select an execution node
function selectExecutionNode(exec, node) {
    selectedSpanId = exec.span.spanId;

    // Update visual selection
    document.querySelectorAll('.agent-node').forEach(n => {
        n.classList.remove('selected');
    });
    node.classList.add('selected');

    // Render nested spans panel (model/tool calls timeline)
    renderNestedSpansPanel(exec);

    // Render execution details
    renderExecutionDetails(exec);
}

// Render nested spans panel showing model and tool calls for selected agent
function renderNestedSpansPanel(exec) {
    const panel = document.getElementById('agentNestedSpansPanel');
    if (!panel) return;

    const modelCalls = exec.nestedTasks.filter(t => isModelSpan(t.span.meta));
    const toolCalls = exec.nestedTasks.filter(t => isToolSpan(t.span.meta));
    const allCalls = [...modelCalls, ...toolCalls].sort((a, b) => a.span.startTime - b.span.startTime);

    if (allCalls.length === 0) {
        panel.innerHTML = `
            <div class="agent-nested-empty">
                <span>No model or tool calls in this execution</span>
            </div>
        `;
        return;
    }

    // Calculate time bounds for this execution
    const execStart = exec.span.startTime;
    const execDuration = exec.span.duration;

    let html = `
        <div class="agent-nested-header">
            <div class="agent-nested-title">
                <span class="agent-nested-icon">${ICON.metrics}</span>
                <span>${exec.agentId} - Calls Timeline</span>
            </div>
            <div class="agent-nested-stats">
                <span class="agent-nested-stat">${ICON.model} ${modelCalls.length} LLM</span>
                <span class="agent-nested-stat">${ICON.tool} ${toolCalls.length} Tools</span>
            </div>
        </div>
        <div class="agent-nested-timeline">
    `;

    allCalls.forEach(call => {
        const span = call.span;
        const isModel = isModelSpan(span.meta);
        const spanClass = isModel ? 'model' : 'tool';
        const icon = isModel ? ICON.model : ICON.tool;
        const hasError = span.meta?.error;

        // Calculate position within execution timeline
        const startPercent = ((span.startTime - execStart) / execDuration) * 100;
        const widthPercent = (span.duration / execDuration) * 100;

        // Get additional info (just duration)
        const extraInfo = formatDuration(span.duration);

        html += `
            <div class="agent-nested-row ${hasError ? 'has-error' : ''}" data-span-id="${span.spanId}" onclick="selectSpan('${span.spanId}')">
                <div class="agent-nested-label">
                    <span class="agent-nested-row-icon">${icon}</span>
                    <span class="agent-nested-row-name">${span.operationName}</span>
                    ${hasError ? '<span class="agent-nested-error-badge">' + ICON.warn + '</span>' : ''}
                </div>
                <div class="agent-nested-bar-container">
                    <div class="agent-nested-bar ${spanClass}" style="left: ${startPercent}%; width: ${Math.max(widthPercent, 0.5)}%"></div>
                </div>
                <div class="agent-nested-duration">${extraInfo}</div>
            </div>
        `;
    });

    html += '</div>';
    panel.innerHTML = html;
}

// Render execution details in the detail panel
function renderExecutionDetails(exec) {
    const panel = document.getElementById('detailPanel');

    const duration = exec.span.duration;
    const modelCalls = exec.nestedTasks.filter(t => isModelSpan(t.span.meta));
    const toolCalls = exec.nestedTasks.filter(t => isToolSpan(t.span.meta));
    const errors = [exec.span, ...exec.nestedTasks.map(t => t.span)].filter(s => s.meta?.error);

    // Calculate cost and tokens for this execution
    let totalCost = 0;
    let totalTokens = 0;
    let promptTokens = 0;
    let completionTokens = 0;

    modelCalls.forEach(t => {
        if (t.span.tags) {
            if (t.span.tags['model.cost_usd']) totalCost += t.span.tags['model.cost_usd'];
            if (t.span.tags['model.tokens.total']) totalTokens += t.span.tags['model.tokens.total'];
            if (t.span.tags['model.tokens.prompt']) promptTokens += t.span.tags['model.tokens.prompt'];
            if (t.span.tags['model.tokens.completion']) completionTokens += t.span.tags['model.tokens.completion'];
        }
    });

    const shortObjectId = String(exec.objectId).slice(-6);

    let html = `
        <div class="detail-title">${ICON.agent} ${exec.agentId} <span class="text-muted-sm">#${shortObjectId}</span></div>

        <div class="detail-section">
            <div class="section-header">Execution Overview</div>
            <div class="detail-item">
                <div class="detail-label">Agent ID</div>
                <div class="detail-value"><code>${exec.agentId}</code></div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Span ID</div>
                <div class="detail-value"><code>${exec.span.spanId}</code></div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Object ID</div>
                <div class="detail-value"><code>${exec.objectId}</code></div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Duration</div>
                <div class="detail-value"><strong>${formatDuration(duration)}</strong></div>
            </div>
        </div>

        <div class="detail-section">
            <div class="section-header">Resource Usage</div>
            <div class="token-grid">
                <div class="token-card">
                    <div class="token-label">${ICON.model} LLM Calls</div>
                    <div class="token-value">${modelCalls.length}</div>
                </div>
                <div class="token-card">
                    <div class="token-label">${ICON.tool} Tool Calls</div>
                    <div class="token-value">${toolCalls.length}</div>
                </div>
                <div class="token-card">
                    <div class="token-label">Total Tokens</div>
                    <div class="token-value">${totalTokens.toLocaleString()}</div>
                </div>
                <div class="token-card cost">
                    <div class="token-label">$ Cost</div>
                    <div class="token-value">$${totalCost.toFixed(5)}</div>
                </div>
            </div>
        </div>
    `;

    // Errors section
    if (errors.length > 0) {
        html += `
            <div class="detail-section error-section">
                <div class="section-header" style="color: var(--status-error); border-color: var(--badge-error-bg);">${ICON.warn} Errors (${errors.length})</div>
        `;
        errors.forEach((errSpan, idx) => {
            const errorMsg = typeof errSpan.meta?.error === 'string' ? errSpan.meta.error : JSON.stringify(errSpan.meta?.error, null, 2);
            html += `
                <div class="message-box">
                    <div class="message-header" onclick="toggleMessage('exec-error-${idx}')">
                        <span>${errSpan.operationName || errSpan.meta?.func || 'Error'}</span>
                        <span class="toggle-icon" id="exec-error-${idx}-icon">▼</span>
                    </div>
                    <div class="message-content" id="exec-error-${idx}">
                        <pre style="white-space: pre-wrap; color: var(--error-code-color);">${escapeHtml(errorMsg)}</pre>
                    </div>
                </div>
            `;
        });
        html += `</div>`;
    }

    // Model calls summary
    if (modelCalls.length > 0) {
        html += `
            <div class="detail-section">
                <div class="section-header clickable" onclick="toggleMessage('exec-models')">
                    <span>${ICON.model} LLM Calls (${modelCalls.length})</span>
                    <span class="toggle-icon" id="exec-models-icon">▼</span>
                </div>
                <div class="message-content full-height" id="exec-models">
        `;
        modelCalls.forEach((call) => {
            const cost = call.span.tags?.['model.cost_usd'] || 0;
            const tokens = call.span.tags?.['model.tokens.total'] || 0;
            html += `
                <div class="message-box" style="cursor: pointer;" onclick="selectSpan('${call.span.spanId}')">
                    <div class="message-header">
                        <span>${call.span.operationName}</span>
                        <span>${tokens} tokens • $${cost.toFixed(4)}</span>
                    </div>
                </div>
            `;
        });
        html += `</div></div>`;
    }

    // Tool calls summary
    if (toolCalls.length > 0) {
        html += `
            <div class="detail-section">
                <div class="section-header clickable" onclick="toggleMessage('exec-tools')">
                    <span>${ICON.tool} Tool Calls (${toolCalls.length})</span>
                    <span class="toggle-icon" id="exec-tools-icon">▼</span>
                </div>
                <div class="message-content full-height" id="exec-tools">
        `;
        toolCalls.forEach((call) => {
            html += `
                <div class="message-box" style="cursor: pointer;" onclick="selectSpan('${call.span.spanId}')">
                    <div class="message-header">
                        <span>${call.span.operationName}</span>
                        <span>${formatDuration(call.span.duration)}</span>
                    </div>
                </div>
            `;
        });
        html += `</div></div>`;
    }

    panel.innerHTML = html;
    restoreDetailPanelState();
}

function renderTimeline() {
    const panel = document.getElementById('timelinePanel');
    panel.innerHTML = '';

    // Ensure container doesn't have agent-view class for flex layout
    const container = document.querySelector('.container');
    container.classList.remove('agent-view');

    // Build hierarchy from cache
    traceData.rebuild();
    const childrenMap = traceData.getChildrenMap();
    const rootSpans = traceData.getRoots();

    // Initialize collapse state only for NEW parent spans (preserve existing state)
    childrenMap.forEach((_, parentId) => {
        if (!initializedSpans.has(parentId)) {
            collapsedSpans.add(parentId);
            initializedSpans.add(parentId);
        }
    });

    function renderSpanRow(span, level = 0, parentSpanId = null) {
        const children = traceData.getChildren(span.spanId);
        const isChild = parentSpanId !== null;

        // Skip magic tasks when filtered, promote children at same level
        if (hideMagicTasks && isMagicSpan(span.meta)) {
            children.forEach(child => renderSpanRow(child, level, parentSpanId));
            return;
        }

        const hasChildren = children.length > 0;
        const row = document.createElement('div');
        row.className = isChild ? 'timeline-row timeline-child' : 'timeline-row';
        row.dataset.spanId = span.spanId;
        row.dataset.level = level;
        if (isChild) {
            row.dataset.parentSpanId = parentSpanId;
            row.style.display = collapsedSpans.has(parentSpanId) ? 'none' : 'flex';
        }

        const indent = level * 24;
        const vs = viewStart ?? minTime;
        const vd = viewDuration ?? (totalDuration || 1);
        const startPercent = ((span.startTime - vs) / vd) * 100;
        const widthPercent = (span.duration / vd) * 100;
        const spanClass = getSpanClass(span.meta);
        const icon = getSpanIcon(span.meta);
        const hasError = span.meta && span.meta.error;
        const errorIndicator = hasError ? '<span class="error-indicator" title="Error occurred">' + ICON.warn + '</span>' : '';

        let toggleHtml = '';
        if (hasChildren) {
            const isCollapsed = collapsedSpans.has(span.spanId);
            const chevronClass = isCollapsed ? 'collapsed' : 'expanded';
            toggleHtml = `<div class="span-toggle"><div class="toggle-chevron ${chevronClass}">▶</div></div>`;
        } else {
            toggleHtml = `<div class="span-toggle"></div>`;
        }

        const childCountHtml = hasChildren ? `<span class="child-count">(${children.length})</span>` : '';
        const displayName = getSpanDisplayName(span);

        let guideLinesHtml = '';
        if (isChild) {
            for (let i = 0; i < level; i++) {
                const lineLeft = 12 + i * 24 + 10;
                guideLinesHtml += `<div class="tree-guide-line" style="left: ${lineLeft}px"></div>`;
            }
        }

        row.innerHTML = `
            ${guideLinesHtml}
            <div class="span-label-wrapper" style="padding-left: ${indent}px">
                ${toggleHtml}
                <div class="span-icon">${icon}</div>
                <div class="span-label">
                    ${displayName}
                    ${childCountHtml}
                    ${errorIndicator}
                </div>
            </div>
            <div class="span-bar-container">
                <div class="span-bar ${spanClass}"
                     style="left: ${startPercent}%; width: ${widthPercent}%">
                </div>
            </div>
            <div class="span-duration">${formatDuration(span.duration)}</div>
        `;

        attachClickHandlers(row, {
            onSingleClick: (e) => { selectSpan(span.spanId); if (hasChildren) toggleChildren(e, span.spanId); },
            onDoubleClick: () => { selectSpan(span.spanId); zoomToSpan(span); },
        });

        panel.appendChild(row);

        children.forEach((child) => {
            renderSpanRow(child, level + 1, span.spanId);
        });
    }

    rootSpans.forEach(span => renderSpanRow(span));
    applyTraceStats();
}

// ── Task view: layered timeline rendering ───────────────────

function renderTaskTimeline() {
    const spanListScroll = document.getElementById('spanListScroll');
    const canvasBars = document.getElementById('canvasBars');
    if (!spanListScroll || !canvasBars) return;

    const { visibleSpans } = buildVisibleSpans();

    renderSpanLabels(visibleSpans, spanListScroll);
    renderCanvasBars(visibleSpans, canvasBars);
    renderTimeAxis();
    renderGridlines();
    syncScrollSetup();
    updatePanelShadows();
    applyTraceStats();

    // Re-apply selection
    if (selectedSpanId) {
        reapplySelectionHighlight(selectedSpanId);
    }
}

function renderSpanLabels(visibleSpans, container) {
    container.innerHTML = '';

    // Precompute isLastChild for tree guide lines
    const isLast = new Array(visibleSpans.length).fill(false);
    for (let i = 0; i < visibleSpans.length; i++) {
        const level = visibleSpans[i].level;
        if (level === 0) continue;
        let hasSibling = false;
        for (let j = i + 1; j < visibleSpans.length; j++) {
            if (visibleSpans[j].level < level) break;
            if (visibleSpans[j].level === level) { hasSibling = true; break; }
        }
        isLast[i] = !hasSibling;
    }

    // Track which levels have continuing vertical lines
    const activeLines = new Set();

    visibleSpans.forEach(({ span, level, hasChildren, childCount }, idx) => {
        const row = document.createElement('div');
        row.className = 'span-label-row';
        row.dataset.spanId = span.spanId;
        if (level > 0) row.classList.add('timeline-child');

        const indent = level * 24;
        const icon = getSpanIcon(span.meta);
        const hasError = span.meta && span.meta.error;
        const errorIndicator = hasError ? '<span class="error-indicator" title="Error occurred">' + ICON.warn + '</span>' : '';
        const displayName = getSpanDisplayName(span);

        let toggleHtml = '';
        if (hasChildren) {
            const isCollapsed = collapsedSpans.has(span.spanId);
            const chevronClass = isCollapsed ? 'collapsed' : 'expanded';
            toggleHtml = `<div class="span-toggle"><div class="toggle-chevron ${chevronClass}">▶</div></div>`;
        } else {
            toggleHtml = `<div class="span-toggle"></div>`;
        }

        const childCountHtml = hasChildren ? `<span class="child-count">(${childCount})</span>` : '';

        // Build tree guide lines
        let guideHtml = '';
        if (level > 0) {
            // Vertical continuation lines for ancestor levels
            for (let l = 1; l < level; l++) {
                if (activeLines.has(l)) {
                    guideHtml += `<div class="tree-vline" style="left:${(l - 1) * 24 + 11}px"></div>`;
                }
            }
            // Connector at this item's level: ├ or └
            const x = (level - 1) * 24 + 11;
            if (isLast[idx]) {
                guideHtml += `<div class="tree-vline last" style="left:${x}px"></div>`;
                activeLines.delete(level);
            } else {
                guideHtml += `<div class="tree-vline" style="left:${x}px"></div>`;
                activeLines.add(level);
            }
            guideHtml += `<div class="tree-hline" style="left:${x}px"></div>`;
        }

        row.innerHTML = `
            <div class="span-label-wrapper" style="padding-left: ${indent}px">
                ${guideHtml}
                ${toggleHtml}
                <div class="span-icon">${icon}</div>
                <div class="span-label">
                    ${displayName}
                    ${childCountHtml}
                    ${errorIndicator}
                </div>
            </div>
        `;

        attachClickHandlers(row, {
            onSingleClick: (e) => { selectSpan(span.spanId); if (hasChildren) toggleChildren(e, span.spanId); },
            onDoubleClick: () => { selectSpan(span.spanId); zoomToSpan(span); },
        });

        container.appendChild(row);
    });
}

function renderCanvasBars(visibleSpans, container) {
    container.innerHTML = '';

    const vs = getViewStart();
    const vd = getViewDuration();
    const panelLeft = spanListPanelWidth;
    const panelRight = detailOverlayWidth;

    visibleSpans.forEach(({ span, rowIndex }) => {
        const barRow = document.createElement('div');
        barRow.className = 'canvas-bar-row';
        barRow.dataset.spanId = span.spanId;
        barRow.style.top = (rowIndex * ROW_HEIGHT) + 'px';
        barRow.style.left = panelLeft + 'px';
        barRow.style.right = panelRight + 'px';

        const spanClass = getSpanClass(span.meta);
        const startPercent = ((span.startTime - vs) / vd) * 100;
        const widthPercent = (span.duration / vd) * 100;

        const bar = document.createElement('div');
        bar.className = `span-bar ${spanClass}`;
        bar.style.left = startPercent + '%';
        bar.style.width = widthPercent + '%';

        bar.addEventListener('click', () => selectSpan(span.spanId));
        bar.addEventListener('dblclick', (e) => {
            e.stopPropagation();
            zoomToSpan(span);
        });

        barRow.appendChild(bar);
        container.appendChild(barRow);
    });
}

function renderTimeAxis() {
    const axisEl = document.getElementById('canvasTimeAxis');
    if (!axisEl) return;
    axisEl.innerHTML = '';

    const vs = getViewStart();
    const vd = getViewDuration();
    const interval = niceTickInterval(vd);
    const panelLeft = spanListPanelWidth;
    const panelRight = detailOverlayWidth;

    // Style axis to span between panels
    axisEl.style.left = panelLeft + 'px';
    axisEl.style.right = panelRight + 'px';

    const firstTick = Math.ceil(vs / interval) * interval;

    for (let t = firstTick; t <= vs + vd; t += interval) {
        const pct = ((t - vs) / vd) * 100;
        if (pct < -1 || pct > 101) continue;

        const tick = document.createElement('div');
        tick.className = 'time-axis-tick';
        tick.style.left = pct + '%';
        tick.textContent = formatDuration(t - minTime);
        axisEl.appendChild(tick);
    }
}

function renderGridlines() {
    const gridEl = document.getElementById('canvasGrid');
    if (!gridEl) return;
    gridEl.innerHTML = '';

    const vs = getViewStart();
    const vd = getViewDuration();
    const interval = niceTickInterval(vd);
    const panelLeft = spanListPanelWidth;
    const panelRight = detailOverlayWidth;

    // Style grid to span between panels
    gridEl.style.left = panelLeft + 'px';
    gridEl.style.right = panelRight + 'px';

    const firstTick = Math.ceil(vs / interval) * interval;

    for (let t = firstTick; t <= vs + vd; t += interval) {
        const pct = ((t - vs) / vd) * 100;
        if (pct < -1 || pct > 101) continue;

        const line = document.createElement('div');
        line.className = 'canvas-grid-line';
        line.style.left = pct + '%';
        gridEl.appendChild(line);
    }
}

function syncScrollSetup() {
    const spanListScroll = document.getElementById('spanListScroll');
    const canvasBars = document.getElementById('canvasBars');
    if (!spanListScroll || !canvasBars) return;

    // Remove old listener to avoid duplicates
    spanListScroll.removeEventListener('scroll', onSpanListScroll);
    spanListScroll.addEventListener('scroll', onSpanListScroll);
}

function onSpanListScroll() {
    const spanListScroll = document.getElementById('spanListScroll');
    const canvasBars = document.getElementById('canvasBars');
    if (!spanListScroll || !canvasBars) return;

    const scrollTop = spanListScroll.scrollTop;
    canvasBars.style.transform = `translateY(${-scrollTop}px)`;
}

// ── Timeline viewport (continuous zoom/pan) ─────────────────

function getViewStart() { return viewStart ?? minTime; }
function getViewDuration() { return viewDuration ?? (totalDuration || 1); }

function updateBars() {
    const vs = getViewStart();
    const vd = getViewDuration();
    // Build lookup map for O(1) access
    const spanMap = new Map();
    spans.forEach(s => spanMap.set(s.spanId, s));

    // Update canvas bar rows (task view layered layout)
    document.querySelectorAll('.canvas-bar-row .span-bar').forEach(bar => {
        const row = bar.closest('.canvas-bar-row');
        const span = spanMap.get(row?.dataset.spanId);
        if (!span) return;
        bar.style.left = ((span.startTime - vs) / vd) * 100 + '%';
        bar.style.width = (span.duration / vd) * 100 + '%';
    });

    // Update old-style timeline rows (agent view)
    document.querySelectorAll('.timeline-row').forEach(row => {
        const span = spanMap.get(row.dataset.spanId);
        if (!span) return;
        const bar = row.querySelector('.span-bar');
        if (!bar) return;
        bar.style.left = ((span.startTime - vs) / vd) * 100 + '%';
        bar.style.width = (span.duration / vd) * 100 + '%';
    });

    // Update time axis and gridlines for task view
    if (document.getElementById('canvasTimeAxis')) {
        renderTimeAxis();
        renderGridlines();
        updatePanelShadows();
    }

    updateZoomBar();
}

let _lastZoomedSpanId = null;

function zoomToSpan(span) {
    // If already zoomed to this span, reset instead
    if (_lastZoomedSpanId === span.spanId) {
        _lastZoomedSpanId = null;
        resetZoom();
        return;
    }
    const padding = span.duration * 0.05 || 0.001;
    const targetStart = span.startTime - padding;
    const targetDuration = span.duration + padding * 2;
    _lastZoomedSpanId = span.spanId;
    animateViewport(targetStart, targetDuration);
}

function resetZoom() {
    _lastZoomedSpanId = null;
    animateViewport(null, null);
}

function animateViewport(targetStart, targetDuration) {
    const fromStart = getViewStart();
    const fromDuration = getViewDuration();
    const toStart = targetStart ?? minTime;
    const toDuration = targetDuration ?? (totalDuration || 1);
    const duration = 300; // ms
    const startTime = performance.now();

    function tick(now) {
        const t = Math.min((now - startTime) / duration, 1);
        const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        viewStart = fromStart + (toStart - fromStart) * ease;
        viewDuration = fromDuration + (toDuration - fromDuration) * ease;
        clampViewport();
        updateBars();
        updatePanelShadows();
        if (t < 1) requestAnimationFrame(tick);
        else {
            viewStart = targetStart;
            viewDuration = targetDuration;
            clampViewport();
            updateBars();
            updatePanelShadows();
        }
    }
    requestAnimationFrame(tick);
}

function updateZoomBar() {
    const isZoomed = viewStart !== null || viewDuration !== null;
    let btn = document.getElementById('zoomResetBtn');

    if (!btn) {
        // Create a small reset button in the header-right area
        const headerRight = document.querySelector('.header-right');
        if (!headerRight) return;
        btn = document.createElement('button');
        btn.id = 'zoomResetBtn';
        btn.className = 'zoom-reset-btn';
        btn.onclick = resetZoom;
        headerRight.insertBefore(btn, headerRight.firstChild);
    }

    if (isZoomed) {
        const vd = getViewDuration();
        btn.textContent = `Reset zoom (${formatDuration(vd)})`;
        btn.style.display = '';
    } else {
        btn.style.display = 'none';
    }
}

function initTimelineInteractions() {
    const panel = document.getElementById('timelinePanel');
    if (!panel) return;

    // Ctrl+Wheel → zoom centered on cursor
    panel.addEventListener('wheel', (e) => {
        if (!e.ctrlKey && !e.metaKey) return;
        e.preventDefault();

        const vs = getViewStart();
        const vd = getViewDuration();
        const globalDur = totalDuration || 1;

        // Find the bar container under cursor to get relative position
        const barContainer = e.target.closest('.span-bar-container');
        let cursorRatio = 0.5; // default: zoom centered
        if (barContainer) {
            const rect = barContainer.getBoundingClientRect();
            cursorRatio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        }

        const factor = e.deltaY > 0 ? 1.08 : 1 / 1.08;
        let newDuration = vd * factor;

        // Clamp: don't zoom out past global range, don't zoom in past 0.1ms
        newDuration = Math.max(0.0001, Math.min(globalDur * 1.1, newDuration));

        // Adjust start to keep cursor position stable
        const cursorTime = vs + vd * cursorRatio;
        let newStart = cursorTime - newDuration * cursorRatio;

        viewStart = newStart;
        viewDuration = newDuration;

        // If fully zoomed out, reset to null
        if (newDuration >= globalDur) {
            viewStart = null;
            viewDuration = null;
        }

        clampViewport();
        updateBars();
    }, { passive: false });

    // Drag to pan on bar containers
    let dragState = null;

    panel.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        const barContainer = e.target.closest('.span-bar-container');
        if (!barContainer) return;

        e.preventDefault();
        const rect = barContainer.getBoundingClientRect();
        dragState = {
            startX: e.clientX,
            containerWidth: rect.width,
            origViewStart: getViewStart(),
        };
        panel.style.cursor = 'grabbing';
    });

    window.addEventListener('mousemove', (e) => {
        if (!dragState) return;
        const dx = e.clientX - dragState.startX;
        const vd = getViewDuration();
        const timeDelta = -(dx / dragState.containerWidth) * vd;
        viewStart = dragState.origViewStart + timeDelta;
        if (viewDuration === null) viewDuration = totalDuration;
        clampViewport();
        updateBars();
    });

    window.addEventListener('mouseup', () => {
        if (dragState) {
            dragState = null;
            const panel = document.getElementById('timelinePanel');
            if (panel) panel.style.cursor = '';
        }
    });
}

// Canvas interactions for task view layered layout
let canvasDragState = null;

function onCanvasMouseMove(e) {
    if (!canvasDragState) return;
    const dx = e.clientX - canvasDragState.startX;
    const vd = getViewDuration();
    const timeDelta = -(dx / canvasDragState.containerWidth) * vd;
    viewStart = canvasDragState.origViewStart + timeDelta;
    if (viewDuration === null) viewDuration = totalDuration;
    clampViewport();
    updateBars();
    updatePanelShadows();
}

function onCanvasMouseUp() {
    if (canvasDragState) {
        canvasDragState = null;
        const canvas = document.getElementById('timelineCanvas');
        if (canvas) canvas.style.cursor = '';
    }
}

// Ensure global canvas drag handlers are registered once
let canvasGlobalHandlersRegistered = false;
function ensureCanvasGlobalHandlers() {
    if (canvasGlobalHandlersRegistered) return;
    canvasGlobalHandlersRegistered = true;
    window.addEventListener('mousemove', onCanvasMouseMove);
    window.addEventListener('mouseup', onCanvasMouseUp);
}

// Clamp viewport so it never goes before minTime or after minTime+totalDuration
function clampViewport() {
    const globalDur = totalDuration || 1;
    const vs = getViewStart();
    const vd = getViewDuration();

    // Don't clamp when fully zoomed out
    if (viewStart === null && viewDuration === null) return;

    let newStart = vs;
    let newDur = vd;

    // Clamp duration
    if (newDur > globalDur) newDur = globalDur;

    // Clamp start: never before minTime
    if (newStart < minTime) newStart = minTime;
    // Clamp end: never past minTime + globalDur
    if (newStart + newDur > minTime + globalDur) {
        newStart = minTime + globalDur - newDur;
    }
    // Edge case: after clamping end, start might be < minTime again
    if (newStart < minTime) newStart = minTime;

    viewStart = newStart;
    viewDuration = newDur;

    // If viewport matches full range, reset to null
    if (Math.abs(newStart - minTime) < 0.001 && Math.abs(newDur - globalDur) < 0.001) {
        viewStart = null;
        viewDuration = null;
    }
}

// Update shadow visibility based on viewport position
function updatePanelShadows() {
    const spanListPanel = document.getElementById('spanListPanel');
    const detailOverlay = document.getElementById('detailPanelOverlay');
    if (!spanListPanel || !detailOverlay) return;

    const vs = getViewStart();
    const vd = getViewDuration();
    const globalDur = totalDuration || 1;

    const atLeft = (vs <= minTime + 0.001);
    const atRight = (vs + vd >= minTime + globalDur - 0.001);

    spanListPanel.style.boxShadow = atLeft ? 'none' : '4px 0 12px var(--panel-shadow-color)';
    detailOverlay.style.boxShadow = atRight ? 'none' : '-4px 0 12px var(--panel-shadow-color)';
}

function initCanvasInteractions() {
    const canvas = document.getElementById('timelineCanvas');
    if (!canvas) return;

    ensureCanvasGlobalHandlers();

    // Wheel events on canvas area:
    // - Ctrl/Cmd + wheel Y → zoom
    // - Plain wheel Y → vertical scroll (sync with span list)
    // - Plain wheel X (trackpad two-finger horizontal) → horizontal pan
    canvas.addEventListener('wheel', (e) => {
        // Don't intercept events on the left or right panels (they scroll themselves)
        if (e.target.closest('.span-list-panel') || e.target.closest('.detail-panel-overlay')) return;

        const isZoom = e.ctrlKey || e.metaKey;

        if (isZoom) {
            // Zoom centered on cursor
            e.preventDefault();

            const vs = getViewStart();
            const vd = getViewDuration();
            const globalDur = totalDuration || 1;

            const canvasBarsEl = document.getElementById('canvasBars');
            let cursorRatio = 0.5;
            if (canvasBarsEl) {
                const rect = canvasBarsEl.getBoundingClientRect();
                cursorRatio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
            }

            const factor = e.deltaY > 0 ? 1.08 : 1 / 1.08;
            let newDuration = vd * factor;
            newDuration = Math.max(0.0001, Math.min(globalDur, newDuration));

            const cursorTime = vs + vd * cursorRatio;
            let newStart = cursorTime - newDuration * cursorRatio;

            viewStart = newStart;
            viewDuration = newDuration;
            clampViewport();
            updateBars();
            updatePanelShadows();
            return;
        }

        // Horizontal scroll (trackpad two-finger swipe / shift+wheel / mouse H-scroll)
        const hasHorizontal = Math.abs(e.deltaX) > Math.abs(e.deltaY) * 0.5 && Math.abs(e.deltaX) > 1;
        if (hasHorizontal) {
            e.preventDefault();

            const vs = getViewStart();
            const vd = getViewDuration();
            const globalDur = totalDuration || 1;

            // Convert pixel delta to time delta
            const canvasBarsEl = document.getElementById('canvasBars');
            const canvasPixelWidth = canvasBarsEl ? canvasBarsEl.getBoundingClientRect().width : 1;
            const timeDelta = (e.deltaX / canvasPixelWidth) * vd;

            viewStart = vs + timeDelta;
            if (viewDuration === null) viewDuration = globalDur;
            clampViewport();
            updateBars();
            updatePanelShadows();
            return;
        }

        // Vertical scroll → scroll the span list (and sync bars)
        if (Math.abs(e.deltaY) > 1) {
            e.preventDefault();
            const spanListScroll = document.getElementById('spanListScroll');
            if (spanListScroll) {
                spanListScroll.scrollTop += e.deltaY;
                // Trigger sync
                onSpanListScroll();
            }
        }
    }, { passive: false });

    // Drag to pan on canvas
    canvas.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        // Don't intercept clicks on bars or left/right panels
        if (e.target.closest('.span-list-panel') || e.target.closest('.detail-panel-overlay')) return;
        if (e.target.closest('.span-bar')) return;

        e.preventDefault();
        const canvasBarsEl = document.getElementById('canvasBars');
        const rect = canvasBarsEl ? canvasBarsEl.getBoundingClientRect() : { width: 1 };
        canvasDragState = {
            startX: e.clientX,
            containerWidth: rect.width,
            origViewStart: getViewStart(),
        };
        canvas.style.cursor = 'grabbing';
    });
}

// Dual resize handles for task view
function initTaskViewResize() {
    const spanListResize = document.getElementById('spanListResize');
    const detailResize = document.getElementById('detailResize');
    const spanListPanel = document.getElementById('spanListPanel');
    const detailPanelOverlay = document.getElementById('detailPanelOverlay');

    if (spanListResize && spanListPanel) {
        initDragResize(spanListResize, {
            getStartWidth: () => spanListPanel.getBoundingClientRect().width,
            onMove(dx, startWidth) {
                let newWidth = startWidth + dx;
                newWidth = Math.max(SPAN_LIST_MIN_WIDTH, Math.min(SPAN_LIST_MAX_WIDTH, newWidth));
                spanListPanelWidth = newWidth;
                spanListPanel.style.width = newWidth + 'px';
                updateCanvasBarPositions();
                renderTimeAxis();
                renderGridlines();
            },
        });
    }
    if (detailResize && detailPanelOverlay) {
        initDragResize(detailResize, {
            getStartWidth: () => detailPanelOverlay.getBoundingClientRect().width,
            onMove(dx, startWidth) {
                let newWidth = startWidth - dx;
                newWidth = Math.max(DETAIL_OVERLAY_MIN_WIDTH, Math.min(DETAIL_OVERLAY_MAX_WIDTH, newWidth));
                detailOverlayWidth = newWidth;
                detailPanelOverlay.style.width = newWidth + 'px';
                updateCanvasBarPositions();
                renderTimeAxis();
                renderGridlines();
            },
        });
    }
}


function updateCanvasBarPositions() {
    const panelLeft = spanListPanelWidth;
    const panelRight = detailOverlayWidth;

    document.querySelectorAll('.canvas-bar-row').forEach(row => {
        row.style.left = panelLeft + 'px';
        row.style.right = panelRight + 'px';
    });
}

function toggleChildren(event, parentSpanId) {
    if (collapsedSpans.has(parentSpanId)) {
        collapsedSpans.delete(parentSpanId);
    } else {
        collapsedSpans.add(parentSpanId);
        collapseAllDescendants(parentSpanId);
    }

    // In task view layered layout, rebuild the visible list
    if (document.querySelector('.container.task-view')) {
        renderTaskTimeline();
        return;
    }

    // Old-style: update visibility of child rows
    const panel = document.getElementById('timelinePanel');
    if (!panel) return;
    const row = event.currentTarget;
    const chevron = row ? row.querySelector('.toggle-chevron') : null;
    const childRows = panel.querySelectorAll(`[data-parent-span-id="${parentSpanId}"]`);
    const isCollapsed = collapsedSpans.has(parentSpanId);

    childRows.forEach(row => {
        row.style.display = isCollapsed ? 'none' : 'flex';
    });

    if (chevron) {
        chevron.classList.toggle('expanded', !isCollapsed);
        chevron.classList.toggle('collapsed', isCollapsed);
    }
}

function collapseAllDescendants(parentSpanId) {
    function collapseRecursive(pid) {
        traceData.getChildren(pid).forEach(child => {
            collapsedSpans.add(child.spanId);
            collapseRecursive(child.spanId);
        });
    }
    collapseRecursive(parentSpanId);
}

// Re-apply selection highlight on timeline DOM without re-rendering the detail panel
let _selectedElements = [];
function reapplySelectionHighlight(spanId) {
    _selectedElements.forEach(el => el.classList.remove('selected'));
    _selectedElements = [...document.querySelectorAll(`[data-span-id="${spanId}"]`)];
    _selectedElements.forEach(el => el.classList.add('selected'));
}

function selectSpan(spanId) {
    selectedSpanId = spanId;
    reapplySelectionHighlight(spanId);

    // Render details
    const span = traceData.getSpan(spanId);
    if (span) {
        renderDetails(span);
    }
}

// ── Detail section builders ──────────────────────────────────
// Each builder returns an HTML string for one detail section.
// Used by both renderDetails (full render) and updateDetailsIncremental.

function buildErrorSectionHtml(span) {
    if (!span.meta.error) return '';
    return `
        <div class="detail-section error-section" data-detail-section="error">
            <div class="section-header" style="color: var(--status-error); border-color: var(--badge-error-bg);">${ICON.warn} Error</div>
            <div class="code-block error-code">${escapeHtml(typeof span.meta.error === 'string' ? span.meta.error : JSON.stringify(span.meta.error, null, 2))}<div class="resize-handle"></div></div>
        </div>
    `;
}

function isStructuredContent(text) {
    const trimmed = text.trim();
    try { JSON.parse(trimmed); return true; } catch (e) {}
    if (/^\s*[\{\[]/.test(trimmed)) return true;
    if (/^[\w_-]+\s*:/m.test(trimmed) && !trimmed.includes('<')) return true;
    return false;
}

function highlightCode(code, lang) {
    if (typeof hljs !== 'undefined' && hljs.getLanguage(lang)) {
        try { return hljs.highlight(code, { language: lang }).value; } catch (e) {}
    }
    return escapeHtml(code);
}

function renderResponseText(text) {
    // Returns inner HTML only (no wrapper div)
    const trimmed = text.trim();
    try {
        const parsed = JSON.parse(trimmed);
        return highlightCode(JSON.stringify(parsed, null, 2), 'json');
    } catch (e) {}
    if (/^\s*[\{\[]/.test(trimmed)) return highlightCode(trimmed, 'json');
    if (/^[\w_-]+\s*:/m.test(trimmed) && !trimmed.includes('<')) return highlightCode(trimmed, 'yaml');
    return renderMarkdown(text);
}

function buildModelOutputSectionHtml(span) {
    if (!span.meta.model_output_meta) return '';
    const output = span.meta.model_output_meta;
    let html = '';

    // Response Content (with optional reasoning) — toggleable, default open
    const outputContent = output.content || output.choices?.[0]?.message?.content;
    if (outputContent || output.reasoning) {
        const msgOutId = `msg-out-${span.spanId}`;
        let innerHtml = '';

        // Reasoning shown as a collapsed sub-block within the response
        if (output.reasoning) {
            const reasoningId = `reasoning-${span.spanId}`;
            const renderedReasoning = renderResponseText(output.reasoning);
            innerHtml += `
                <div class="message-box" style="margin-bottom: 8px;">
                    <div class="message-header" onclick="toggleMessage('${reasoningId}')" style="cursor: pointer;">
                        <span>${ICON.reasoning} Reasoning</span>
                        <span class="toggle-icon" id="${reasoningId}-icon">▼</span>
                    </div>
                    <div class="message-content" id="${reasoningId}">
                        <div class="markdown-body">${renderedReasoning}</div>
                    </div>
                </div>
            `;
        }

        if (outputContent) {
            const structured = isStructuredContent(outputContent);
            const renderedContent = renderResponseText(outputContent);
            innerHtml += structured
                ? `<div class="code-block">${renderedContent}<div class="resize-handle"></div></div>`
                : `<div class="markdown-body">${renderedContent}</div>`;
        }

        html += `
            <div class="detail-section" data-detail-section="response-content">
                <div class="section-header clickable" onclick="toggleMessage('${msgOutId}')">
                    <span>${ICON.response} Response Content</span>
                    <span class="toggle-icon expanded" id="${msgOutId}-icon">▼</span>
                </div>
                <div class="message-content expanded full-height" id="${msgOutId}">
                    ${innerHtml}
                </div>
            </div>
        `;
    }

    // Tool Calls — toggleable, default open
    const outputToolCalls = output.tool_calls || output.choices?.[0]?.message?.tool_calls;
    if (outputToolCalls) {
        const toolCalls = outputToolCalls;
        const tcWrapperId = `tc-wrapper-${span.spanId}`;
        html += `
            <div class="detail-section" data-detail-section="tool-calls">
                <div class="section-header clickable" onclick="toggleMessage('${tcWrapperId}')">
                    <span>${ICON.tool} Tool Calls (${toolCalls.length})</span>
                    <span class="toggle-icon expanded" id="${tcWrapperId}-icon">▼</span>
                </div>
                <div class="message-content expanded full-height" id="${tcWrapperId}">
        `;
        toolCalls.forEach((tc, idx) => {
            const tcId = `tc-${span.spanId}-${idx}`;
            const tcArgs = tc.function?.arguments || '';

            html += `
                <div class="message-box">
                    <div class="message-header" onclick="toggleMessage('${tcId}')">
                        <span>${escapeHtml(tc.function?.name || 'tool')}</span>
                        <span class="toggle-icon" id="${tcId}-icon">▼</span>
                    </div>
                    <div class="message-content" id="${tcId}">
                        <div class="code-block">${highlightCode(tcArgs, 'json')}<div class="resize-handle"></div></div>
                    </div>
                </div>
            `;
        });
        html += `</div></div>`;
    }

    // Token usage — always at bottom, as a peer section
    if (output.usage) {
        const hasCost = span.tags && span.tags['model.cost_usd'] !== undefined;
        const gridCols = hasCost ? 'repeat(auto-fit, minmax(120px, 1fr))' : 'repeat(2, 1fr)';
        html += `
            <div class="detail-section" data-detail-section="token-usage">
                <div class="section-header">${ICON.metrics} Token Usage</div>
                <div class="token-grid" style="grid-template-columns: ${gridCols};">
                    <div class="token-card">
                        <div class="token-label">Total Tokens</div>
                        <div class="token-value">${output.usage.total_tokens || 0}</div>
                    </div>
                    <div class="token-card">
                        <div class="token-label">Prompt</div>
                        <div class="token-value">${output.usage.prompt_tokens || 0}</div>
                    </div>
                    <div class="token-card">
                        <div class="token-label">Completion</div>
                        <div class="token-value">${output.usage.completion_tokens || 0}</div>
                    </div>
        `;
        html += `
                    <div class="token-card">
                        <div class="token-label">Reasoning</div>
                        <div class="token-value">${output.usage.completion_tokens_details?.reasoning_tokens || 0}</div>
                    </div>
        `;
        if (hasCost) {
            const cost = span.tags['model.cost_usd'];
            html += `
                    <div class="token-card cost">
                        <div class="token-label">$ Cost (USD)</div>
                        <div class="token-value">$${cost.toFixed(6)}</div>
                    </div>
            `;
        }
        html += `</div></div>`;
    }

    return html;
}

function buildToolOutputSectionHtml(span) {
    if (!span.meta.tool_output_meta) return '';
    const toolOutContent = JSON.stringify(span.meta.tool_output_meta, null, 2);
    return `
        <div class="detail-section" data-detail-section="tool-output">
            <div class="section-header">${ICON.check} Tool Output</div>
            <div class="code-block">${highlightCode(toolOutContent, 'json')}<div class="resize-handle"></div></div>
        </div>
    `;
}

// ── renderDetails (full render) ──────────────────────────────

function renderDetails(span) {
    const panel = document.getElementById('detailPanel');
    const spanClass = getSpanClass(span.meta);
    const badgeClass = 'badge-' + spanClass;
    const displayName = getSpanDisplayName(span);

    let html = `
        <div class="detail-title">${escapeHtml(displayName)}</div>

        <div class="detail-section" data-detail-section="span-info">
            <div class="section-header">Span Information</div>
            <div class="detail-item">
                <div class="detail-label">Span ID</div>
                <div class="detail-value"><code>${span.spanId}</code></div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Parent Span</div>
                <div class="detail-value">${span.parentSpanId ? '<code>' + span.parentSpanId + '</code>' : 'None (root)'}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Duration</div>
                <div class="detail-value"><strong data-detail-duration>${formatDuration(span.duration)}</strong></div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Start Time</div>
                <div class="detail-value">${formatAbsoluteTime(span)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">End Time</div>
                <div class="detail-value">${formatAbsoluteTime(span, true)}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Type</div>
                <div class="detail-value"><span class="badge ${badgeClass}">${spanClass}</span></div>
            </div>
        </div>
    `;

    html += buildErrorSectionHtml(span);

    // Tool Schemas (tool_meta) - collapsible, folded by default
    if (span.meta.tool_meta && span.meta.tool_meta.length > 0) {
        const toolListId = `tool-list-${span.spanId}`;
        html += `
            <div class="detail-section" data-detail-section="tool-schemas">
                <div class="section-header clickable" onclick="toggleMessage('${toolListId}')">
                    <span>${ICON.tool} Available Tools (${span.meta.tool_meta.length})</span>
                    <span class="toggle-icon" id="${toolListId}-icon">▼</span>
                </div>
                <div class="message-content full-height" id="${toolListId}">
        `;
        span.meta.tool_meta.forEach((tool, idx) => {
            const toolId = `tool-schema-${span.spanId}-${idx}`;
            const toolName = tool.function?.name || 'unknown';
            const toolDesc = tool.function?.description || 'No description';
            html += `
                <div class="message-box">
                    <div class="message-header" onclick="toggleMessage('${toolId}')">
                        <span><strong>${toolName}</strong></span>
                        <span class="toggle-icon" id="${toolId}-icon">▼</span>
                    </div>
                    <div class="message-content full-height" id="${toolId}">
                        <div class="tool-desc-box">
                            <strong>Description:</strong> ${toolDesc}
                        </div>
                        <div style="margin-bottom: 6px;"><strong>Parameters Schema:</strong></div>
                        <div class="code-block">${highlightCode(JSON.stringify(tool.function?.parameters || {}, null, 2), 'json')}<div class="resize-handle"></div></div>
                    </div>
                </div>
            `;
        });
        html += `</div></div>`;
    }

    // Model Input — collapsed by default
    if (span.meta.model_input_meta && span.meta.model_input_meta.length > 0) {
        const modelInputWrapperId = `model-input-${span.spanId}`;
        html += `
            <div class="detail-section" data-detail-section="model-input">
                <div class="section-header clickable" onclick="toggleMessage('${modelInputWrapperId}')">
                    <span>${ICON.input} Model Input (${span.meta.model_input_meta.length} messages)</span>
                    <span class="toggle-icon" id="${modelInputWrapperId}-icon">▼</span>
                </div>
                <div class="message-content full-height" id="${modelInputWrapperId}">
        `;
        span.meta.model_input_meta.forEach((msg, idx) => {
            const msgId = `msg-in-${span.spanId}-${idx}`;

            // Build display content based on message type
            let displayContent = '';
            let headerLabel = msg.role || 'message';

            if (msg.role === 'tool') {
                headerLabel = `tool (${msg.name || 'unknown'})`;
                displayContent = msg.content || '(no content)';
            } else if (msg.role === 'assistant') {
                // Assistant: build sub-blocks for reasoning, content, tool calls
                if (msg.tool_calls && msg.tool_calls.length > 0) {
                    headerLabel = `assistant (${msg.tool_calls.length} tool calls)`;
                }
            } else {
                displayContent = msg.content || '(no content)';
            }

            // For assistant messages, render structured sub-blocks
            if (msg.role === 'assistant') {
                let innerHtml = '';

                if (msg.reasoning) {
                    const reasoningSubId = `${msgId}-reasoning`;
                    innerHtml += `
                        <div class="message-box" style="margin-bottom: 6px; background: #f8f5ff; border-left: 3px solid #8b5cf6;">
                            <div class="message-header" onclick="toggleMessage('${reasoningSubId}')" style="cursor: pointer;">
                                <span>🧠 Reasoning</span>
                                <span class="toggle-icon" id="${reasoningSubId}-icon">▼</span>
                            </div>
                            <div class="message-content" id="${reasoningSubId}">
                                <div class="code-block"><pre style="white-space: pre-wrap; margin: 0;">${escapeHtml(msg.reasoning)}</pre><div class="resize-handle"></div></div>
                            </div>
                        </div>
                    `;
                }

                if (msg.content) {
                    innerHtml += `<div class="code-block"><pre style="white-space: pre-wrap; margin: 0;">${escapeHtml(msg.content)}</pre><div class="resize-handle"></div></div>`;
                }

                if (msg.tool_calls && msg.tool_calls.length > 0) {
                    msg.tool_calls.forEach((tc, tcIdx) => {
                        const tcSubId = `${msgId}-tc-${tcIdx}`;
                        const tcName = tc.function?.name || tc.name || 'unknown';
                        const tcArgs = tc.function?.arguments || JSON.stringify(tc.arguments || {});
                        innerHtml += `
                            <div class="message-box" style="margin-top: 6px;">
                                <div class="message-header" onclick="toggleMessage('${tcSubId}')" style="cursor: pointer;">
                                    <span>🔧 ${escapeHtml(tcName)}</span>
                                    <span class="toggle-icon" id="${tcSubId}-icon">▼</span>
                                </div>
                                <div class="message-content" id="${tcSubId}">
                                    <div class="code-block"><pre style="white-space: pre-wrap; margin: 0;">${escapeHtml(tcArgs)}</pre><div class="resize-handle"></div></div>
                                </div>
                            </div>
                        `;
                    });
                }

                if (!innerHtml) innerHtml = '<pre style="margin: 0;">(no content)</pre>';

                html += `
                    <div class="message-box">
                        <div class="message-header" onclick="toggleMessage('${msgId}')">
                            <span>${headerLabel}</span>
                            <span class="toggle-icon" id="${msgId}-icon">▼</span>
                        </div>
                        <div class="message-content" id="${msgId}">
                            ${innerHtml}
                        </div>
                    </div>
                `;
            } else {
                html += `
                    <div class="message-box">
                        <div class="message-header" onclick="toggleMessage('${msgId}')">
                            <span>${headerLabel}</span>
                            <span class="toggle-icon" id="${msgId}-icon">▼</span>
                        </div>
                        <div class="message-content" id="${msgId}">
                            <div class="code-block"><pre style="white-space: pre-wrap; margin: 0;">${escapeHtml(displayContent)}</pre><div class="resize-handle"></div></div>
                        </div>
                    </div>
                `;
            }
        });
        html += `</div></div>`;
    }

    html += buildModelOutputSectionHtml(span);

    // Tool Input
    if (span.meta.tool_input_meta) {
        const toolInputs = Array.isArray(span.meta.tool_input_meta) ? span.meta.tool_input_meta : [span.meta.tool_input_meta];
        html += `
            <div class="detail-section" data-detail-section="tool-input">
                <div class="section-header">${ICON.tool} Tool Execution (${toolInputs.length} call${toolInputs.length > 1 ? 's' : ''})</div>
        `;
        toolInputs.forEach((tool, idx) => {
            const toolId = `tool-input-${span.spanId}-${idx}`;
            // Support both direct properties (from ToolExecuteTaskExtractor) and nested function properties
            const toolName = tool.name || tool.function?.name || 'unknown';
            const rawArgs = tool.arguments ?? tool.function?.arguments ?? '';
            const toolArgs = typeof rawArgs === 'object' ? JSON.stringify(rawArgs, null, 2) : rawArgs;

            html += `
                <div class="message-box">
                    <div class="message-header" onclick="toggleMessage('${toolId}')">
                        <span><strong>${toolName}</strong></span>
                        <span class="toggle-icon" id="${toolId}-icon">▼</span>
                    </div>
                    <div class="message-content" id="${toolId}">
                        <div style="margin-bottom: 6px;"><strong>Arguments:</strong></div>
                        <div class="code-block">${highlightCode(toolArgs, 'json')}<div class="resize-handle"></div></div>
                    </div>
                </div>
            `;
        });
        html += `</div>`;
    }

    html += buildToolOutputSectionHtml(span);

    panel.innerHTML = html;
    panel.dataset.spanId = span.spanId;

    // Restore expanded state for detail panel elements and init resize handles
    restoreDetailPanelState();
}

// ── Incremental detail update ────────────────────────────────
// When the same span is already rendered, only update duration and
// append sections that are newly available (e.g. model output arriving
// after model input). Existing DOM is left untouched — no scroll reset.

function updateDetailsIncremental(span) {
    const panel = document.getElementById('detailPanel');
    let appended = false;

    // 1. Update duration text
    const durationEl = panel.querySelector('[data-detail-duration]');
    if (durationEl) {
        durationEl.textContent = formatDuration(span.duration);
    }

    // 2. Append error section if newly appeared
    if (span.meta.error && !panel.querySelector('[data-detail-section="error"]')) {
        const html = buildErrorSectionHtml(span);
        if (html) {
            // Insert after span-info section
            const spanInfo = panel.querySelector('[data-detail-section="span-info"]');
            if (spanInfo) {
                spanInfo.insertAdjacentHTML('afterend', html);
            } else {
                panel.insertAdjacentHTML('beforeend', html);
            }
            appended = true;
        }
    }

    // 3. Append model output section if newly appeared
    if (span.meta.model_output_meta && !panel.querySelector('[data-detail-section="model-output"]')) {
        panel.insertAdjacentHTML('beforeend', buildModelOutputSectionHtml(span));
        appended = true;
    }

    // 4. Append tool output section if newly appeared
    if (span.meta.tool_output_meta && !panel.querySelector('[data-detail-section="tool-output"]')) {
        panel.insertAdjacentHTML('beforeend', buildToolOutputSectionHtml(span));
        appended = true;
    }

    if (appended) {
        restoreDetailPanelState();
    }
}

function restoreDetailPanelState() {
    // Restore expanded state for all tracked toggle elements
    expandedDetailElements.forEach(id => {
        const content = document.getElementById(id);
        const icon = document.getElementById(id + '-icon');
        if (content && icon) {
            content.classList.add('expanded');
            icon.classList.add('expanded');
        }
    });

    // Initialize resize handles for code blocks
    initResizableBlocks();
}

function toggleMessage(id) {
    const content = document.getElementById(id);
    const icon = document.getElementById(id + '-icon');
    if (content && icon) {
        const isExpanding = !content.classList.contains('expanded');
        content.classList.toggle('expanded');
        icon.classList.toggle('expanded');

        // Track expanded state for persistence across refreshes
        if (isExpanding) {
            expandedDetailElements.add(id);
        } else {
            expandedDetailElements.delete(id);
        }
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Drag-to-resize for code blocks
function initResizableBlocks() {
    document.querySelectorAll('.code-block .resize-handle').forEach(handle => {
        if (handle._resizeInit) return; // already initialized
        handle._resizeInit = true;

        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation();
            const block = handle.closest('.code-block');
            if (!block) return;

            const startY = e.clientY;
            const startHeight = block.offsetHeight;

            document.body.style.cursor = 'ns-resize';
            document.body.style.userSelect = 'none';

            function onMouseMove(e) {
                const dy = e.clientY - startY;
                const newHeight = Math.max(60, startHeight + dy);
                block.style.maxHeight = newHeight + 'px';
            }

            function onMouseUp() {
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
            }

            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
    });
}

// SSE live-update functionality
let autoRefreshEnabled = true;
let eventSource = null;

// Re-render the current view.
// updatedSpanId: if provided, only re-render the detail panel when the
// selected span is the one that was updated. This avoids unnecessary
// DOM destruction (and scroll position loss) in the detail panel when
// an unrelated span arrives via SSE.
function rerenderCurrentView(updatedSpanId) {
    const previouslySelectedSpanId = selectedSpanId;

    if (currentPage === 'traces') {
        if (currentView === 'task') {
            renderTaskTimeline();
        } else {
            renderAgentTimeline();
        }
    } else if (currentPage === 'spans') {
        renderSpansPage();
    } else if (currentPage === 'logs') {
        renderLogsPage();
    } else if (currentPage === 'metrics') {
        renderMetricsPage();
    } else if (currentPage === 'agents') {
        renderAgentsPage();
    }

    if (previouslySelectedSpanId) {
        const spanStillExists = spans.some(s => s.spanId === previouslySelectedSpanId);
        if (spanStillExists) {
            // Timeline DOM was just rebuilt, so re-apply the selection highlight
            reapplySelectionHighlight(previouslySelectedSpanId);

            if (!updatedSpanId) {
                // Full refresh (e.g. SSE init) — do a complete re-render
                const span = traceData.getSpan(previouslySelectedSpanId);
                if (span) {
                    renderDetails(span);
                }
            } else if (updatedSpanId === previouslySelectedSpanId) {
                // The selected span was updated — try incremental append
                const panel = document.getElementById('detailPanel');
                const span = traceData.getSpan(previouslySelectedSpanId);
                if (span && panel && panel.dataset.spanId === previouslySelectedSpanId) {
                    updateDetailsIncremental(span);
                } else if (span) {
                    renderDetails(span);
                }
            }
            // else: a different span was updated — detail panel untouched
        }
    }
}

function flashLiveIndicator() {
    const status = document.getElementById('liveStatus');
    if (!status) return;
    status.style.color = 'var(--status-connecting)';
    setTimeout(() => {
        if (autoRefreshEnabled) status.style.color = 'var(--status-ok)';
    }, 200);
}

function upsertSpan(span) {
    const idx = spans.findIndex(s => s.spanId === span.spanId);
    if (idx >= 0) {
        spans[idx] = span;
    } else {
        spans.push(span);
    }
    traceData.invalidate();
}

function recalcTimeBounds() {
    if (spans.length === 0) return;
    minTime = Math.min(...spans.map(s => s.startTime));
    const maxTime = Math.max(...spans.map(s => s.startTime + s.duration));
    totalDuration = (maxTime - minTime) || 1; // guard against division by zero
}

function showFinished() {
    if (eventSource) { eventSource.close(); eventSource = null; }
    autoRefreshEnabled = false;
    const status = document.getElementById('liveStatus');
    const btn = document.getElementById('toggleLive');
    if (status) {
        status.style.color = 'var(--status-ok)';
        status.textContent = 'FINISHED';
        status.classList.remove('live-status');
    }
    if (btn) btn.style.display = 'none';
}

let sseWasConnected = false;

function connectSSE() {
    if (window.location.protocol === 'file:') return; // offline mode
    if (eventSource) return; // already connected

    sseWasConnected = false;
    eventSource = new EventSource('/events');

    eventSource.addEventListener('init', (e) => {
        try {
            const data = JSON.parse(e.data);
            spans = data.spans;
            minTime = data.minTime;
            totalDuration = data.totalDuration || 1;
            traceData.invalidate();
            rerenderCurrentView();
            flashLiveIndicator();
        } catch (err) {
            console.error('SSE init error:', err);
        }
    });

    eventSource.addEventListener('span', (e) => {
        if (!autoRefreshEnabled) return;
        try {
            const span = JSON.parse(e.data);
            upsertSpan(span);
            recalcTimeBounds();
            rerenderCurrentView(span.spanId);
            flashLiveIndicator();
        } catch (err) {
            console.error('SSE span error:', err);
        }
    });

    eventSource.addEventListener('finish', () => { showFinished(); });

    eventSource.onerror = () => {
        // Let EventSource auto-reconnect on transient errors.
        // Only showFinished() is called from the explicit 'finish' event.
        const status = document.getElementById('liveStatus');
        if (status && autoRefreshEnabled) {
            status.style.color = 'var(--status-connecting)';
            status.textContent = 'RECONNECTING';
        }
    };

    eventSource.onopen = () => {
        sseWasConnected = true;
        const status = document.getElementById('liveStatus');
        if (status && autoRefreshEnabled) {
            status.style.color = 'var(--status-ok)';
            status.textContent = 'LIVE';
        }
    };
}

function toggleAutoRefresh() {
    autoRefreshEnabled = !autoRefreshEnabled;
    const btn = document.getElementById('toggleLive');
    const status = document.getElementById('liveStatus');

    if (autoRefreshEnabled) {
        btn.textContent = 'Pause';
        status.style.color = 'var(--status-ok)';
        status.textContent = 'LIVE';
        status.classList.add('live-status');
        connectSSE();
    } else {
        btn.textContent = 'Resume';
        status.style.color = 'var(--text-muted)';
        status.textContent = 'PAUSED';
        status.classList.remove('live-status');
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }
}

// Navigation between pages
function navigateToPage(page) {
    if (currentPage === page) return;
    currentPage = page;

    // Update nav item states
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
    });
    const activeNav = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (activeNav) activeNav.classList.add('active');

    // Update header title
    const headerTitle = document.querySelector('.header h1');
    const pageTitles = {
        'traces': 'Trace Viewer',
        'spans': 'All Spans',
        'logs': 'Logs & Errors',
        'metrics': 'Metrics Dashboard',
        'agents': 'Agents Overview'
    };
    if (headerTitle) headerTitle.textContent = pageTitles[page] || 'Trace Viewer';

    // Show/hide trace-only controls
    const viewToggle = document.querySelector('.view-toggle');
    const filterMagic = document.getElementById('filterMagicLabel');
    if (viewToggle) viewToggle.style.display = page === 'traces' ? 'flex' : 'none';
    if (filterMagic) filterMagic.style.display = page === 'traces' ? 'flex' : 'none';

    // Render the appropriate page
    switch (page) {
        case 'traces':
            renderTracesPage();
            break;
        case 'spans':
            renderSpansPage();
            break;
        case 'logs':
            renderLogsPage();
            break;
        case 'metrics':
            renderMetricsPage();
            break;
        case 'agents':
            renderAgentsPage();
            break;
    }

}

// Render Traces page (original timeline view)
function renderTracesPage() {
    const container = document.querySelector('.container');
    container.classList.remove('agent-view', 'single-panel', 'task-view');
    container.style.gridTemplateColumns = '';

    if (currentView === 'task') {
        container.classList.add('task-view');
        container.innerHTML = `
            <div class="timeline-canvas" id="timelineCanvas">
                <div class="canvas-grid" id="canvasGrid"></div>
                <div class="canvas-bars" id="canvasBars"></div>
                <div class="canvas-time-axis" id="canvasTimeAxis"></div>
            </div>
            <div class="span-list-panel" id="spanListPanel" style="width: ${spanListPanelWidth}px">
                <div class="span-list-scroll" id="spanListScroll"></div>
                <div class="span-list-resize" id="spanListResize"></div>
            </div>
            <div class="detail-panel-overlay" id="detailPanelOverlay" style="width: ${detailOverlayWidth}px">
                <div class="detail-resize" id="detailResize"></div>
                <div class="detail-content" id="detailPanel">
                    <div class="empty-state">
                        <div class="empty-state-icon">${ICON.magic}</div>
                        <div class="empty-state-text">Select a span to view details</div>
                    </div>
                </div>
            </div>
        `;
        initTaskViewResize();
        initCanvasInteractions();
        renderTaskTimeline();
    } else {
        // Agent view — use original flex layout with wrappers
        container.innerHTML = `
            <div class="timeline-panel-wrapper">
                <div class="timeline-panel" id="timelinePanel"></div>
                <div class="timeline-fade-right"></div>
            </div>
            <div class="detail-panel" id="detailPanel">
                <div class="empty-state">
                    <div class="empty-state-icon">${ICON.magic}</div>
                    <div class="empty-state-text">Select a span to view details</div>
                </div>
            </div>
        `;
        ensureDivider();
        initTimelineInteractions();
        renderAgentTimeline();
    }
}

// Render Spans page - flat searchable list
function renderSpansPage() {
    const container = document.querySelector('.container');
    container.classList.remove('agent-view', 'task-view');
    container.classList.add('single-panel');

    // Sort spans by start time
    const sortedSpans = [...spans]
        .sort((a, b) => a.startTime - b.startTime);

    let html = `
        <div class="page-panel">
            <div class="page-header">
                <div class="page-title">
                    <span class="page-icon">${ICON.spans}</span>
                    <span>All Spans</span>
                    <span class="page-count">${sortedSpans.length} spans</span>
                </div>
                <div class="page-search">
                    <input type="text" id="spanSearchInput" placeholder="Search spans..." onkeyup="filterSpans()">
                </div>
            </div>
            <div class="page-content">
                <table class="data-table" id="spansTable">
                    <thead>
                        <tr>
                            <th>Operation</th>
                            <th>Type</th>
                            <th>Duration</th>
                            <th>Start Time</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
    `;

    sortedSpans.forEach(span => {
        const spanClass = getSpanClass(span.meta);
        const displayName = getSpanDisplayName(span);
        const hasError = span.meta && span.meta.error;
        const relativeStart = span.startTime - minTime;

        html += `
            <tr class="data-row ${hasError ? 'error-row' : ''}" onclick="showSpanInPanel('${span.spanId}')">
                <td class="col-operation">
                    <span class="span-type-icon">${getSpanIcon(span.meta)}</span>
                    <span class="operation-name">${escapeHtml(displayName)}</span>
                </td>
                <td><span class="badge badge-${spanClass}">${spanClass}</span></td>
                <td class="col-duration">${formatDuration(span.duration)}</td>
                <td class="col-time">+${formatDuration(relativeStart)}</td>
                <td class="col-status">${hasError ? '<span class="status-error">' + ICON.warn + ' Error</span>' : '<span class="status-ok">' + ICON.check + ' OK</span>'}</td>
            </tr>
        `;
    });

    html += `
                    </tbody>
                </table>
            </div>
        </div>
        <div class="detail-panel" id="detailPanel">
            <div class="empty-state">
                <div class="empty-state-icon">${ICON.model}</div>
                <div class="empty-state-text">Select a span to view details</div>
            </div>
        </div>
    `;

    container.innerHTML = html;
    container.classList.remove('single-panel');
}

function filterSpans() {
    const input = document.getElementById('spanSearchInput');
    const filter = input.value.toLowerCase();
    const rows = document.querySelectorAll('#spansTable tbody tr');

    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(filter) ? '' : 'none';
    });
}

function showSpanInPanel(spanId) {
    const span = traceData.getSpan(spanId);
    if (span) {
        // Highlight row
        document.querySelectorAll('.data-row').forEach(r => r.classList.remove('selected'));
        const row = document.querySelector(`.data-row[onclick*="${spanId}"]`);
        if (row) row.classList.add('selected');

        renderDetails(span);
    }
}

// Render Logs page - errors and warnings
function renderLogsPage() {
    const container = document.querySelector('.container');
    container.classList.remove('agent-view', 'task-view');
    container.classList.add('single-panel');

    // Collect all errors from spans
    const logs = [];
    spans.forEach(span => {
        if (span.meta && span.meta.error) {
            logs.push({
                type: 'error',
                span: span,
                message: typeof span.meta.error === 'string' ? span.meta.error : JSON.stringify(span.meta.error),
                time: span.startTime
            });
        }
    });

    // Sort by time (most recent first)
    logs.sort((a, b) => b.time - a.time);

    let html = `
        <div class="page-panel full-width">
            <div class="page-header">
                <div class="page-title">
                    <span class="page-icon">${ICON.logs}</span>
                    <span>Logs & Errors</span>
                    <span class="page-count">${logs.length} entries</span>
                </div>
                <div class="log-filters">
                    <button class="filter-btn active" onclick="filterLogs('all', this)">All</button>
                    <button class="filter-btn" onclick="filterLogs('error', this)">Errors (${logs.filter(l => l.type === 'error').length})</button>
                </div>
            </div>
            <div class="page-content logs-content">
    `;

    if (logs.length === 0) {
        html += `
            <div class="empty-state">
                <div class="empty-state-icon">${ICON.check}</div>
                <div class="empty-state-text">No errors or warnings found</div>
                <div class="empty-state-subtext">All spans completed successfully</div>
            </div>
        `;
    } else {
        logs.forEach((log) => {
            const relativeTime = log.time - minTime;
            const displayName = getSpanDisplayName(log.span);

            html += `
                <div class="log-entry log-${log.type}" data-type="${log.type}">
                    <div class="log-header">
                        <span class="log-level">${log.type === 'error' ? ICON.error + ' ERROR' : ICON.warn + ' WARNING'}</span>
                        <span class="log-source">${escapeHtml(displayName)}</span>
                        <span class="log-time">+${formatDuration(relativeTime)}</span>
                    </div>
                    <div class="log-message">${escapeHtml(log.message.substring(0, 500))}${log.message.length > 500 ? '...' : ''}</div>
                    <div class="log-actions">
                        <button class="log-action-btn" onclick="navigateToPage('traces'); setTimeout(() => selectSpan('${log.span.spanId}'), 100);">View Span →</button>
                    </div>
                </div>
            `;
        });
    }

    html += `
            </div>
        </div>
    `;

    container.innerHTML = html;
}

function filterLogs(type, btn) {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    document.querySelectorAll('.log-entry').forEach(entry => {
        if (type === 'all' || entry.dataset.type === type) {
            entry.style.display = '';
        } else {
            entry.style.display = 'none';
        }
    });
}

// Chart helper functions
function generatePieChart(data, colors, size = 120) {
    if (Object.keys(data).length === 0) return '';

    const total = Object.values(data).reduce((a, b) => a + b, 0);
    if (total === 0) return '';

    const cx = size / 2;
    const cy = size / 2;
    const radius = size / 2 - 5;

    let paths = '';
    let startAngle = -90; // Start from top
    let legendItems = '';

    Object.entries(data).forEach(([label, value], idx) => {
        const percentage = value / total;
        const angle = percentage * 360;
        const endAngle = startAngle + angle;

        const startRad = (startAngle * Math.PI) / 180;
        const endRad = (endAngle * Math.PI) / 180;

        const x1 = cx + radius * Math.cos(startRad);
        const y1 = cy + radius * Math.sin(startRad);
        const x2 = cx + radius * Math.cos(endRad);
        const y2 = cy + radius * Math.sin(endRad);

        const largeArc = angle > 180 ? 1 : 0;
        const color = Array.isArray(colors) ? (colors[idx % colors.length]) : (colors[label] || colors.default || '#6b7280');

        paths += `<path d="M ${cx} ${cy} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} Z" fill="${color}" stroke="var(--pie-stroke)" stroke-width="2"/>`;

        legendItems += `
            <div class="chart-legend-item">
                <span class="chart-legend-color" style="background: ${color}"></span>
                <span class="chart-legend-label">${label}</span>
                <span class="chart-legend-value">${(percentage * 100).toFixed(1)}%</span>
            </div>
        `;

        startAngle = endAngle;
    });

    return `
        <div class="chart-container">
            <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
                ${paths}
            </svg>
            <div class="chart-legend">${legendItems}</div>
        </div>
    `;
}

function generateBarChart(data, colors, maxHeight = 100) {
    if (Object.keys(data).length === 0) return '';

    const maxValue = Math.max(...Object.values(data), 1);
    const barWidth = 60;
    const gap = 20;
    const entries = Object.entries(data);
    const width = entries.length * (barWidth + gap);

    let bars = '';
    entries.forEach(([label, value], idx) => {
        const height = (value / maxValue) * maxHeight;
        const x = idx * (barWidth + gap) + gap / 2;
        const y = maxHeight - height;
        const color = colors[label] || colors.default || '#3b82f6';

        bars += `
            <g class="bar-group">
                <rect x="${x}" y="${y}" width="${barWidth}" height="${height}" fill="${color}" rx="4"/>
                <text x="${x + barWidth / 2}" y="${maxHeight + 16}" text-anchor="middle" class="bar-label">${label}</text>
                <text x="${x + barWidth / 2}" y="${y - 5}" text-anchor="middle" class="bar-value">${value.toLocaleString()}</text>
            </g>
        `;
    });

    return `
        <div class="bar-chart-container">
            <svg width="${width}" height="${maxHeight + 30}" viewBox="0 0 ${width} ${maxHeight + 30}">
                ${bars}
            </svg>
        </div>
    `;
}

function generateHorizontalBarChart(data, colors) {
    if (Object.keys(data).length === 0) return '<div class="metric-list-empty">No data</div>';

    const maxValue = Math.max(...Object.values(data), 1);
    const entries = Object.entries(data);

    let bars = '';
    entries.forEach(([label, value]) => {
        const color = colors[label] || colors.default || '#3b82f6';

        bars += `
            <div class="h-bar-row">
                <div class="h-bar-label">${escapeHtml(label)}</div>
                <div class="h-bar-track">
                    <div class="h-bar-fill" style="width: ${(value / maxValue) * 100}%; background: ${color}"></div>
                </div>
                <div class="h-bar-value">${value.toLocaleString()}</div>
            </div>
        `;
    });

    return `<div class="h-bar-chart">${bars}</div>`;
}

// Aggregate cost and call data by model
function aggregateByModel(spans) {
    const models = {};

    spans.forEach(span => {
        if (isModelSpan(span.meta)) {
            const modelName = span.tags?.['model.name'] || 'unknown';
            if (!models[modelName]) {
                models[modelName] = {
                    calls: 0,
                    totalCost: 0,
                    totalTokens: 0,
                    promptTokens: 0,
                    completionTokens: 0,
                    totalDuration: 0
                };
            }
            models[modelName].calls++;
            models[modelName].totalCost += span.tags?.['model.cost_usd'] || 0;
            models[modelName].totalTokens += span.tags?.['model.tokens.total'] || 0;
            models[modelName].promptTokens += span.tags?.['model.tokens.prompt'] || 0;
            models[modelName].completionTokens += span.tags?.['model.tokens.completion'] || 0;
            models[modelName].totalDuration += span.duration || 0;
        }
    });

    return models;
}

// Aggregate data by agent
function aggregateByAgent(spans) {
    const agents = {};

    // First pass: identify all agents from AgentBase._execute spans
    spans.forEach(span => {
        if (isAgentSpan(span.meta) && span.meta?.agent_id) {
            const agentId = span.meta.agent_id;
            if (!agents[agentId]) {
                agents[agentId] = {
                    agentId: agentId,
                    totalCost: 0,
                    totalDuration: 0,
                    modelCalls: 0,
                    toolCalls: 0,
                    spans: []
                };
            }
            agents[agentId].totalDuration += span.duration || 0;
            agents[agentId].spans.push(span);
        }
    });

    // Second pass: attribute model/tool calls to agents based on parent hierarchy
    function findAgentForSpan(span) {
        let current = span;
        while (current) {
            if (isAgentSpan(current.meta) && current.meta?.agent_id) {
                return current.meta.agent_id;
            }
            if (current.parentSpanId) {
                current = traceData.getSpan(current.parentSpanId);
            } else {
                break;
            }
        }
        return null;
    }

    spans.forEach(span => {
        const agentId = findAgentForSpan(span);

        if (agentId && agents[agentId]) {
            if (isModelSpan(span.meta)) {
                agents[agentId].modelCalls++;
                agents[agentId].totalCost += span.tags?.['model.cost_usd'] || 0;
            } else if (isToolSpan(span.meta)) {
                agents[agentId].toolCalls++;
            }
        }
    });

    return agents;
}

// Calculate latency breakdown
function calculateLatencyBreakdown(spans) {
    let modelLatency = 0;
    let toolLatency = 0;
    const modelDurations = [];
    const toolDurations = [];

    spans.forEach(span => {
        if (isModelSpan(span.meta)) {
            modelLatency += span.duration || 0;
            modelDurations.push(span.duration || 0);
        } else if (isToolSpan(span.meta)) {
            toolLatency += span.duration || 0;
            toolDurations.push(span.duration || 0);
        }
    });

    // Calculate percentiles
    const calcPercentile = (arr, p) => {
        if (arr.length === 0) return 0;
        const sorted = [...arr].sort((a, b) => a - b);
        const idx = Math.ceil((p / 100) * sorted.length) - 1;
        return sorted[Math.max(0, idx)];
    };

    const otherLatency = Math.max(0, totalDuration - modelLatency - toolLatency);

    return {
        total: totalDuration,
        model: modelLatency,
        tool: toolLatency,
        other: otherLatency,
        modelPercent: totalDuration > 0 ? (modelLatency / totalDuration * 100).toFixed(1) : 0,
        toolPercent: totalDuration > 0 ? (toolLatency / totalDuration * 100).toFixed(1) : 0,
        otherPercent: totalDuration > 0 ? (otherLatency / totalDuration * 100).toFixed(1) : 0,
        modelP50: calcPercentile(modelDurations, 50),
        modelP95: calcPercentile(modelDurations, 95),
        modelP99: calcPercentile(modelDurations, 99),
        toolP50: calcPercentile(toolDurations, 50),
        toolP95: calcPercentile(toolDurations, 95),
        toolP99: calcPercentile(toolDurations, 99),
        slowestOps: [...spans]
            .sort((a, b) => (b.duration || 0) - (a.duration || 0))
            .slice(0, 5)
            .map(s => ({ name: s.operationName, duration: s.duration || 0 }))
    };
}

// Render Cost Breakdown tab content
function renderCostBreakdownTab() {
    const modelData = aggregateByModel(spans);
    const modelNames = Object.keys(modelData);

    // Calculate totals
    let totalCost = 0;
    let totalCalls = 0;
    modelNames.forEach(name => {
        totalCost += modelData[name].totalCost;
        totalCalls += modelData[name].calls;
    });

    // Summary cards
    const summaryCards = `
        <div class="cost-summary-grid">
            <div class="metric-card">
                <div class="metric-header">Total Cost</div>
                <div class="metric-value highlight-green">$${totalCost.toFixed(5)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-header">Total Model Calls</div>
                <div class="metric-value">${totalCalls}</div>
            </div>
            <div class="metric-card">
                <div class="metric-header">Avg Cost per Call</div>
                <div class="metric-value highlight-green">$${totalCalls > 0 ? (totalCost / totalCalls).toFixed(6) : '0.000000'}</div>
            </div>
        </div>
    `;

    // Cost by model bar chart
    const costByModel = {};
    const callsByModel = {};
    modelNames.forEach(name => {
        costByModel[name] = modelData[name].totalCost;
        callsByModel[name] = modelData[name].calls;
    });

    const modelColors = {
        'gpt-4': '#10b981',
        'gpt-4-turbo': '#059669',
        'gpt-3.5-turbo': '#34d399',
        'claude-3-opus': '#8b5cf6',
        'claude-3-sonnet': '#a78bfa',
        'claude-3-haiku': '#c4b5fd',
        'default': '#6b7280'
    };

    const costByModelChart = modelNames.length > 0 ? `
        <div class="metric-card large">
            <div class="metric-header">Cost by Model</div>
            ${generateHorizontalBarChart(costByModel, modelColors)}
        </div>
    ` : '';

    // Cost per call by model
    const costPerCallCards = modelNames.length > 0 ? `
        <div class="metric-card large">
            <div class="metric-header">Cost per Call by Model</div>
            <div class="metric-list">
                ${modelNames.map(name => {
                    const avgCost = modelData[name].calls > 0 ? modelData[name].totalCost / modelData[name].calls : 0;
                    return `
                        <div class="metric-list-item">
                            <span class="model-name">${escapeHtml(name)}</span>
                            <span class="metric-list-value">$${avgCost.toFixed(6)} / call</span>
                        </div>
                    `;
                }).join('')}
            </div>
        </div>
    ` : '';

    // Token cost analysis table
    const tokenCostTable = modelNames.length > 0 ? `
        <div class="metric-card large">
            <div class="metric-header">Token Cost Analysis</div>
            <table class="token-cost-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Prompt Tokens</th>
                        <th>Completion Tokens</th>
                        <th>Total Tokens</th>
                        <th>Cost/1K Tokens</th>
                    </tr>
                </thead>
                <tbody>
                    ${modelNames.map(name => {
                        const data = modelData[name];
                        const costPer1k = data.totalTokens > 0 ? (data.totalCost / data.totalTokens * 1000) : 0;
                        return `
                            <tr>
                                <td class="model-col">${escapeHtml(name)}</td>
                                <td class="number-col">${data.promptTokens.toLocaleString()}</td>
                                <td class="number-col">${data.completionTokens.toLocaleString()}</td>
                                <td class="number-col">${data.totalTokens.toLocaleString()}</td>
                                <td class="number-col">$${costPer1k.toFixed(5)}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    ` : '';

    // Model call distribution pie chart
    const pieColors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#6b7280'];
    const callDistributionChart = modelNames.length > 0 ? `
        <div class="metric-card">
            <div class="metric-header">Model Call Distribution</div>
            ${generatePieChart(callsByModel, pieColors, 120)}
        </div>
    ` : '';

    if (modelNames.length === 0) {
        return `
            ${summaryCards}
            <div class="metric-card large">
                <div class="metric-header">No Model Calls</div>
                <div class="metric-list-empty">No model calls recorded in this trace.</div>
            </div>
        `;
    }

    return `
        ${summaryCards}
        <div class="metrics-grid">
            ${costByModelChart}
            ${costPerCallCards}
            ${tokenCostTable}
            ${callDistributionChart}
        </div>
    `;
}

// Render Latency Breakdown tab content
function renderLatencyBreakdownTab() {
    const latency = calculateLatencyBreakdown(spans);

    // Overview stats
    const statsCards = `
        <div class="latency-stats-grid">
            <div class="latency-stat-card">
                <div class="latency-stat-value">${formatDuration(latency.total)}</div>
                <div class="latency-stat-label">Total Duration</div>
            </div>
            <div class="latency-stat-card">
                <div class="latency-stat-value color-green">${formatDuration(latency.model)}</div>
                <div class="latency-stat-label">Model Time</div>
                <div class="latency-stat-percent">${latency.modelPercent}%</div>
            </div>
            <div class="latency-stat-card">
                <div class="latency-stat-value color-purple">${formatDuration(latency.tool)}</div>
                <div class="latency-stat-label">Tool Time</div>
                <div class="latency-stat-percent">${latency.toolPercent}%</div>
            </div>
            <div class="latency-stat-card">
                <div class="latency-stat-value color-muted">${formatDuration(latency.other)}</div>
                <div class="latency-stat-label">Other/Overhead</div>
                <div class="latency-stat-percent">${latency.otherPercent}%</div>
            </div>
        </div>
    `;

    // Pie chart for latency distribution
    const pieData = {
        'Model': latency.model,
        'Tool': latency.tool,
        'Other': latency.other
    };
    const pieColors = {
        'Model': '#10b981',
        'Tool': '#8b5cf6',
        'Other': '#6b7280'
    };

    const pieChart = `
        <div class="metric-card">
            <div class="metric-header">Latency Distribution</div>
            ${generatePieChart(pieData, pieColors, 140)}
        </div>
    `;

    // Percentiles
    const percentiles = `
        <div class="metric-card">
            <div class="metric-header">Latency Percentiles</div>
            <div class="percentile-grid">
                <div class="percentile-section">
                    <div class="percentile-section-title">Model Calls</div>
                    <div class="percentile-row">
                        <span class="percentile-label">P50</span>
                        <span class="percentile-value">${formatDuration(latency.modelP50)}</span>
                    </div>
                    <div class="percentile-row">
                        <span class="percentile-label">P95</span>
                        <span class="percentile-value">${formatDuration(latency.modelP95)}</span>
                    </div>
                    <div class="percentile-row">
                        <span class="percentile-label">P99</span>
                        <span class="percentile-value">${formatDuration(latency.modelP99)}</span>
                    </div>
                </div>
                <div class="percentile-section">
                    <div class="percentile-section-title">Tool Calls</div>
                    <div class="percentile-row">
                        <span class="percentile-label">P50</span>
                        <span class="percentile-value">${formatDuration(latency.toolP50)}</span>
                    </div>
                    <div class="percentile-row">
                        <span class="percentile-label">P95</span>
                        <span class="percentile-value">${formatDuration(latency.toolP95)}</span>
                    </div>
                    <div class="percentile-row">
                        <span class="percentile-label">P99</span>
                        <span class="percentile-value">${formatDuration(latency.toolP99)}</span>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Slowest operations
    const slowestOps = `
        <div class="metric-card large">
            <div class="metric-header">Slowest Operations (Top 5)</div>
            <div class="slowest-ops-list">
                ${latency.slowestOps.map(op => `
                    <div class="slowest-op-item">
                        <span class="slowest-op-name">${escapeHtml(op.name)}</span>
                        <span class="slowest-op-duration">${formatDuration(op.duration)}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;

    return `
        ${statsCards}
        <div class="metrics-grid">
            ${pieChart}
            ${percentiles}
            ${slowestOps}
        </div>
    `;
}

// Render Agent Breakdown tab content
function renderAgentBreakdownTab() {
    const agentData = aggregateByAgent(spans);
    const agentIds = Object.keys(agentData);

    if (agentIds.length === 0) {
        return `
            <div class="metric-card large">
                <div class="metric-header">No Agents Found</div>
                <div class="metric-list-empty">No agent execution spans found in this trace.</div>
            </div>
        `;
    }

    // Find max values for relative bars
    let maxCost = 0;
    let maxDuration = 0;
    agentIds.forEach(id => {
        maxCost = Math.max(maxCost, agentData[id].totalCost);
        maxDuration = Math.max(maxDuration, agentData[id].totalDuration);
    });

    // Sort by cost descending
    agentIds.sort((a, b) => agentData[b].totalCost - agentData[a].totalCost);

    const header = `
        <div class="agent-breakdown-header">
            <span>Agent ID</span>
            <span>Cost</span>
            <span>Duration</span>
            <span>Model</span>
            <span>Tool</span>
            <span>Avg/Call</span>
        </div>
    `;

    const rows = agentIds.map(id => {
        const agent = agentData[id];
        const avgCostPerCall = agent.modelCalls > 0 ? agent.totalCost / agent.modelCalls : 0;
        const durationBarWidth = maxDuration > 0 ? (agent.totalDuration / maxDuration * 100) : 0;

        return `
            <div class="agent-breakdown-row">
                <div>
                    <div class="agent-breakdown-name">${escapeHtml(agent.agentId)}</div>
                    <div class="agent-breakdown-bar">
                        <div class="agent-breakdown-bar-fill duration" style="width: ${durationBarWidth}%"></div>
                    </div>
                </div>
                <div class="agent-breakdown-value ${agent.totalCost > 0 ? 'cost' : 'muted'}">
                    $${agent.totalCost.toFixed(5)}
                </div>
                <div class="agent-breakdown-value">
                    ${formatDuration(agent.totalDuration)}
                </div>
                <div class="agent-breakdown-value ${agent.modelCalls > 0 ? '' : 'muted'}">
                    ${agent.modelCalls}
                </div>
                <div class="agent-breakdown-value ${agent.toolCalls > 0 ? '' : 'muted'}">
                    ${agent.toolCalls}
                </div>
                <div class="agent-breakdown-value ${avgCostPerCall > 0 ? 'cost' : 'muted'}">
                    $${avgCostPerCall.toFixed(6)}
                </div>
            </div>
        `;
    }).join('');

    return `
        <div class="agent-breakdown-grid">
            ${header}
            ${rows}
        </div>
    `;
}

// Render the current metrics tab content
function renderMetricsTabContent() {
    switch (currentMetricsTab) {
        case 'cost':
            return renderCostBreakdownTab();
        case 'latency':
            return renderLatencyBreakdownTab();
        case 'agent':
            return renderAgentBreakdownTab();
        default:
            return renderCostBreakdownTab();
    }
}

// Render Metrics page - aggregated statistics
function renderMetricsPage() {
    const container = document.querySelector('.container');
    container.classList.remove('agent-view', 'task-view');
    container.classList.add('single-panel');

    const html = `
        <div class="page-panel full-width">
            <div class="page-header">
                <div class="page-title">
                    <span class="page-icon">${ICON.metrics}</span>
                    <span>Metrics Dashboard</span>
                </div>
                <div class="metrics-tabs">
                    <button class="metrics-tab-btn ${currentMetricsTab === 'cost' ? 'active' : ''}"
                            data-tab="cost" onclick="switchMetricsTab('cost')">Cost Breakdown</button>
                    <button class="metrics-tab-btn ${currentMetricsTab === 'latency' ? 'active' : ''}"
                            data-tab="latency" onclick="switchMetricsTab('latency')">Latency Breakdown</button>
                    <button class="metrics-tab-btn ${currentMetricsTab === 'agent' ? 'active' : ''}"
                            data-tab="agent" onclick="switchMetricsTab('agent')">Agent Breakdown</button>
                </div>
            </div>
            <div class="page-content metrics-content">
                ${renderMetricsTabContent()}
            </div>
        </div>
    `;

    container.innerHTML = html;
}

// Render Agents page - agent overview
function renderAgentsPage() {
    const container = document.querySelector('.container');
    container.classList.remove('agent-view', 'task-view');
    container.classList.add('single-panel');

    // Group by unique agents
    const agentMap = new Map();

    spans.forEach(span => {
        if (isAgentSpan(span.meta) && span.meta?.agent_id) {
            const agentId = span.meta.agent_id;
            if (!agentMap.has(agentId)) {
                agentMap.set(agentId, {
                    agentId: agentId,
                    executions: [],
                    totalDuration: 0,
                    totalCost: 0,
                    totalTokens: 0,
                    modelCalls: 0,
                    toolCalls: 0,
                    errors: 0
                });
            }
            const agent = agentMap.get(agentId);
            agent.executions.push(span);
            agent.totalDuration += span.duration;
            if (span.meta?.error) agent.errors++;
        }
    });

    // Calculate nested stats for each agent
    agentMap.forEach((agent) => {
        agent.executions.forEach(exec => {
            collectAgentStats(exec, agent);
        });
    });

    const agents = Array.from(agentMap.values());
    agents.sort((a, b) => b.executions.length - a.executions.length);

    let html = `
        <div class="page-panel full-width">
            <div class="page-header">
                <div class="page-title">
                    <span class="page-icon">${ICON.agents}</span>
                    <span>Agents Overview</span>
                    <span class="page-count">${agents.length} agents</span>
                </div>
            </div>
            <div class="page-content agents-content">
    `;

    if (agents.length === 0) {
        html += `
            <div class="empty-state">
                <div class="empty-state-icon">${ICON.agents}</div>
                <div class="empty-state-text">No agents found</div>
                <div class="empty-state-subtext">Agent data appears when spans have agent_id tags</div>
            </div>
        `;
    } else {
        html += `<div class="agents-grid">`;
        agents.forEach(agent => {
            const avgDuration = agent.totalDuration / agent.executions.length;
            const hasErrors = agent.errors > 0;

            html += `
                <div class="agent-card ${hasErrors ? 'has-error' : ''}" onclick="navigateToPage('traces'); switchView('agent');">
                    <div class="agent-card-header">
                        <div class="agent-card-icon">${ICON.agent}</div>
                        <div class="agent-card-title">${escapeHtml(agent.agentId)}</div>
                        ${hasErrors ? '<span class="agent-error-badge">' + ICON.warn + ' ' + agent.errors + '</span>' : ''}
                    </div>
                    <div class="agent-card-stats">
                        <div class="agent-stat">
                            <div class="agent-stat-value">${agent.executions.length}</div>
                            <div class="agent-stat-label">Executions</div>
                        </div>
                        <div class="agent-stat">
                            <div class="agent-stat-value">${agent.modelCalls}</div>
                            <div class="agent-stat-label">LLM Calls</div>
                        </div>
                        <div class="agent-stat">
                            <div class="agent-stat-value">${agent.toolCalls}</div>
                            <div class="agent-stat-label">Tool Calls</div>
                        </div>
                        <div class="agent-stat">
                            <div class="agent-stat-value">${formatDuration(avgDuration)}</div>
                            <div class="agent-stat-label">Avg Duration</div>
                        </div>
                    </div>
                    <div class="agent-card-footer">
                        <span class="agent-tokens">${agent.totalTokens.toLocaleString()} tokens</span>
                        <span class="agent-cost">$${agent.totalCost.toFixed(4)}</span>
                    </div>
                </div>
            `;
        });
        html += `</div>`;
    }

    html += `
            </div>
        </div>
    `;

    container.innerHTML = html;
}

function collectAgentStats(parentSpan, agent) {
    traceData.getChildren(parentSpan.spanId).forEach(child => {
        if (isModelSpan(child.meta)) {
            agent.modelCalls++;
            if (child.tags) {
                if (child.tags['model.cost_usd']) agent.totalCost += child.tags['model.cost_usd'];
                if (child.tags['model.tokens.total']) agent.totalTokens += child.tags['model.tokens.total'];
            }
        }
        if (isToolSpan(child.meta)) {
            agent.toolCalls++;
        }
        if (child.meta?.error) agent.errors++;
        collectAgentStats(child, agent);
    });
}

// Populate nav icons from ICON object
document.querySelectorAll('.nav-item[data-page]').forEach(item => {
    const page = item.dataset.page;
    const iconEl = item.querySelector('.nav-icon');
    if (iconEl && ICON[page]) iconEl.innerHTML = ICON[page];
});

// Initialize
renderTracesPage();

// Connect SSE if live mode is enabled
if (autoRefreshEnabled) {
    connectSSE();
}
