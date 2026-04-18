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
const collapsedSpans = new Set();

// Timeline viewport state (continuous zoom/pan)
let viewStart = null;     // left edge of visible time range (null = minTime)
let viewDuration = null;  // visible time range width (null = totalDuration)
let hideMagicTasks = true; // Filter out magic_task spans by default
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



function toggleMagicTaskFilter() {
    hideMagicTasks = document.getElementById('filterMagicTasks')?.checked ?? true;
    renderTaskTimeline();
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

    // Update time axis and gridlines
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

    renderTaskTimeline();
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

    renderTaskTimeline();

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

// Render the trace viewer (task view with span tree + timeline)
function renderTracesPage() {
    const container = document.querySelector('.container');
    container.classList.remove('agent-view', 'single-panel');
    container.classList.add('task-view');
    container.style.gridTemplateColumns = '';

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
