const colorScale = ["#4ad7d1", "#ff8a5b", "#9bff6d", "#7aa6ff", "#ffd166", "#ff6577"];

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderMetricCards(data) {
  const container = document.getElementById("hero-metrics");
  const template = document.getElementById("metric-card-template");
  const metrics = [
    ["Training rows", data.overview.train_rows.toLocaleString()],
    ["Testing rows", data.overview.test_rows.toLocaleString()],
    ["Features used", data.overview.feature_count],
    ["Best accuracy", percent(data.overview.best_accuracy)],
  ];

  metrics.forEach(([label, value]) => {
    const node = template.content.cloneNode(true);
    node.querySelector(".metric-label").textContent = label;
    node.querySelector(".metric-value").textContent = value;
    container.appendChild(node);
  });
}

function renderBars(containerId, items, valueKey, formatter = (value) => value) {
  const root = document.getElementById(containerId);
  root.innerHTML = "";
  const maxValue = Math.max(...items.map((item) => item[valueKey]), 1);

  items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = item.name || item.label || item.protocol || item.feature;
    const width = (item[valueKey] / maxValue) * 100;

    row.innerHTML = `
      <span class="bar-label">${label}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${width}%; background:linear-gradient(90deg, ${colorScale[index % colorScale.length]}, #ffffff22);"></div>
      </div>
      <span class="bar-value">${formatter(item[valueKey])}</span>
    `;

    root.appendChild(row);
  });
}

function renderAttackDonut(data) {
  const attackShare = data.overview.attack_share_test;
  const attackPercent = attackShare * 100;
  const donut = document.getElementById("attack-donut");

  donut.style.background =
    `conic-gradient(#ff6577 0 ${attackPercent}%, rgba(255,255,255,0.08) ${attackPercent}% 100%)`;

  document.getElementById("attack-share").textContent = percent(attackShare);

  const legend = document.getElementById("attack-legend");
  legend.innerHTML = "";

  [
    {
      label: "Attack",
      value: data.overview.attack_share_test,
      color: "#ff6577"
    },
    {
      label: "Normal",
      value: data.overview.normal_share_test,
      color: "#4ad7d1"
    },
  ].forEach((item) => {
    const row = document.createElement("div");
    row.className = "legend-item";

    row.innerHTML = `
      <div class="legend-left">
        <span class="legend-swatch" style="background:${item.color}"></span>
        <span>${item.label}</span>
      </div>
      <strong>${percent(item.value)}</strong>
    `;

    legend.appendChild(row);
  });
}

function renderClassMetrics(perClass) {
  const tbody = document.getElementById("class-metrics");
  tbody.innerHTML = "";

  perClass.forEach((row) => {
    const tr = document.createElement("tr");

    tr.innerHTML = `
      <td>${row.label}</td>
      <td><span class="score-chip">${percent(row.precision)}</span></td>
      <td><span class="score-chip">${percent(row.recall)}</span></td>
      <td><span class="score-chip">${percent(row.f1_score)}</span></td>
      <td>${row.support.toLocaleString()}</td>
    `;

    tbody.appendChild(tr);
  });
}

function renderModelConfusionMatrices(confusionMatrices) {
  const root = document.getElementById("confusion-matrix-grid");
  root.innerHTML = "";

  confusionMatrices.forEach((entry) => {
    const matrix = entry.matrix;
    const labels = entry.labels;
    const maxValue = Math.max(...matrix.flat(), 1);

    const grid = document.createElement("div");
    grid.className = "matrix";

    const header = document.createElement("div");
    header.className = "matrix-header";

    header.innerHTML =
      `<div class="matrix-label">Actual -></div>` +
      labels
        .map((label) => `<div class="matrix-label">${label}</div>`)
        .join("");

    grid.appendChild(header);

    matrix.forEach((row, rowIndex) => {
      const rowEl = document.createElement("div");
      rowEl.className = "matrix-row";

      rowEl.innerHTML =
        `<div class="matrix-label">${labels[rowIndex]}</div>` +
        row
          .map((cell) => {
            const intensity = cell / maxValue;
            const bg =
              `rgba(255, 190, 85, ${0.14 + intensity * 0.62})`;

            return `
              <div class="matrix-cell" style="background:${bg}">
                ${cell.toLocaleString()}
              </div>
            `;
          })
          .join("");

      grid.appendChild(rowEl);
    });

    const card = document.createElement("article");
    card.className = "matrix-card";

    card.innerHTML = `
      <div class="matrix-card-head">
        <div>
          <h3>${entry.model}</h3>
          <div class="roc-subtle">Binary confusion matrix</div>
        </div>
        <div class="roc-subtle">Normal vs Attack</div>
      </div>
    `;

    card.appendChild(grid);
    root.appendChild(card);
  });
}

function buildRocPath(points, width, height, padding) {
  return points
    .map((point, index) => {
      const x =
        padding + point.fpr * (width - padding * 2);

      const y =
        height -
        padding -
        point.tpr * (height - padding * 2);

      return `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

function renderRocCurves(rocCurves) {
  const root = document.getElementById("roc-grid");
  root.innerHTML = "";

  const width = 360;
  const height = 260;
  const padding = 34;

  rocCurves.forEach((modelEntry) => {
    const card = document.createElement("article");
    card.className = "roc-card";

    const axisLines = [0.25, 0.5, 0.75]
      .map((step) => {
        const x =
          padding + step * (width - padding * 2);

        const y =
          height -
          padding -
          step * (height - padding * 2);

        return `
          <line
            x1="${padding}"
            y1="${y}"
            x2="${width - padding}"
            y2="${y}"
            stroke="rgba(255,255,255,0.08)"
          />

          <line
            x1="${x}"
            y1="${padding}"
            x2="${x}"
            y2="${height - padding}"
            stroke="rgba(255,255,255,0.08)"
          />
        `;
      })
      .join("");

    const curvePaths = modelEntry.classes
      .map(
        (curve, index) => `
          <path
            d="${buildRocPath(
              curve.points,
              width,
              height,
              padding
            )}"
            fill="none"
            stroke="${colorScale[index % colorScale.length]}"
            stroke-width="3"
            stroke-linecap="round"
          />
        `
      )
      .join("");

    const legend = modelEntry.classes
      .map(
        (curve, index) => `
          <div class="roc-legend-item">
            <div class="roc-legend-left">
              <span
                class="roc-swatch"
                style="background:${colorScale[index % colorScale.length]}"
              ></span>
              <span>${curve.label}</span>
            </div>

            <strong>AUC ${curve.auc.toFixed(3)}</strong>
          </div>
        `
      )
      .join("");

    card.innerHTML = `
      <div class="roc-card-head">
        <div>
          <h3>${modelEntry.model}</h3>
          <div class="roc-subtle">
            Macro AUC ${modelEntry.macro_auc.toFixed(3)}
          </div>
        </div>

        <div class="roc-subtle">
          TPR vs FPR
        </div>
      </div>

      <svg
        class="roc-svg"
        viewBox="0 0 ${width} ${height}"
        role="img"
        aria-label="ROC curve for ${modelEntry.model}"
      >
        ${axisLines}

        <line
          x1="${padding}"
          y1="${height - padding}"
          x2="${width - padding}"
          y2="${padding}"
          stroke="rgba(255,255,255,0.24)"
          stroke-dasharray="5 5"
        />

        <line
          x1="${padding}"
          y1="${height - padding}"
          x2="${width - padding}"
          y2="${height - padding}"
          stroke="rgba(255,255,255,0.42)"
        />

        <line
          x1="${padding}"
          y1="${height - padding}"
          x2="${padding}"
          y2="${padding}"
          stroke="rgba(255,255,255,0.42)"
        />

        ${curvePaths}

        <text
          x="${width / 2}"
          y="${height - 8}"
          text-anchor="middle"
          fill="#9eb4d3"
          font-size="12"
        >
          False Positive Rate
        </text>

        <text
          x="18"
          y="${height / 2}"
          text-anchor="middle"
          fill="#9eb4d3"
          font-size="12"
          transform="rotate(-90 18 ${height / 2})"
        >
          True Positive Rate
        </text>
      </svg>

      <div class="roc-legend">
        ${legend}
      </div>
    `;

    root.appendChild(card);
  });
}

function renderAlerts(alerts) {
  const root = document.getElementById("alerts-grid");
  root.innerHTML = "";

  alerts.forEach((alert) => {
    const card = document.createElement("article");
    const severityClass =
      `badge-${alert.severity.toLowerCase()}`;

    card.className = "alert-card";

    card.innerHTML = `
      <div class="alert-head">
        <div class="alert-title">
          ${alert.predicted_label}
        </div>

        <span class="badge ${severityClass}">
          ${alert.severity}
        </span>
      </div>

      <div class="alert-ip">
        ${alert.source_ip} -> ${alert.destination_ip}
      </div>

      <div class="alert-meta">
        <span>Protocol</span>
        <strong>${alert.protocol}</strong>
      </div>

      <div class="alert-pairs">
        <span>Actual / Status</span>
        <strong>
          ${alert.actual_label} / ${alert.status}
        </strong>
      </div>

      <div class="alert-pairs">
        <span>Connections</span>
        <strong>${alert.connections}</strong>
      </div>

      <div class="alert-pairs">
        <span>Failed logins</span>
        <strong>${alert.failed_logins}</strong>
      </div>

      <div class="alert-pairs">
        <span>Bytes sent</span>
        <strong>${alert.bytes_sent.toLocaleString()}</strong>
      </div>

      <div class="alert-score">
        <div class="bar-track">
          <div
            class="bar-fill"
            style="
              width:${alert.risk_score}%;
              background:linear-gradient(
                90deg,
                #ffbe55,
                #ff6577
              )
            "
          ></div>
        </div>

        <div class="alert-meta">
          <span>Risk score</span>
          <strong>${alert.risk_score}/100</strong>
        </div>
      </div>
    `;

    root.appendChild(card);
  });
}


/* ---------------------------------------------------------------- */
/* PHASE 4: Synthetic flow alert feed (WebSocket-driven)            */
/* ---------------------------------------------------------------- */

const LIVE_FEED_MAX_ITEMS = 60;

const liveState = {
  totalFlows: 0,
  totalAttacks: 0,
  bySeverity: {
    Critical: 0,
    High: 0,
    Medium: 0,
    Low: 0
  },
  socket: null,
  reconnectDelayMs: 1000,
};

const controlState = {
  status: "running"
};

function updateControlButtons(status) {
  controlState.status = status;

  const start = document.getElementById("start-feed-btn");
  const pause = document.getElementById("pause-feed-btn");
  const stop = document.getElementById("stop-feed-btn");

  if (!start || !pause || !stop) return;

  start.disabled = status === "running";
  pause.disabled = status !== "running";
  stop.disabled = status === "stopped";

  start.textContent =
    status === "paused"
      ? "▶ Resume Feed"
      : "▶ Start Feed";
}

async function setSyntheticControl(action) {
  const endpoint = {
    start: "/api/synthetic/start",
    pause: "/api/synthetic/pause",
    resume: "/api/synthetic/resume",
    stop: "/api/synthetic/stop",
  }[action];

  if (!endpoint) return;

  try {
    const response = await fetch(endpoint, {
      method: "POST"
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.error || "Control request failed"
      );
    }

    updateControlButtons(data.status);

    setLiveStatus(
      data.status === "running"
        ? "live"
        : data.status === "paused"
        ? "connecting"
        : "down"
    );

    if (data.status === "paused") {
      document.getElementById(
        "live-status-text"
      ).textContent = "Synthetic feed paused";

    } else if (data.status === "stopped") {
      document.getElementById(
        "live-status-text"
      ).textContent = "Synthetic feed stopped";
    }

  } catch (error) {
    console.error(
      "Synthetic feed control failed",
      error
    );
  }
}

async function clearLiveEvents() {
  try {
    const response = await fetch(
      "/api/alerts/clear",
      {
        method: "POST"
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.error || "Clear failed"
      );
    }

    liveState.totalFlows = 0;
    liveState.totalAttacks = 0;

    liveState.bySeverity = {
      Critical: 0,
      High: 0,
      Medium: 0,
      Low: 0
    };

    const feed = document.getElementById(
      "live-feed"
    );

    feed.innerHTML =
      '<p class="live-feed-empty" id="live-feed-empty">' +
      'Waiting for synthetic network flows…' +
      '</p>';

    renderLiveStats();

    // Confirm UI matches SQLite after clearing.
    await refreshLiveStats();

  } catch (error) {
    console.error(
      "Could not clear synthetic events",
      error
    );
  }
}

function bindLiveControls() {
  document
    .getElementById("start-feed-btn")
    ?.addEventListener(
      "click",
      () => {
        setSyntheticControl(
          controlState.status === "paused"
            ? "resume"
            : "start"
        );
      }
    );

  document
    .getElementById("pause-feed-btn")
    ?.addEventListener(
      "click",
      () => setSyntheticControl("pause")
    );

  document
    .getElementById("stop-feed-btn")
    ?.addEventListener(
      "click",
      () => setSyntheticControl("stop")
    );

  document
    .getElementById("clear-events-btn")
    ?.addEventListener(
      "click",
      clearLiveEvents
    );
}

function setLiveStatus(state) {
  // state: 'connecting' | 'live' | 'down'

  const pill =
    document.getElementById(
      "live-status-pill"
    );

  const text =
    document.getElementById(
      "live-status-text"
    );

  pill.classList.remove(
    "status-live",
    "status-down",
    "status-connecting"
  );

  pill.classList.add(
    `status-${state}`
  );

  text.textContent = {
    connecting: "Starting…",
    live: "Synthetic feed active",
    down: "Synthetic feed stopped / reconnecting…",
  }[state];
}

function renderLiveStats() {
  const root = document.getElementById("live-stats");

  // Prevent the entire dashboard from stopping if
  // the live statistics container is not present.
  if (!root) return;

  const attackRate = liveState.totalFlows
    ? percent(liveState.totalAttacks / liveState.totalFlows)
    : "0.0%";

  root.innerHTML = "";

  [
    [
      "Flows analyzed",
      liveState.totalFlows.toLocaleString()
    ],
    [
      "Attacks detected",
      liveState.totalAttacks.toLocaleString()
    ],
    [
      "Attack rate",
      attackRate
    ],
    [
      "Critical alerts",
      liveState.bySeverity.Critical.toLocaleString()
    ],
  ].forEach(
    ([label, value]) => {
      const card =
        document.createElement(
          "div"
        );

      card.className =
        "live-stat-card";

      card.innerHTML = `
        <span class="live-stat-label">
          ${label}
        </span>

        <strong class="live-stat-value">
          ${value}
        </strong>
      `;

      root.appendChild(card);
    }
  );
}

function formatTimestamp(isoString) {
  try {
    const date = new Date(
      isoString
    );

    return date.toLocaleTimeString(
      [],
      {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
      }
    );

  } catch (error) {
    return isoString;
  }
}

function buildLiveAlertNode(
  alert,
  { flash = false } = {}
) {
  const item =
    document.createElement(
      "div"
    );

  const isAttack =
    Boolean(alert.is_attack);

  const severity =
    (
      alert.severity ||
      "Low"
    ).toLowerCase();

  item.className =
    `live-alert-item ${
      isAttack ? "is-attack" : ""
    } ${
      flash
        ? (
            isAttack
              ? "flash-attack"
              : "flash-normal"
          )
        : ""
    }`;

  item.innerHTML = `
    <span class="live-alert-time">
      ${formatTimestamp(
        alert.timestamp
      )}
    </span>

    <span class="live-alert-flow">
      <strong>
        ${alert.attack_type}
      </strong>
      —
      ${alert.source_ip || "?"}
      →
      ${alert.destination_ip || "?"}
      (${alert.protocol || "?"})
    </span>

    <span class="badge badge-${severity}">
      ${alert.severity || "Low"}
    </span>

    <span class="live-alert-confidence">
      ${percent(
        alert.confidence || 0
      )}
      confidence
    </span>
  `;

  return item;
}

function prependLiveAlert(
  alert,
  { flash = true } = {}
) {
  const feed =
    document.getElementById(
      "live-feed"
    );

  const empty =
    document.getElementById(
      "live-feed-empty"
    );

  if (empty) {
    empty.remove();
  }

  const node =
    buildLiveAlertNode(
      alert,
      { flash }
    );

  feed.prepend(node);

  while (
    feed.children.length >
    LIVE_FEED_MAX_ITEMS
  ) {
    feed.removeChild(
      feed.lastChild
    );
  }

  liveState.totalFlows += 1;

  if (alert.is_attack) {
    liveState.totalAttacks += 1;

    const severity =
      alert.severity &&
      liveState.bySeverity.hasOwnProperty(
        alert.severity
      )
        ? alert.severity
        : "Medium";

    liveState.bySeverity[
      severity
    ] =
      (
        liveState.bySeverity[
          severity
        ] || 0
      ) + 1;
  }

  renderLiveStats();
}

function loadLiveBacklog(_alerts) {
  // The live feed is intentionally session/connection based.
  // Do NOT repopulate old SQLite history when the browser is refreshed.
  // The persistent totals are loaded separately by refreshLiveStats.

  const feed =
    document.getElementById(
      "live-feed"
    );

  if (!feed) return;

  feed.innerHTML =
    '<p class="live-feed-empty" id="live-feed-empty">' +
    'Waiting for new synthetic network flows…' +
    '</p>';
}

async function refreshLiveStats() {
  try {
    const response =
      await fetch(
        "/api/alerts/stats?ts=" +
        Date.now()
      );

    if (!response.ok) {
      throw new Error(
        "Failed to load live statistics"
      );
    }

    const data =
      await response.json();

    liveState.totalFlows =
      data.total_flows_seen || 0;

    liveState.totalAttacks =
      data.total_attacks || 0;

    liveState.bySeverity = {
      Critical:
        data.by_severity?.Critical || 0,

      High:
        data.by_severity?.High || 0,

      Medium:
        data.by_severity?.Medium || 0,

      Low:
        data.by_severity?.Low || 0,
    };

    renderLiveStats();

  } catch (error) {
    console.error(
      "Could not refresh live statistics",
      error
    );
  }
}

function connectLiveFeed() {
  setLiveStatus(
    "connecting"
  );

  const protocol =
    window.location.protocol ===
    "https:"
      ? "wss:"
      : "ws:";

  const url =
    `${protocol}//${window.location.host}/ws/alerts`;

  const socket =
    new WebSocket(url);

  liveState.socket =
    socket;

  socket.onopen = () => {
    // WebSocket connection is healthy; actual generator state is synchronized
    // separately by /api/health.

    setLiveStatus(
      "live"
    );

    liveState.reconnectDelayMs =
      1000;

    syncSyntheticStatus();
  };

  socket.onmessage =
    (event) => {
      let message;

      try {
        message =
          JSON.parse(
            event.data
          );

      } catch (error) {
        console.error(
          "Malformed WebSocket message",
          error
        );

        return;
      }

      if (
        message.type ===
        "backlog"
      ) {
        // Backend deliberately sends an empty backlog on a new connection.
        // Refresh only the persistent counters; old alert cards stay hidden.

        loadLiveBacklog(
          message.data
        );

        refreshLiveStats();

      } else if (
        message.type ===
        "alert"
      ) {
        prependLiveAlert(
          message.data,
          {
            flash: true
          }
        );

        refreshLiveStats();
      }
    };

  socket.onclose = () => {
    setLiveStatus(
      "down"
    );

    // Exponential backoff reconnect, capped at 15s,
    // so a restarted backend or laptop waking from sleep
    // recovers the live feed automatically.

    setTimeout(
      connectLiveFeed,
      liveState.reconnectDelayMs
    );

    liveState.reconnectDelayMs =
      Math.min(
        liveState.reconnectDelayMs * 1.5,
        15000
      );
  };

  socket.onerror = () => {
    socket.close();
  };
}

renderLiveStats();

async function boot() {
  const status =
    document.getElementById(
      "dataset-status"
    );

  try {
    const response =
      await fetch(
        `data/dashboard_data.json?ts=${Date.now()}`
      );

    const data =
      await response.json();

    status.textContent =
      `Loaded from ${data.meta.generated_from}`;

    renderMetricCards(
      data
    );

    document.getElementById(
      "best-model-note"
    ).textContent =
      `${data.overview.best_model} leads the evaluated models with ${percent(
        data.overview.best_accuracy
      )} test accuracy.`;

    renderBars(
      "model-bars",
      data.model_results,
      "test_accuracy",
      percent
    );

    renderAttackDonut(
      data
    );

    renderBars(
      "train-distribution",
      data.class_distribution.train,
      "count",
      (value) =>
        value.toLocaleString()
    );

    renderBars(
      "test-distribution",
      data.class_distribution.test,
      "count",
      (value) =>
        value.toLocaleString()
    );

    renderBars(
      "train-protocols",
      data.protocol_distribution.train,
      "count",
      (value) =>
        value.toLocaleString()
    );

    renderBars(
      "test-protocols",
      data.protocol_distribution.test,
      "count",
      (value) =>
        value.toLocaleString()
    );

    renderClassMetrics(
      data.best_model_report.per_class
    );

    renderBars(
      "feature-importance",
      data.feature_importance,
      "importance",
      (value) =>
        value.toFixed(4)
    );

    renderModelConfusionMatrices(
      data.model_confusion_matrices
    );

    renderRocCurves(
      data.roc_curves
    );

    renderAlerts(
      data.alerts
    );

    // Dynamic monitoring statistics come from SQLite,
    // not the static dashboard JSON and not a hard-coded
    // WebSocket backlog size.

    await refreshLiveStats();

  } catch (error) {
    status.textContent =
      "Could not load dashboard data";

    console.error(error);
  }
}

async function syncSyntheticStatus() {
  try {
    const response =
      await fetch(
        "/api/health"
      );

    const data =
      await response.json();

    const status =
      data.synthetic_flow_status ||
      (
        data.synthetic_flow_active
          ? "running"
          : "stopped"
      );

    updateControlButtons(
      status
    );

    if (
      status ===
      "paused"
    ) {
      setLiveStatus(
        "connecting"
      );

      document.getElementById(
        "live-status-text"
      ).textContent =
        "Synthetic feed paused";

    } else if (
      status ===
      "stopped"
    ) {
      setLiveStatus(
        "down"
      );

      document.getElementById(
        "live-status-text"
      ).textContent =
        "Synthetic feed stopped";
    }

  } catch (error) {
    console.error(
      "Could not read synthetic feed status",
      error
    );
  }
}

boot();
bindLiveControls();
syncSyntheticStatus();
connectLiveFeed();
