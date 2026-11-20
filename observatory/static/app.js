(() => {
  const state = {
    currentView: "command",
    selectedOrganismId: null,
    autoRunTimer: null,
    livePlasticityTimer: null,
    liveStepInFlight: false,
    manualRpeTouched: false,
    signalsPaused: false,
    offscreenPaused: false,
    showPhaseOverlay: true,
    telemetryEventSource: null,
    connectomeGraph: null,
    selectedNodeId: null,
    camRotY: 0.35,
    camRotX: 0.22,
    camZoom: 1.0,
    threeCtx: null,
    signalPhase: 0,
    lastResearchData: null,
  };

  const $ = (sel) => document.querySelector(sel);
  const announce = (msg) => {
    const el = $("#live-announcer");
    if (el) el.textContent = msg;
  };

  async function api(path, opts = {}) {
    const { background, ...fetchOpts } = opts;
    const storedKey = window.localStorage?.getItem("omnis_api_key");
    const urlKey = new URLSearchParams(window.location.search).get("api_key");
    const apiKey = urlKey || storedKey;
    const headers = {
      "Content-Type": "application/json",
      ...(apiKey ? { "X-API-Key": apiKey } : {}),
      ...(fetchOpts.headers || {}),
    };
    const res = await fetch(path, {
      ...fetchOpts,
      headers,
      ...(background ? { priority: "low" } : {}),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`${res.status}: ${txt}`);
    }
    return res.json();
  }

  function kuramotoPhaseHue(phaseRad) {
    const twoPi = Math.PI * 2;
    const norm = (((Number(phaseRad || 0) % twoPi) + twoPi) % twoPi) / twoPi;
    return Math.round(norm * 360);
  }

  function formatFrontierHookTelemetry(hookStatus) {
    if (!hookStatus || !hookStatus.active) {
      return "Neutral hook inactive (0 core overhead). Enable hook & step simulation to stream live observer telemetry.";
    }
    const latest = hookStatus.latest_telemetry;
    if (!latest) {
      return `Hook ACTIVE (${(hookStatus.spec && Object.keys(hookStatus.spec).filter((k) => k.startsWith("enable_") && hookStatus.spec[k]).length) || 0} flags enabled) · Waiting for next core tick...`;
    }
    return `Tick ${latest.tick} Observers · ${JSON.stringify(latest.observers)}`;
  }

  function renderKV(containerSelector, entries) {
    const el = $(containerSelector);
    if (!el) return;
    el.innerHTML = entries
      .map(
        ([k, v]) =>
          `<div class="kv-item"><dt>${escapeHtml(k)}</dt><dd>${escapeHtml(String(v))}</dd></div>`
      )
      .join("");
  }

  function escapeHtml(s) {
    return s
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  // Two-state color-scheme toggle synchronized with <meta name="color-scheme"> and storage events
  const colorSchemeMeta = document.querySelector('meta[name="color-scheme"]');
  const sysDarkQuery = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function getStoredScheme() {
    try {
      const v = localStorage.getItem("color-scheme");
      return v === "light" || v === "dark" ? v : null;
    } catch (_) {
      return null;
    }
  }
  function getRenderedScheme() {
    const stored = getStoredScheme();
    if (stored) return stored;
    return sysDarkQuery && sysDarkQuery.matches ? "dark" : "light";
  }
  function applySchemeMeta(scheme) {
    if (colorSchemeMeta) {
      colorSchemeMeta.content = scheme || "light dark";
    }
  }
  $("#btn-theme-toggle")?.addEventListener("click", () => {
    const current = getRenderedScheme();
    const target = current === "dark" ? "light" : "dark";
    const sysPref = sysDarkQuery && sysDarkQuery.matches ? "dark" : "light";
    if (target === sysPref) {
      try { localStorage.removeItem("color-scheme"); } catch (_) {}
      applySchemeMeta("light dark");
      announce(`Theme reset to system preference (${sysPref}).`);
    } else {
      try { localStorage.setItem("color-scheme", target); } catch (_) {}
      applySchemeMeta(target);
      announce(`Switched observatory theme to ${target} mode.`);
    }
  });
  window.addEventListener("storage", (ev) => {
    if (ev.key === "color-scheme") {
      applySchemeMeta(ev.newValue === "light" || ev.newValue === "dark" ? ev.newValue : "light dark");
    }
  });

  // Navigation across A-I Observatory panels
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.dataset.view;
      state.currentView = view;
      document.querySelectorAll(".nav-btn").forEach((b) => {
        b.setAttribute("aria-current", b === btn ? "page" : "false");
      });
      btn.scrollIntoView({ block: "nearest", inline: "nearest" });
      document.querySelectorAll(".view-panel").forEach((panel) => {
        const show = panel.id === `view-${view}`;
        panel.hidden = !show;
        panel.classList.toggle("active", show);
      });
      document.title = `${btn.textContent.trim()} | OMNIS 1.0 Cognitive Observatory`;
      refreshCurrentView();
    });
  });

  // Simulation Controls
  $("#btn-step-1").addEventListener("click", async () => {
    await api("/api/simulation/step", { method: "POST", body: JSON.stringify({ steps: 1 }) });
    await refreshAll();
    announce("Advanced simulation by 1 tick.");
  });

  $("#btn-step-10").addEventListener("click", async () => {
    await api("/api/simulation/step", { method: "POST", body: JSON.stringify({ steps: 10 }) });
    await refreshAll();
    announce("Advanced simulation by 10 ticks.");
  });

  $("#btn-play-pause").addEventListener("click", () => {
    const btn = $("#btn-play-pause");
    if (state.autoRunTimer) {
      clearInterval(state.autoRunTimer);
      state.autoRunTimer = null;
      btn.textContent = "Auto-Run";
      announce("Auto-run paused.");
    } else {
      const intervalMs = Number($("#speed-select").value || 700);
      state.autoRunTimer = setInterval(async () => {
        try {
          await api("/api/simulation/step", { method: "POST", body: JSON.stringify({ steps: 1 }), background: true });
          await refreshAll();
        } catch (_) {}
      }, intervalMs);
      btn.textContent = "Pause";
      announce("Auto-run started.");
    }
  });

  $("#speed-select").addEventListener("change", () => {
    if (state.autoRunTimer) {
      $("#btn-play-pause").click();
      $("#btn-play-pause").click();
    }
  });

  $("#btn-persist").addEventListener("click", async () => {
    const res = await api("/api/simulation/persist", { method: "POST" });
    announce(`Saved ${res.saved_organisms} organisms and ${res.saved_memories} memories to SQLite.`);
  });

  $("#btn-reset-sim").addEventListener("click", async () => {
    await api("/api/simulation/reset", { method: "POST" });
    state.selectedOrganismId = null;
    await refreshAll();
    announce("Ecosystem reset to initial generation.");
  });

  $("#organism-select").addEventListener("change", (e) => {
    state.selectedOrganismId = e.target.value || null;
    refreshAll();
  });

  // View A: Command Centre
  async function refreshCommandCentre() {
    const [cc, ev] = await Promise.all([api("/api/command-centre"), api("/api/events?limit=25", { background: true })]);
    $("#header-tick").textContent = cc.tick;
    $("#header-gen").textContent = cc.generation;
    $("#sim-status-text").textContent = cc.status;
    $("#kpi-organisms").textContent = `${cc.active_organisms} / ${cc.max_population}`;
    $("#kpi-generation").textContent = `Generation ${cc.generation}`;
    $("#kpi-neural").textContent = cc.mean_neural_activity.toFixed(3);
    $("#kpi-synapses").textContent = `${cc.total_active_synapses} active synapses (${cc.total_active_neurons} neurons)`;
    $("#kpi-memories").textContent = cc.total_memories;
    $("#kpi-fitness").textContent = cc.mean_fitness.toFixed(3);
    $("#kpi-latency").textContent = `${cc.step_latency_ms} ms / tick`;
    $("#kpi-cpu").textContent = `${cc.hardware.cpu_percent}%`;
    $("#kpi-gpu").textContent = cc.hardware.gpu_available ? "CUDA GPU Active" : `${cc.hardware.active_backend} · ${cc.hardware.process_memory_mb} MB`;
    $("#kpi-experiments").textContent = cc.active_experiments;

    drawCommandChart(cc.telemetry_history || []);

    const logEl = $("#event-log-list");
    logEl.innerHTML = (ev.events || [])
      .slice()
      .reverse()
      .map(
        (item) =>
          `<li><span class="cyan-text">[t=${item.tick}]</span> <strong>${escapeHtml(item.event_type)}</strong> · ${escapeHtml(item.source)}</li>`
      )
      .join("");
  }

  function drawCommandChart(history) {
    const canvas = $("#command-chart");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(110,168,254,0.12)";
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 45) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (!history.length) {
      ctx.fillStyle = "#93a7cb";
      ctx.font = "13px monospace";
      ctx.fillText("Step simulation to collect telemetry trajectory...", 24, h / 2);
      return;
    }

    const pad = 28;
    const fitnessVals = history.map((d) => d.mean_fitness);
    const synVals = history.map((d) => d.total_synapses);
    const minF = Math.min(...fitnessVals, -0.5);
    const maxF = Math.max(...fitnessVals, 1.0);
    const maxS = Math.max(...synVals, 10);

    const plotLine = (vals, minV, maxV, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      vals.forEach((v, i) => {
        const x = pad + (i / Math.max(1, vals.length - 1)) * (w - 2 * pad);
        const y = h - pad - ((v - minV) / Math.max(1e-6, maxV - minV)) * (h - 2 * pad);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    plotLine(fitnessVals, minF, maxF, "#00e5ff");
    plotLine(synVals, 0, maxS, "#9d6bff");

    ctx.fillStyle = "#00e5ff";
    ctx.font = "12px monospace";
    ctx.fillText(`Mean Fitness (${fitnessVals.at(-1).toFixed(3)})`, pad, 18);
    ctx.fillStyle = "#9d6bff";
    ctx.fillText(`Total Synapses (${synVals.at(-1)})`, pad + 210, 18);
  }

  // View B: 3D Connectome Explorer (Three.js + Built-in 3D Projection Canvas Fallback)
  function init3DConnectomeHost() {
    const host = $("#connectome-3d-container");
    if (!host || host.dataset.ready) return;
    host.dataset.ready = "1";

    const useThree = typeof window.THREE !== "undefined";
    if (useThree) {
      try {
        const THREE = window.THREE;
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, host.clientWidth / Math.max(1, host.clientHeight), 0.1, 200);
        camera.position.set(0, 1.5, 16);
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(host.clientWidth || 680, host.clientHeight || 460);
        host.appendChild(renderer.domElement);
        const group = new THREE.Group();
        scene.add(group);
        state.threeCtx = {
          THREE,
          scene,
          camera,
          renderer,
          group,
          nodeMeshes: [],
          nodeMeshMap: {},
          signalMeshes: [],
        };

        const raycaster = new THREE.Raycaster();
        const pointer = new THREE.Vector2();
        const hitTestNode = (ev) => {
          const rect = renderer.domElement.getBoundingClientRect();
          pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
          pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
          raycaster.setFromCamera(pointer, camera);
          const hits = raycaster.intersectObjects(state.threeCtx.nodeMeshes);
          return hits.length ? hits[0].object.userData.node : null;
        };
        renderer.domElement.addEventListener("click", (ev) => {
          const node = hitTestNode(ev);
          if (node) inspectNode(node);
        });
        renderer.domElement.addEventListener("dblclick", (ev) => {
          const node = hitTestNode(ev);
          if (node) {
            inspectNode(node);
            stimulateNeuron(node.neuron_id);
          }
        });
      } catch (_) {
        state.threeCtx = null;
      }
    }

    if (!state.threeCtx) {
      const canvas = document.createElement("canvas");
      canvas.width = host.clientWidth || 720;
      canvas.height = host.clientHeight || 460;
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      host.appendChild(canvas);
      state.fallback3DCanvas = canvas;
      const pick2DNode = (ev) => {
        if (!state.projectedNodes) return null;
        const rect = canvas.getBoundingClientRect();
        const mx = ((ev.clientX - rect.left) / rect.width) * canvas.width;
        const my = ((ev.clientY - rect.top) / rect.height) * canvas.height;
        const hit = state.projectedNodes.find((n) => Math.hypot(n.sx - mx, n.sy - my) < 14);
        return hit ? hit.raw : null;
      };
      canvas.addEventListener("click", (ev) => {
        const node = pick2DNode(ev);
        if (node) inspectNode(node);
      });
      canvas.addEventListener("dblclick", (ev) => {
        const node = pick2DNode(ev);
        if (node) {
          inspectNode(node);
          stimulateNeuron(node.neuron_id);
        }
      });
    }

    const stageEl = $("#connectome-canvas-stage") || host;
    const supportsCV = "contentVisibility" in document.documentElement.style;
    const handleVisibility = (skipped) => {
      state.offscreenPaused = Boolean(skipped);
      const sseEl = $("#hud-sse-status");
      if (state.offscreenPaused && state.telemetryEventSource) {
        state.telemetryEventSource.close();
        state.telemetryEventSource = null;
        if (sseEl) sseEl.textContent = "PAUSED";
      } else if (!state.offscreenPaused && state.livePlasticityTimer && !state.telemetryEventSource) {
        startTelemetryStream();
      }
    };
    stageEl.addEventListener("contentvisibilityautostatechange", (ev) => handleVisibility(ev.skipped));
    if (!supportsCV && typeof IntersectionObserver !== "undefined") {
      const io = new IntersectionObserver((entries) => {
        entries.forEach((entry) => handleVisibility(!entry.isIntersecting));
      });
      io.observe(stageEl);
    }

    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    host.addEventListener("mousedown", (e) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    window.addEventListener("mouseup", () => { dragging = false; });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      state.camRotY += (e.clientX - lastX) * 0.008;
      state.camRotX += (e.clientY - lastY) * 0.008;
      lastX = e.clientX;
      lastY = e.clientY;
    });
    host.addEventListener("wheel", (e) => {
      e.preventDefault();
      state.camZoom = Math.max(0.45, Math.min(2.5, state.camZoom - e.deltaY * 0.001));
    }, { passive: false });

    const motionQuery = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
    const animate = () => {
      requestAnimationFrame(animate);
      const reduced = Boolean(motionQuery && motionQuery.matches);
      if (!state.signalsPaused && !state.offscreenPaused && !reduced) {
        state.signalPhase = (state.signalPhase + 0.028) % 1.0;
      }
      if (state.currentView === "connectome" && state.connectomeGraph && !state.offscreenPaused) {
        render3DConnectomeFrame();
      }
    };
    requestAnimationFrame(animate);
  }

  function inspectNode(node) {
    if (!node) return;
    state.selectedNodeId = node.neuron_id;
    const edges = (state.connectomeGraph?.edges || []).filter(
      (e) => e.source === node.neuron_id || e.target === node.neuron_id
    );
    edges.sort((a, b) => Math.abs(b.last_dw || 0) - Math.abs(a.last_dw || 0) || Math.abs(b.weight) - Math.abs(a.weight));
    const inspector = $("#node-inspector");
    inspector.innerHTML =
      `<strong>${escapeHtml(node.neuron_id)}</strong> (${escapeHtml(node.role)} · ${escapeHtml(node.neuron_model)})<br>` +
      `Active: ${node.active} · Excitatory: ${node.excitatory}<br>` +
      `Membrane u_i: ${node.u.toFixed(4)} · Activation a_i: ${node.a.toFixed(4)}<br>` +
      `BCM θ_m: ${Number(node.metaplastic_theta ?? 0.35).toFixed(4)} · Kuramoto φ: ${Number(node.kuramoto_phase ?? 0).toFixed(3)} rad<br>` +
      `Connected synapses: ${edges.length}<br>` +
      edges
        .slice(0, 6)
        .map(
          (e) =>
            `${escapeHtml(e.source)} → ${escapeHtml(e.target)} (w=${e.weight.toFixed(3)}, Δw=${Number(e.last_dw || 0) >= 0 ? "+" : ""}${Number(e.last_dw || 0).toFixed(4)})`
        )
        .join("<br>") +
      `<div class="inspector-actions"><button type="button" id="btn-inspect-stimulate" class="btn btn-primary">Pulse ${escapeHtml(node.neuron_id)}</button></div>`;
    const btn = $("#btn-inspect-stimulate");
    if (btn) {
      btn.addEventListener("click", () => stimulateNeuron(node.neuron_id));
    }
  }

  function rebuildThreeGraph(graph) {
    if (!state.threeCtx) return;
    const { THREE, group } = state.threeCtx;
    while (group.children.length) group.remove(group.children[0]);
    state.threeCtx.nodeMeshes = [];
    state.threeCtx.nodeMeshMap = {};
    state.threeCtx.signalMeshes = [];

    const posMap = {};
    graph.nodes.forEach((n) => {
      const [x, y, z] = n.pos_3d || [0, 0, 0];
      posMap[n.neuron_id] = new THREE.Vector3(x, y, z);
      const geo = new THREE.SphereGeometry(n.role === "hidden" ? 0.34 : 0.42, 16, 16);
      const color = !n.active
        ? 0xffb020
        : n.role === "input"
        ? 0x26e3a8
        : n.role === "output"
        ? 0xffffff
        : n.excitatory
        ? 0x00e5ff
        : 0x9d6bff;
      const mat = new THREE.MeshBasicMaterial({ color });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.copy(posMap[n.neuron_id]);
      const baseScale = 1.0 + Math.min(0.65, Math.abs(n.a || 0) * 0.55);
      mesh.scale.setScalar(baseScale);
      mesh.userData = { node: n, baseScale };
      group.add(mesh);
      state.threeCtx.nodeMeshes.push(mesh);
      state.threeCtx.nodeMeshMap[n.neuron_id] = mesh;

      if (state.selectedNodeId === n.neuron_id) {
        const haloGeo = new THREE.SphereGeometry(n.role === "hidden" ? 0.52 : 0.62, 12, 12);
        const haloMat = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.65 });
        const halo = new THREE.Mesh(haloGeo, haloMat);
        halo.position.copy(mesh.position);
        group.add(halo);
      }

      if (state.showPhaseOverlay && n.active && Math.abs(Number(n.kuramoto_phase || 0)) > 1e-4) {
        const hue = kuramotoPhaseHue(n.kuramoto_phase);
        const phaseGeo = new THREE.RingGeometry(0.46, 0.56, 20);
        const phaseMat = new THREE.MeshBasicMaterial({
          color: new THREE.Color(`hsl(${hue}, 90%, 62%)`),
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
        });
        const phaseRing = new THREE.Mesh(phaseGeo, phaseMat);
        phaseRing.position.copy(mesh.position);
        group.add(phaseRing);
      }
    });

    graph.edges.forEach((e) => {
      const p1 = posMap[e.source];
      const p2 = posMap[e.target];
      if (!p1 || !p2) return;
      const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
      const dw = Number(e.last_dw || 0);
      const dwStrong = Math.abs(dw) > 0.002;
      const color = !e.active
        ? 0x553311
        : dw > 0.002
        ? 0x26e3a8
        : dw < -0.002
        ? 0xff5b84
        : e.is_graft
        ? 0xffd166
        : e.weight >= 0
        ? 0x00e5ff
        : 0x9d6bff;
      const opacity = !e.active
        ? 0.16
        : dwStrong
        ? Math.min(0.95, 0.6 + Math.abs(dw) * 8)
        : e.is_graft
        ? 0.88
        : Math.min(0.75, 0.25 + Math.abs(e.weight) * 0.35);
      const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity });
      group.add(new THREE.Line(geom, mat));
    });

    const signals = Array.isArray(graph.recent_signals) && graph.recent_signals.length
      ? graph.recent_signals.slice(0, 64)
      : graph.edges
          .filter((e) => e.active && Math.abs(e.weight) > 0.22)
          .slice(0, 14)
          .map((e) => ({
            pre_id: e.source,
            post_id: e.target,
            flux: e.weight * 0.35,
            excitatory: e.weight >= 0,
            ambient: true,
          }));

    signals.forEach((sig, idx) => {
      const p1 = posMap[sig.pre_id];
      const p2 = posMap[sig.post_id];
      if (!p1 || !p2) return;
      const mag = Math.abs(Number(sig.flux || 0));
      const pRadius = sig.ambient ? 0.09 : Math.min(0.24, 0.11 + mag * 0.09);
      const sGeo = new THREE.SphereGeometry(pRadius, 8, 8);
      const sColor = sig.ambient
        ? 0xffb020
        : Number(sig.flux || 0) >= 0
        ? mag > 0.45
          ? 0xffffff
          : 0xffb020
        : 0xff5b84;
      const sMat = new THREE.MeshBasicMaterial({
        color: sColor,
        transparent: true,
        opacity: sig.ambient ? 0.45 : 0.95,
      });
      const sMesh = new THREE.Mesh(sGeo, sMat);
      sMesh.position.copy(p1);
      sMesh.userData = {
        startVec: p1,
        endVec: p2,
        phaseOffset: (idx * 0.17) % 1.0,
        speed: sig.ambient ? 0.65 : 0.8 + Math.min(1.4, mag * 0.9),
      };
      group.add(sMesh);
      state.threeCtx.signalMeshes.push(sMesh);
    });
  }

  function render3DConnectomeFrame() {
    const graph = state.connectomeGraph;
    if (!graph) return;
    if (state.threeCtx) {
      const { renderer, scene, camera, group, signalMeshes, nodeMeshes } = state.threeCtx;
      (signalMeshes || []).forEach((sMesh) => {
        const d = sMesh.userData;
        const t = (state.signalPhase * d.speed + d.phaseOffset) % 1.0;
        sMesh.position.lerpVectors(d.startVec, d.endVec, t);
        const pulse = 0.85 + 0.3 * Math.sin(t * Math.PI);
        sMesh.scale.setScalar(pulse);
      });
      (nodeMeshes || []).forEach((nMesh) => {
        const d = nMesh.userData;
        if (d && d.node && Math.abs(d.node.a || 0) > 0.05) {
          const breathe = 1 + 0.08 * Math.sin(state.signalPhase * Math.PI * 2) * Math.min(1, Math.abs(d.node.a));
          nMesh.scale.setScalar(d.baseScale * breathe);
        }
      });
      group.rotation.y = state.camRotY;
      group.rotation.x = state.camRotX;
      camera.position.z = 16 / state.camZoom;
      renderer.render(scene, camera);
      return;
    }

    const canvas = state.fallback3DCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cosY = Math.cos(state.camRotY);
    const sinY = Math.sin(state.camRotY);
    const cosX = Math.cos(state.camRotX);
    const sinX = Math.sin(state.camRotX);

    const proj = {};
    const projectedList = [];
    graph.nodes.forEach((n) => {
      const [x0, y0, z0] = n.pos_3d || [0, 0, 0];
      const x1 = x0 * cosY - z0 * sinY;
      const z1 = x0 * sinY + z0 * cosY;
      const y2 = y0 * cosX - z1 * sinX;
      const z2 = y0 * sinX + z1 * cosX;
      const scale = (38 * state.camZoom) / Math.max(3.0, 14 - z2 * 0.5);
      const sx = w / 2 + x1 * scale * 9;
      const sy = h / 2 - y2 * scale * 9;
      proj[n.neuron_id] = { sx, sy, z: z2, raw: n };
      projectedList.push(proj[n.neuron_id]);
    });
    state.projectedNodes = projectedList;

    graph.edges.forEach((e) => {
      const a = proj[e.source];
      const b = proj[e.target];
      if (!a || !b) return;
      const dw = Number(e.last_dw || 0);
      const dwStrong = Math.abs(dw) > 0.002;
      ctx.strokeStyle = !e.active
        ? "rgba(255,176,32,0.18)"
        : dw > 0.002
        ? `rgba(38,227,168,${Math.min(0.95, 0.55 + Math.abs(dw) * 8)})`
        : dw < -0.002
        ? `rgba(255,91,132,${Math.min(0.95, 0.55 + Math.abs(dw) * 8)})`
        : e.is_graft
        ? "rgba(255,209,102,0.88)"
        : e.weight >= 0
        ? `rgba(0,229,255,${Math.min(0.85, 0.2 + Math.abs(e.weight) * 0.45)})`
        : `rgba(157,107,255,${Math.min(0.85, 0.2 + Math.abs(e.weight) * 0.45)})`;
      ctx.lineWidth = Math.max(1, Math.min(4, Math.abs(e.weight) * 1.8 + (dwStrong ? 1.0 : 0)));
      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    });

    const sigs = Array.isArray(graph.recent_signals) && graph.recent_signals.length
      ? graph.recent_signals.slice(0, 64)
      : graph.edges
          .filter((e) => e.active && Math.abs(e.weight) > 0.22)
          .slice(0, 14)
          .map((e) => ({ pre_id: e.source, post_id: e.target, flux: e.weight * 0.35, ambient: true }));

    sigs.forEach((sig, idx) => {
      const a = proj[sig.pre_id];
      const b = proj[sig.post_id];
      if (!a || !b) return;
      const mag = Math.abs(Number(sig.flux || 0));
      const spd = sig.ambient ? 0.65 : 0.8 + Math.min(1.4, mag * 0.9);
      const t = (state.signalPhase * spd + (idx * 0.17) % 1.0) % 1.0;
      const px = a.sx + (b.sx - a.sx) * t;
      const py = a.sy + (b.sy - a.sy) * t;
      const rad = sig.ambient ? 2.1 : Math.min(5.0, 2.5 + mag * 2.1);
      ctx.fillStyle = sig.ambient ? "rgba(255,176,32,0.55)" : Number(sig.flux || 0) >= 0 ? "#ffb020" : "#ff5b84";
      ctx.beginPath();
      ctx.arc(px, py, rad, 0, Math.PI * 2);
      ctx.fill();
    });

    projectedList
      .slice()
      .sort((p, q) => p.z - q.z)
      .forEach(({ sx, sy, raw }) => {
        const radius = raw.role === "hidden" ? 6.5 : 8;
        ctx.fillStyle = !raw.active
          ? "#ffb020"
          : raw.role === "input"
          ? "#26e3a8"
          : raw.role === "output"
          ? "#ffffff"
          : raw.excitatory
          ? "#00e5ff"
          : "#9d6bff";
        ctx.beginPath();
        ctx.arc(sx, sy, radius + Math.min(4.5, Math.abs(raw.a) * 3.2), 0, Math.PI * 2);
        ctx.fill();
        if (state.showPhaseOverlay && raw.active && Math.abs(Number(raw.kuramoto_phase || 0)) > 1e-4) {
          ctx.strokeStyle = `hsl(${kuramotoPhaseHue(raw.kuramoto_phase)}, 90%, 62%)`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(sx, sy, radius + Math.min(4.5, Math.abs(raw.a) * 3.2) + 4, 0, Math.PI * 2);
          ctx.stroke();
        }
        if (state.selectedNodeId === raw.neuron_id) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      });
  }

  $("#btn-conn-zoom-in").addEventListener("click", () => { state.camZoom = Math.min(2.5, state.camZoom + 0.2); });
  $("#btn-conn-zoom-out").addEventListener("click", () => { state.camZoom = Math.max(0.45, state.camZoom - 0.2); });
  $("#btn-conn-reset-cam").addEventListener("click", () => { state.camRotX = 0.22; state.camRotY = 0.35; state.camZoom = 1.0; });
  $("#playback-slider").addEventListener("input", () => {
    refreshConnectome(Number($("#playback-slider").value));
  });

  if ($("#conn-reward-slider")) {
    $("#conn-reward-slider").addEventListener("input", (ev) => {
      state.manualRpeTouched = true;
      const val = Number(ev.target.value);
      const out = $("#conn-reward-val");
      if (out) out.textContent = `${val >= 0 ? "+" : ""}${val.toFixed(2)}`;
    });
  }

  if ($("#btn-stimulate-conn")) {
    $("#btn-stimulate-conn").addEventListener("click", () => stimulateNeuron(null));
  }

  if ($("#conn-plasticity-mode")) {
    $("#conn-plasticity-mode").addEventListener("change", () => stimulateNeuron(null));
  }

  function startTelemetryStream() {
    if (typeof EventSource === "undefined" || state.telemetryEventSource) return;
    const sseEl = $("#hud-sse-status");
    const qs = state.selectedOrganismId
      ? `?interval_ms=650&organism_id=${encodeURIComponent(state.selectedOrganismId)}`
      : "?interval_ms=650";
    try {
      const es = new EventSource(`/api/stream/telemetry${qs}`);
      es.addEventListener("telemetry", (ev) => {
        try {
          const d = JSON.parse(ev.data);
          if (d.tick !== undefined && $("#header-tick")) $("#header-tick").textContent = String(d.tick);
          if (d.recent_signals_count !== undefined && $("#hud-signals-count")) {
            $("#hud-signals-count").textContent = String(d.recent_signals_count);
          }
          if (d.last_weight_delta_norm !== undefined && $("#hud-dw-norm")) {
            $("#hud-dw-norm").textContent = Number(d.last_weight_delta_norm).toFixed(5);
          }
          if ($("#frontier-hook-telemetry") && d.frontier_hook) {
            $("#frontier-hook-telemetry").textContent = formatFrontierHookTelemetry(d.frontier_hook);
          }
          if (sseEl) sseEl.textContent = "STREAMING";
        } catch (_) {}
      });
      es.onerror = () => {
        if (sseEl) sseEl.textContent = "RECONNECT";
      };
      state.telemetryEventSource = es;
      if (sseEl) sseEl.textContent = "STREAMING";
    } catch (_) {}
  }

  function stopTelemetryStream() {
    if (state.telemetryEventSource) {
      state.telemetryEventSource.close();
      state.telemetryEventSource = null;
    }
    const sseEl = $("#hud-sse-status");
    if (sseEl) sseEl.textContent = "IDLE";
  }

  if ($("#btn-toggle-signals")) {
    $("#btn-toggle-signals").addEventListener("click", (ev) => {
      state.signalsPaused = !state.signalsPaused;
      ev.currentTarget.textContent = state.signalsPaused ? "Resume Pulses" : "Pause Pulses";
      announce(state.signalsPaused ? "Live signal pulses paused" : "Live signal pulses resumed");
    });
  }

  if ($("#btn-toggle-phase-overlay")) {
    $("#btn-toggle-phase-overlay").addEventListener("click", (ev) => {
      state.showPhaseOverlay = !state.showPhaseOverlay;
      ev.currentTarget.textContent = state.showPhaseOverlay ? "Phase Overlay: ON" : "Phase Overlay: OFF";
      if (state.connectomeGraph) rebuildThreeGraph(state.connectomeGraph);
      announce(state.showPhaseOverlay ? "Kuramoto phase overlay enabled on 3D neurons" : "Kuramoto phase overlay hidden");
    });
  }

  $("#btn-conn-diagnostics")?.addEventListener("click", async () => {
    const qs = state.selectedOrganismId ? `?organism_id=${encodeURIComponent(state.selectedOrganismId)}` : "";
    const res = await api(`/api/connectome/diagnostics${qs}`, { method: "POST" });
    const c = res.connectivity || {};
    const sae = res.sparse_dictionary || {};
    const box = $("#connectome-diagnostics-box");
    if (box) {
      box.innerHTML =
        `<strong>Connectivity (${res.steps_probed} steps):</strong> FC=${Number(c.mean_fc || 0).toFixed(4)} · ` +
        `EC=${Number(c.mean_ec || 0).toFixed(4)} · Asym=${Number(c.asymmetry_index || 0).toFixed(4)}<br>` +
        `<strong>Top-K SAE (K=${sae.k_active || 3}):</strong> MSE=${Number(sae.recon_mse || 0).toFixed(5)} · ` +
        `Causal Shift=${Number(sae.causal_ablation_shift || 0).toFixed(4)} vs Ctrl=${Number(sae.control_ablation_shift || 0).toFixed(4)} ` +
        `(${Number(sae.causal_specificity_ratio || 1).toFixed(2)}x)<br>` +
        `<strong>BCM Mean θ:</strong> ${Number(res.mean_metaplastic_theta || 0.35).toFixed(4)} · ` +
        `<strong>Kuramoto R(t):</strong> ${Number(res.kuramoto_order_parameter_R || 0).toFixed(4)}`;
    }
    announce(`Connectome FC/EC & Top-K SAE audit complete (specificity=${sae.causal_specificity_ratio || 1}x).`);
  });

  if ($("#btn-live-plasticity")) {
    $("#btn-live-plasticity").addEventListener("click", (ev) => {
      const btn = ev.currentTarget;
      if (state.livePlasticityTimer) {
        clearInterval(state.livePlasticityTimer);
        state.livePlasticityTimer = null;
        stopTelemetryStream();
        btn.textContent = "Live Stream: OFF";
        btn.classList.remove("btn-stream-active");
        announce("Live plasticity stream stopped");
      } else {
        btn.textContent = "Live Stream: ON";
        btn.classList.add("btn-stream-active");
        runLivePlasticityStep();
        startTelemetryStream();
        state.livePlasticityTimer = setInterval(() => {
          if (state.currentView === "connectome" && !state.offscreenPaused) {
            runLivePlasticityStep();
          }
        }, 450);
        announce("Live plasticity + SSE telemetry stream started");
      }
    });
  }

  function renderPlasticityOscilloscope(history) {
    const canvas = $("#plasticity-oscilloscope-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 3; i++) {
      const y = (h * i) / 4;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const meta = $("#oscilloscope-meta");
    if (!Array.isArray(history) || !history.length) {
      if (meta) meta.textContent = "Awaiting plasticity ticks";
      ctx.fillStyle = "rgba(148, 163, 184, 0.65)";
      ctx.font = "12px monospace";
      ctx.fillText("Click 'Pulse & Induce Plasticity' or enable 'Live Stream' to record synaptic plasticity waves.", 16, h / 2);
      return;
    }

    if (meta) {
      meta.textContent = `${history.length} ticks · step ${history[0].step} → ${history[history.length - 1].step}`;
    }

    const pad = 16;
    const plotW = w - pad * 2;
    const plotH = h - pad * 2;
    const n = history.length;
    const xAt = (idx) => (n === 1 ? pad + plotW / 2 : pad + (idx / (n - 1)) * plotW);
    const maxDw = Math.max(0.05, ...history.map((pt) => Number(pt.weight_delta_norm || 0)));
    const maxCount = Math.max(1, ...history.map((pt) => Math.max(Number(pt.ltp_count || 0), Number(pt.ltd_count || 0))));

    history.forEach((pt, idx) => {
      const x = xAt(idx);
      const ltpH = (Number(pt.ltp_count || 0) / maxCount) * (plotH * 0.25);
      const ltdH = (Number(pt.ltd_count || 0) / maxCount) * (plotH * 0.25);
      if (ltpH > 0.5) {
        ctx.fillStyle = "rgba(38, 227, 168, 0.24)";
        ctx.fillRect(x - 2, h - pad - ltpH, 4, ltpH);
      }
      if (ltdH > 0.5) {
        ctx.fillStyle = "rgba(255, 91, 132, 0.22)";
        ctx.fillRect(x + 2, h - pad - ltdH, 3, ltdH);
      }
    });

    const drawSeries = (color, getY) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      history.forEach((pt, idx) => {
        const x = xAt(idx);
        const y = Math.max(pad, Math.min(h - pad, getY(pt)));
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawSeries("#00e5ff", (pt) => h - pad - (Number(pt.weight_delta_norm || 0) / maxDw) * plotH * 0.88);
    drawSeries("#ffb020", (pt) => h - pad - ((Number(pt.dopamine || 0) + 1.0) / 2.0) * plotH * 0.85);
    drawSeries("#26e3a8", (pt) => h - pad - Math.max(0, Math.min(1, Number(pt.acetylcholine || 0.5))) * plotH * 0.85);
    drawSeries("#ff5b84", (pt) => h - pad - Math.max(0, Math.min(1, Number(pt.norepinephrine || 0.2))) * plotH * 0.85);
  }

  function renderPlasticityEvents(graph) {
    const list = $("#plasticity-events-list");
    if (!list) return;
    const pEvents = (graph.recent_plasticity_events || []).slice(0, 14).map((ev) => {
      const badge = ev.kind === "LTP" ? `<span class="ltp-badge">LTP</span>` : `<span class="ltd-badge">LTD</span>`;
      const sign = Number(ev.dw || 0) >= 0 ? "+" : "";
      return `<li>${badge}<strong>${escapeHtml(ev.synapse_id)}</strong> · Δw=${sign}${Number(ev.dw).toFixed(5)} · w=${Number(ev.weight).toFixed(4)} · e=${Number(ev.eligibility).toFixed(3)}</li>`;
    });
    const sEvents = (graph.structural_events || []).slice(-6).reverse().map(
      (se) =>
        `<li><span class="struct-badge">STRUCT</span>[t=${se.tick ?? 0}] <strong>${escapeHtml(se.type || "event")}</strong> ${se.count ? `count=${se.count}` : se.neuron_id ? escapeHtml(se.neuron_id) : ""}</li>`
    );
    const items = [...pEvents, ...sEvents];
    list.innerHTML = items.length
      ? items.join("")
      : `<li>No plasticity events recorded yet — click Pulse &amp; Induce Plasticity or start Live Stream.</li>`;
  }

  function updatePlasticityVisuals(data) {
    const graph = data.graph || {};
    const nm = graph.neuromodulators || {};
    const hist = graph.plasticity_history || [];
    const latest = hist.length ? hist[hist.length - 1] : null;
    const pEvents = graph.recent_plasticity_events || [];
    const ltpCount = latest ? latest.ltp_count : pEvents.filter((e) => e.kind === "LTP").length;
    const ltdCount = latest ? latest.ltd_count : pEvents.filter((e) => e.kind === "LTD").length;

    const sigEl = $("#hud-signals-count");
    if (sigEl) sigEl.textContent = String((graph.recent_signals || []).length);
    const dwEl = $("#hud-dw-norm");
    if (dwEl) dwEl.textContent = Number(graph.last_weight_delta_norm || 0).toFixed(5);
    const ltpEl = $("#hud-ltp-ltd");
    if (ltpEl) ltpEl.textContent = `${ltpCount} / ${ltdCount}`;
    const nmEl = $("#hud-neuromod");
    if (nmEl) {
      nmEl.textContent = `DA ${Number(nm.dopamine || 0).toFixed(2)} · ACh ${Number(nm.acetylcholine ?? 0.5).toFixed(2)} · NE ${Number(nm.norepinephrine ?? 0.2).toFixed(2)}`;
    }
    const synEl = $("#hud-synops");
    if (synEl) synEl.textContent = String(graph.total_synops || 0);
    const modeBadge = $("#plasticity-mode-badge");
    if (modeBadge) modeBadge.textContent = `Mode: ${data.plasticity_mode || "three_factor"}`;
    const modeSel = $("#conn-plasticity-mode");
    if (modeSel && data.plasticity_mode && document.activeElement !== modeSel) {
      modeSel.value = data.plasticity_mode;
    }

    renderPlasticityOscilloscope(hist);
    renderPlasticityEvents(graph);
  }

  function applyConnectomePayload(data, checkpointIdx = null) {
    const slider = $("#playback-slider");
    const maxIdx = Math.max(0, (data.checkpoints_count || 1) - 1);
    slider.max = String(maxIdx);
    if (checkpointIdx === null) slider.value = String(maxIdx);

    state.connectomeGraph = data.graph;
    rebuildThreeGraph(data.graph);
    updatePlasticityVisuals(data);

    if (state.selectedNodeId) {
      const updatedNode = (data.graph?.nodes || []).find((n) => n.neuron_id === state.selectedNodeId);
      if (updatedNode) inspectNode(updatedNode);
    }

    const t = data.topology;
    const ceEl = $("#hud-causal-emergence");
    if (ceEl) ceEl.textContent = Number(t.causal_emergence_index || 0).toFixed(4);
    const somaSel = $("#conn-neuron-model");
    if (somaSel && data.neuron_model && document.activeElement !== somaSel) {
      somaSel.value = data.neuron_model;
    }

    renderKV("#topology-metrics", [
      ["Neurons", t.num_neurons],
      ["Synapses", t.num_synapses],
      ["Density D", t.density.toFixed(4)],
      ["Clustering", t.clustering_coefficient.toFixed(4)],
      ["Modularity Q", t.modularity.toFixed(4)],
      ["Avg Path Len", t.avg_path_length.toFixed(3)],
      ["Redundant Paths", t.redundant_pathways],
      ["Isolated", t.isolated_neurons.length],
      ["Causal Emergence", Number(t.causal_emergence_index || 0).toFixed(4)],
      ["Effective Info EI", Number(t.effective_information_macro || 0).toFixed(3)],
      ["Fractal Dim D_f", Number(t.fractal_dimension_proxy || 1.0).toFixed(3)],
      ["NeuroGrafts", t.graft_synapses ?? 0],
      ["Wiring Len L", Number(t.total_wiring_length || 0).toFixed(1)],
      ["Cable Delay", `${Number(t.mean_cable_delay_ms || 0).toFixed(2)} ms`],
    ]);

    $("#cognitive-state-pre").textContent = JSON.stringify(data.cognitive_state || { note: "Step simulation to populate C_t" }, null, 2);
  }

  async function modifyConnectomeArchitecture(operation, extra = {}) {
    init3DConnectomeHost();
    const somaEl = $("#conn-neuron-model");
    const data = await api("/api/connectome/neurograft", {
      method: "POST",
      body: JSON.stringify({
        organism_id: state.selectedOrganismId,
        operation,
        neuron_model: somaEl ? somaEl.value : undefined,
        ...extra,
      }),
    });
    applyConnectomePayload(data, null);
    if (data.graft_result) {
      const gr = data.graft_result;
      announce(
        `Connectome ${gr.operation} applied (model=${data.neuron_model}, grafts=${data.topology?.graft_synapses ?? 0}, CE=${data.topology?.causal_emergence_index ?? 0}).`
      );
    }
  }

  $("#conn-neuron-model")?.addEventListener("change", () => modifyConnectomeArchitecture("switch_model"));
  $("#btn-inject-graft")?.addEventListener("click", () => modifyConnectomeArchitecture("neurograft", { max_grafts: 3 }));
  $("#btn-fractal-reseed")?.addEventListener("click", () => modifyConnectomeArchitecture("fractal", { branch_factor: 2 }));

  async function stimulateNeuron(neuronId = null) {
    init3DConnectomeHost();
    const modeEl = $("#conn-plasticity-mode");
    const rpeEl = $("#conn-reward-slider");
    const data = await api("/api/connectome/stimulate", {
      method: "POST",
      body: JSON.stringify({
        organism_id: state.selectedOrganismId,
        neuron_id: neuronId,
        current: neuronId ? 1.45 : 1.25,
        reward_signal: rpeEl ? Number(rpeEl.value) : 0.85,
        learning_rate: 0.12,
        plasticity_mode: modeEl ? modeEl.value : "neuromodulated",
        structural_adapt: true,
      }),
    });
    applyConnectomePayload(data, null);
    if (data.stimulation) {
      const s = data.stimulation;
      announce(
        `Stimulated ${s.stimulated_neuron} (${s.plasticity_mode}) · ${s.signals_emitted} live signals · ||ΔW||=${s.weight_delta_norm}`
      );
    }
  }

  async function runLivePlasticityStep() {
    if (state.liveStepInFlight) return;
    state.liveStepInFlight = true;
    try {
      init3DConnectomeHost();
      const modeEl = $("#conn-plasticity-mode");
      const rpeEl = $("#conn-reward-slider");
      const data = await api("/api/connectome/plasticity-step", {
        method: "POST",
        body: JSON.stringify({
          organism_id: state.selectedOrganismId,
          steps: 1,
          plasticity_mode: modeEl ? modeEl.value : "neuromodulated",
          reward_override: state.manualRpeTouched && rpeEl ? Number(rpeEl.value) : null,
        }),
      });
      applyConnectomePayload(data, null);
      if (data.command_centre) {
        const tickEl = $("#metric-tick");
        if (tickEl && data.command_centre.tick !== undefined) {
          tickEl.textContent = String(data.command_centre.tick);
        }
      }
    } finally {
      state.liveStepInFlight = false;
    }
  }

  async function refreshConnectome(checkpointIdx = null) {
    init3DConnectomeHost();
    const params = new URLSearchParams();
    if (state.selectedOrganismId) params.set("organism_id", state.selectedOrganismId);
    if (checkpointIdx !== null && Number.isFinite(checkpointIdx)) {
      params.set("checkpoint_index", String(checkpointIdx));
    }
    const qs = params.toString() ? `?${params.toString()}` : "";
    const data = await api(`/api/connectome${qs}`);
    applyConnectomePayload(data, checkpointIdx);
  }

  // View C: Memory Observatory
  $("#btn-consolidate-mem").addEventListener("click", async () => {
    const qs = state.selectedOrganismId ? `?organism_id=${encodeURIComponent(state.selectedOrganismId)}` : "";
    const res = await api(`/api/memory/consolidate${qs}`, { method: "POST" });
    renderMemoryData(res.snapshot);
    announce(`Consolidated ${res.promoted} memories and replayed ${res.replayed} episodes.`);
  });

  async function refreshMemory() {
    const qs = state.selectedOrganismId ? `?organism_id=${encodeURIComponent(state.selectedOrganismId)}` : "";
    const snap = await api(`/api/memory${qs}`);
    renderMemoryData(snap);
    await runMemoryHybridSearch();
  }

  async function runMemoryHybridSearch() {
    const list = $("#memory-hybrid-results");
    if (!list) return;
    const queryText = $("#memory-hybrid-query")?.value || "";
    const res = await api("/api/memory/hybrid-search", {
      method: "POST",
      body: JSON.stringify({
        organism_id: state.selectedOrganismId,
        query_text: queryText,
        top_k: 6,
      }),
    });
    const hits = res.results || [];
    list.innerHTML = hits.length
      ? hits
          .map(
            (h) =>
              `<li><span class="ltp-badge">RRF ${Number(h.rrf_score).toFixed(4)}</span><strong>${escapeHtml(h.memory_id)}</strong> (${escapeHtml(h.memory_class)} · ${escapeHtml(h.provenance)}) · BM25=${Number(h.bm25_score).toFixed(3)} · Vec=${Number(h.vector_score).toFixed(3)} · R=${Number(h.reward_assoc).toFixed(2)}</li>`
          )
          .join("")
      : `<li>No memory records matched query.</li>`;
  }

  $("#btn-memory-hybrid-search")?.addEventListener("click", async () => {
    await runMemoryHybridSearch();
    announce("Executed NexusSearch BM25 + Vector Reciprocal Rank Fusion memory query.");
  });

  $("#btn-carrier-assay")?.addEventListener("click", async () => {
    const qs = state.selectedOrganismId ? `?organism_id=${encodeURIComponent(state.selectedOrganismId)}` : "";
    const res = await api(`/api/memory/carrier-assay${qs}`, { method: "POST" });
    const box = $("#carrier-assay-box");
    if (box) {
      const caps = res.carrier_capacities || {};
      const drops = res.single_removal_drop || {};
      const mat = (res.factorial_matrix || [])
        .filter((r) => ["1111", "0111", "1011", "1101", "1110", "1100", "0011", "0000"].includes(r.mask))
        .map((r) => `[${r.mask}] ret=${Number(r.retention_score).toFixed(3)} (gap=${Number(r.reconstruction_gap).toFixed(3)})`)
        .join(" · ");
      box.innerHTML =
        `<strong>${escapeHtml(res.assay_protocol)}</strong> · Dominant Carrier: <strong>${escapeHtml(res.dominant_carrier)}</strong> · Synergy=${res.synergy_index}<br>` +
        `Capacities: C1(Weights)=${caps.C1_synaptic_weights} · C2(Traces)=${caps.C2_episodic_working_traces} · C3(EWC)=${caps.C3_ewc_consolidated_anchors} · C4(Grafts)=${caps.C4_neuromodulator_grafts}<br>` +
        `Single-Removal Drop: ΔC1=${drops.C1_synaptic_weights} · ΔC2=${drops.C2_episodic_working_traces} · ΔC3=${drops.C3_ewc_consolidated_anchors} · ΔC4=${drops.C4_neuromodulator_grafts}<br>` +
        `2^4 Matrix Sample: ${escapeHtml(mat)}`;
    }
    announce(`Completed 2^4 Carrier Archaeology Assay (dominant carrier: ${res.dominant_carrier}).`);
  });

  function renderMemoryData(snap) {
    const counts = snap.class_counts || {};
    const labels = Object.keys(counts);
    const canvas = $("#memory-bars-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const maxC = Math.max(1, ...Object.values(counts));
    const barW = Math.floor((canvas.width - 40) / Math.max(1, labels.length));

    labels.forEach((lbl, idx) => {
      const val = counts[lbl] || 0;
      const bh = (val / maxC) * (canvas.height - 60);
      const x = 24 + idx * barW;
      const y = canvas.height - 34 - bh;
      ctx.fillStyle = idx % 2 === 0 ? "#00e5ff" : "#9d6bff";
      ctx.fillRect(x, y, Math.max(18, barW - 14), bh);
      ctx.fillStyle = "#eef4ff";
      ctx.font = "11px monospace";
      ctx.fillText(String(val), x + 4, y - 6);
      ctx.fillStyle = "#93a7cb";
      ctx.fillText(lbl.slice(0, 7), x, canvas.height - 14);
    });

    const st = snap.stats || {};
    renderKV("#memory-stats-kv", [
      ["Total", `${snap.total_memories}/${snap.max_memories}`],
      ["Encoded", st.encoded || 0],
      ["Retrieved", st.retrieved || 0],
      ["Consolidated", st.consolidated || 0],
      ["Replayed", st.replayed || 0],
      ["Replay Cost", `${(st.last_replay_overhead_ms || 0).toFixed(3)} ms`],
    ]);

    const gCanvas = $("#memory-graph-canvas");
    const gCtx = gCanvas.getContext("2d");
    gCtx.clearRect(0, 0, gCanvas.width, gCanvas.height);
    const nodes = snap.graph?.nodes || [];
    const pos = {};
    nodes.forEach((n, i) => {
      const ang = (2 * Math.PI * i) / Math.max(1, nodes.length);
      pos[n.id] = {
        x: gCanvas.width / 2 + Math.cos(ang) * 90,
        y: gCanvas.height / 2 + Math.sin(ang) * 80,
        node: n,
      };
    });
    gCtx.strokeStyle = "rgba(157,107,255,0.45)";
    (snap.graph?.edges || []).forEach((e) => {
      const a = pos[e.source];
      const b = pos[e.target];
      if (a && b) {
        gCtx.beginPath();
        gCtx.moveTo(a.x, a.y);
        gCtx.lineTo(b.x, b.y);
        gCtx.stroke();
      }
    });
    Object.values(pos).forEach(({ x, y, node }) => {
      gCtx.fillStyle = node.reward >= 0 ? "#00e5ff" : "#ffb020";
      gCtx.beginPath();
      gCtx.arc(x, y, 4 + node.importance * 5, 0, Math.PI * 2);
      gCtx.fill();
    });

    const conflicts = snap.conflicts || [];
    $("#memory-conflicts-box").textContent = conflicts.length
      ? `${conflicts.length} reward-valence conflict(s) detected (top similarity: ${conflicts[0].similarity}).`
      : "No contradictory memory valence conflicts detected.";
  }

  // View D: Artificial Life Laboratory
  $("#btn-spawn-org").addEventListener("click", async () => {
    const model = $("#spawn-model-select").value;
    const useDeepRl = Boolean($("#spawn-deep-rl-check")?.checked);
    const res = await api("/api/ecosystem/spawn", {
      method: "POST",
      body: JSON.stringify({ neuron_model: model, use_deep_rl: useDeepRl }),
    });
    await refreshAll();
    announce(res.added ? `Spawned ${res.organism.organism_id} (${model}${useDeepRl ? ", Deep-RL" : ""}).` : res.reason);
  });

  $("#btn-evolve-gen").addEventListener("click", async () => {
    const res = await api("/api/ecosystem/evolve", { method: "POST" });
    await refreshAll();
    announce(`Evolved Generation ${res.evolution.generation} (best fitness ${res.evolution.best_fitness}).`);
  });

  function drawLineageTree(lineage) {
    const canvas = $("#lineage-tree-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!lineage.length) {
      ctx.fillStyle = "#93a7cb";
      ctx.font = "12px monospace";
      ctx.fillText("Spawn or evolve organisms to visualize lineage tree.", 24, canvas.height / 2);
      return;
    }

    const byGen = new Map();
    lineage.forEach((node) => {
      const g = Number(node.generation || 0);
      if (!byGen.has(g)) byGen.set(g, []);
      byGen.get(g).push(node);
    });
    const gens = Array.from(byGen.keys()).sort((a, b) => a - b);
    const coords = {};
    const padX = 48;
    const padY = 26;

    gens.forEach((g, gxIdx) => {
      const items = byGen.get(g);
      const x = gens.length === 1 ? canvas.width / 2 : padX + (gxIdx / (gens.length - 1)) * (canvas.width - 2 * padX);
      items.forEach((item, iyIdx) => {
        const y = items.length === 1 ? canvas.height / 2 : padY + (iyIdx / (items.length - 1)) * (canvas.height - 2 * padY);
        coords[item.organism_id] = { x, y, item };
      });
    });

    ctx.strokeStyle = "rgba(157, 107, 255, 0.55)";
    ctx.lineWidth = 1.5;
    lineage.forEach((node) => {
      const childPos = coords[node.organism_id];
      (node.parent_ids || []).forEach((pid) => {
        const parentPos = coords[pid];
        if (parentPos && childPos) {
          ctx.beginPath();
          ctx.moveTo(parentPos.x, parentPos.y);
          ctx.lineTo(childPos.x, childPos.y);
          ctx.stroke();
        }
      });
    });

    Object.values(coords).forEach(({ x, y, item }) => {
      ctx.fillStyle = item.alive === false ? "#93a7cb" : "#00e5ff";
      ctx.beginPath();
      ctx.arc(x, y, 5.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#eef4ff";
      ctx.font = "10px monospace";
      ctx.fillText(`G${item.generation}`, x - 8, y - 9);
    });
  }

  async function refreshALife() {
    const eco = await api("/api/ecosystem");
    const tbody = $("#population-tbody");
    tbody.innerHTML = (eco.organisms || [])
      .map(
        (o) => `<tr>
          <th scope="row">${escapeHtml(o.organism_id)}</th>
          <td data-label="Gen">${o.generation}</td>
          <td data-label="Species">${escapeHtml(o.species_id)}</td>
          <td data-label="Stage">${escapeHtml(o.stage)}</td>
          <td data-label="Energy">${o.energy}</td>
          <td data-label="Fitness">${o.fitness}</td>
          <td data-label="Neurons/Syn">${o.active_neurons}/${o.synapses}</td>
        </tr>`
      )
      .join("");

    const sel = $("#organism-select");
    const prev = state.selectedOrganismId;
    sel.innerHTML = (eco.organisms || [])
      .map((o) => `<option value="${escapeHtml(o.organism_id)}">${escapeHtml(o.organism_id)} (F=${o.fitness})</option>`)
      .join("");
    if (prev && (eco.organisms || []).some((o) => o.organism_id === prev)) {
      sel.value = prev;
    } else if (eco.organisms?.length) {
      state.selectedOrganismId = eco.organisms[0].organism_id;
      sel.value = state.selectedOrganismId;
    }

    const canvas = $("#fitness-dist-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const fits = eco.fitness_distribution || [];
    fits.forEach((f, idx) => {
      const bw = Math.max(14, (canvas.width - 40) / Math.max(1, fits.length) - 6);
      const norm = Math.max(0.08, Math.min(1.0, (f + 1.0) / 3.0));
      const bh = norm * (canvas.height - 40);
      ctx.fillStyle = f >= 0 ? "#00e5ff" : "#ffb020";
      ctx.fillRect(20 + idx * (bw + 6), canvas.height - 20 - bh, bw, bh);
    });

    $("#social-events-list").innerHTML = (eco.recent_social_events || [])
      .slice()
      .reverse()
      .map((ev) => `<li>[t=${ev.tick}] ${escapeHtml(ev.type)} · ${escapeHtml(JSON.stringify(ev))}</li>`)
      .join("") || "<li>No cooperation/competition events recorded yet.</li>";

    const tBox = $("#tessera-library-box");
    if (tBox) {
      const tl = eco.tessera_library || {};
      const tg = eco.trophic_guilds || {};
      const pubs = (tl.published_instructions || [])
        .map((p) => `${p.motif_id} (discoverers=${p.discoverer_count}, lineages=${p.independent_lineage_count})`)
        .join(" · ") || "None yet";
      tBox.innerHTML =
        `<strong>${escapeHtml(tl.protocol || "TESSERA")}</strong> · Published Instructions: <strong>${tl.published_count ?? 0}</strong> / ${tl.candidate_count ?? 0} candidates · Species Diversity H=${eco.species_diversity_index ?? 0}<br>` +
        `Evo-OMEGA Trophic Guilds: Autotrophs=${tg.autotroph_synthesizer ?? 0} · Foragers=${tg.forager_herbivore ?? 0} · Symbionts=${tg.cooperative_symbiont ?? 0}<br>` +
        `Published Set: ${escapeHtml(pubs)}`;
    }

    drawLineageTree(eco.lineage || []);
  }

  $("#btn-tessera-scan")?.addEventListener("click", async () => {
    const res = await api("/api/ecosystem/tessera-discover", { method: "POST" });
    await refreshALife();
    announce(`TESSERA independent-rediscovery scan completed (${res.published_count} published instructions).`);
  });

  // View E: World Simulator
  $("#btn-task-0").addEventListener("click", async () => {
    await api("/api/world/task", { method: "POST", body: JSON.stringify({ task_id: 0 }) });
    await refreshWorld();
    announce("DIAMOND switched to Task 0 (Standard Foraging).");
  });

  $("#btn-task-1").addEventListener("click", async () => {
    await api("/api/world/task", { method: "POST", body: JSON.stringify({ task_id: 1 }) });
    await refreshWorld();
    announce("DIAMOND switched to Task 1 (Shifted Beacon).");
  });

  $("#btn-counterform-id")?.addEventListener("click", async () => {
    const res = await api("/api/world/counterform-id", { method: "POST" });
    const box = $("#counterform-box");
    if (box) {
      const ab = res.tuned_absorber || {};
      const laws = (res.competing_laws || [])
        .map((l) => `${l.law_id} (RMSE=${l.held_out_rmse})`)
        .join(" · ");
      box.innerHTML =
        `<strong>${escapeHtml(res.protocol)}</strong> · Winner: <strong>${escapeHtml(res.winning_law)}</strong> [${escapeHtml(res.winning_equation)}]<br>` +
        `Tuned Absorber: f_opt=${ab.frequency_ratio_f_opt} · ζ_opt=${ab.damping_ratio_zeta_opt} · Peak ${ab.peak_amplitude_before} → ${ab.peak_amplitude_after} (-${ab.attenuation_db} dB)<br>` +
        `Competing Laws: ${escapeHtml(laws)} · Receipt SHA-256: <code>${escapeHtml(res.receipt_sha256)}</code>`;
    }
    announce(`COUNTERFORM identified ${res.winning_law} (attenuation ${res.tuned_absorber?.attenuation_db} dB).`);
  });

  async function refreshWorld() {
    const w = await api("/api/world");
    const canvas = $("#world-canvas");
    const ctx = canvas.getContext("2d");
    const cw = canvas.width / w.width;
    const ch = canvas.height / w.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Subtle grid
    ctx.strokeStyle = "rgba(110,168,254,0.1)";
    for (let x = 0; x <= w.width; x++) {
      ctx.beginPath();
      ctx.moveTo(x * cw, 0);
      ctx.lineTo(x * cw, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y <= w.height; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * ch);
      ctx.lineTo(canvas.width, y * ch);
      ctx.stroke();
    }

    // Obstacles
    ctx.fillStyle = "#1d2948";
    (w.obstacles || []).forEach(([x, y]) => ctx.fillRect(x * cw + 2, y * ch + 2, cw - 4, ch - 4));

    // Hazards
    ctx.fillStyle = "rgba(255, 176, 32, 0.78)";
    (w.hazards || []).forEach(([x, y]) => {
      ctx.beginPath();
      ctx.moveTo((x + 0.5) * cw, y * ch + 4);
      ctx.lineTo((x + 1) * cw - 4, (y + 1) * ch - 4);
      ctx.lineTo(x * cw + 4, (y + 1) * ch - 4);
      ctx.closePath();
      ctx.fill();
    });

    // Resources
    ctx.fillStyle = "#26e3a8";
    (w.resources || []).forEach((r) => {
      ctx.beginPath();
      ctx.arc((r.x + 0.5) * cw, (r.y + 0.5) * ch, cw * 0.28, 0, Math.PI * 2);
      ctx.fill();
    });

    // Beacon
    if (w.beacon) {
      ctx.strokeStyle = "#9d6bff";
      ctx.lineWidth = 3;
      ctx.strokeRect(w.beacon[0] * cw + 3, w.beacon[1] * ch + 3, cw - 6, ch - 6);
    }

    // Moving entities
    ctx.fillStyle = "#ff5b84";
    (w.moving_entities || []).forEach((m) => {
      ctx.fillRect(m.x * cw + cw * 0.22, m.y * ch + ch * 0.22, cw * 0.56, ch * 0.56);
    });

    // Agents
    (w.agents || []).forEach((a) => {
      ctx.fillStyle = "#00e5ff";
      ctx.beginPath();
      ctx.arc((a.x + 0.5) * cw, (a.y + 0.5) * ch, cw * 0.36, 0, Math.PI * 2);
      ctx.fill();
    });

    renderKV("#world-kv", [
      ["World Size", `${w.width}x${w.height}`],
      ["Tick", w.tick],
      ["Active Task", w.task_id],
      ["Active Resources", (w.resources || []).length],
      ["Active Agents", (w.agents || []).length],
      ["Beacon", `(${w.beacon[0]}, ${w.beacon[1]})`],
    ]);

    $("#world-interactions-list").innerHTML = (w.recent_interactions || [])
      .slice()
      .reverse()
      .map((i) => `<li>[t=${i.tick}] ${escapeHtml(i.agent_id)} · ${escapeHtml(i.type)} @ (${i.x},${i.y})</li>`)
      .join("") || "<li>No interactions yet.</li>";
  }

  // View F: Research Laboratory
  $("#btn-run-exp").addEventListener("click", async () => {
    const name = $("#exp-select").value;
    announce(`Running Ghost Engine experiment: ${name}...`);
    await api("/api/research/experiment", {
      method: "POST",
      body: JSON.stringify({ name, seeds: [42, 101, 202], steps: 25 }),
    });
    await refreshResearch();
    announce(`Experiment ${name} completed.`);
  });

  $("#btn-run-bench").addEventListener("click", async () => {
    announce("Running Architecture A-E Benchmark Suite...");
    await api("/api/research/benchmark", {
      method: "POST",
      body: JSON.stringify({ seeds: [42, 101, 202], episodes: 6, steps_per_episode: 20 }),
    });
    await refreshResearch();
    announce("Benchmark A-E completed and exported.");
  });

  $("#btn-propose-si").addEventListener("click", async () => {
    announce("Analyzing bottlenecks and evaluating sandbox proposal...");
    await api("/api/research/self-improve/propose", { method: "POST" });
    await refreshResearch();
    announce("Self-improvement proposal evaluated in sandbox.");
  });

  $("#btn-crosswire-verify")?.addEventListener("click", async () => {
    const qs = state.selectedOrganismId ? `?organism_id=${encodeURIComponent(state.selectedOrganismId)}` : "";
    const res = await api(`/api/governance/crosswire-verify${qs}`, { method: "POST" });
    const box = $("#proposals-box");
    if (box) {
      const adapters = (res.adapters || [])
        .map((a) => `${a.interpretation_id}: ${a.mealy_transition} (ok=${a.contract_satisfied})`)
        .join(" · ");
      box.innerHTML =
        `<strong>${escapeHtml(res.protocol)}</strong> · Verdict: <strong>${escapeHtml(res.verdict)}</strong> (SHA-256: <code>${escapeHtml(res.contract_sha256)}</code>)<br>` +
        `Surviving Interpretations (${res.surviving_interpretations}): ${escapeHtml(adapters)}`;
    }
    announce(`CROSSWIRE safety contract verification: ${res.verdict}.`);
  });

  $("#btn-export-research").addEventListener("click", () => {
    if (!state.lastResearchData) return;
    const blob = new Blob([JSON.stringify(state.lastResearchData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "omnis_research_export.json";
    a.click();
    URL.revokeObjectURL(url);
  });

  async function refreshResearch() {
    const res = await api("/api/research");
    state.lastResearchData = res;
    const bench = res.latest_benchmark;
    const tbody = $("#benchmark-tbody");
    if (bench?.architectures) {
      tbody.innerHTML = Object.entries(bench.architectures)
        .map(
          ([name, m]) => `<tr>
            <th scope="row">${escapeHtml(name)}</th>
            <td data-label="Task Perf">${m.task_performance_mean} ± ${m.task_performance_std}</td>
            <td data-label="Learning Speed">${m.learning_speed_mean}</td>
            <td data-label="Retention">${m.retention_mean}</td>
            <td data-label="Generalisation">${m.generalisation_mean}</td>
            <td data-label="Robustness">${m.robustness_mean}</td>
            <td data-label="Complexity">${m.network_complexity_mean}</td>
            <td data-label="J(theta)">${m.multi_objective_J}</td>
          </tr>`
        )
        .join("");
    } else {
      tbody.innerHTML = '<tr><td colspan="8">Click "Run A-D Benchmark" to generate multi-seed comparative measurements.</td></tr>';
    }

    $("#experiments-list").innerHTML = (res.experiments || [])
      .slice()
      .reverse()
      .map((e) => {
        const st = e.statistical_analysis || {};
        return `<li><strong>${escapeHtml(e.name)}</strong> [${escapeHtml(e.verdict.toUpperCase())}] · p=${(st.p_value ?? 1).toFixed(4)} · d=${(st.cohens_d ?? 0).toFixed(3)}</li>`;
      })
      .join("") || "<li>No experiments recorded in this session yet.</li>";

    const props = res.proposals || [];
    const propsBox = $("#proposals-box");
    if (props.length) {
      const latest = props[props.length - 1];
      propsBox.innerHTML =
        `Proposal <strong>${escapeHtml(latest.proposal_id)}</strong> [${escapeHtml(latest.status)}]<br>` +
        `Patch: ${escapeHtml(JSON.stringify(latest.patch))}<br>` +
        (latest.gate_passed && latest.status !== "deployed"
          ? `<button type="button" id="btn-approve-prop" class="btn btn-cyan mt-sm">Approve &amp; Deploy</button>`
          : "");
      const approveBtn = $("#btn-approve-prop");
      if (approveBtn) {
        approveBtn.addEventListener("click", async () => {
          await api("/api/research/self-improve/approve", {
            method: "POST",
            body: JSON.stringify({ proposal_id: latest.proposal_id }),
          });
          await refreshResearch();
          announce(`Deployed approved proposal ${latest.proposal_id}.`);
        });
      }
    }

    $("#audit-list").innerHTML = (res.audit_logs || [])
      .map((a) => `<li><strong>${escapeHtml(a.action)}</strong> by ${escapeHtml(a.actor)}</li>`)
      .join("");

    drawParetoChart(bench);
  }

  function drawParetoChart(bench) {
    const canvas = $("#pareto-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!bench?.architectures) {
      ctx.fillStyle = "#93a7cb";
      ctx.font = "12px monospace";
      ctx.fillText("Run A-E Benchmark to plot the multi-objective Pareto frontier.", 24, canvas.height / 2);
      return;
    }

    const entries = Object.entries(bench.architectures);
    const paretoSet = new Set(bench.pareto_optimal_architectures || []);
    const costs = entries.map(([, m]) => Number(m.compute_cost_mean_ms || 1));
    const perfs = entries.map(([, m]) => Number(m.task_performance_mean || 0));
    const minC = Math.min(...costs), maxC = Math.max(minC + 1, ...costs);
    const minP = Math.min(...perfs), maxP = Math.max(minP + 0.1, ...perfs);
    const pad = 34;

    ctx.strokeStyle = "rgba(147,167,203,0.35)";
    ctx.beginPath();
    ctx.moveTo(pad, pad);
    ctx.lineTo(pad, canvas.height - pad);
    ctx.lineTo(canvas.width - pad, canvas.height - pad);
    ctx.stroke();

    const mapped = entries.map(([name, m]) => {
      const x = pad + ((Number(m.compute_cost_mean_ms || 1) - minC) / (maxC - minC)) * (canvas.width - 2 * pad);
      const y = canvas.height - pad - ((Number(m.task_performance_mean || 0) - minP) / (maxP - minP)) * (canvas.height - 2 * pad);
      return { name, x, y, pareto: paretoSet.has(name), m };
    });

    const pPts = mapped.filter((p) => p.pareto).sort((a, b) => a.x - b.x);
    if (pPts.length >= 2) {
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = "#26e3a8";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      pPts.forEach((p, idx) => (idx === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.stroke();
      ctx.restore();
    }

    mapped.forEach((pt) => {
      ctx.fillStyle = pt.pareto ? "#00e5ff" : "#9d6bff";
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, pt.pareto ? 6 : 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#eef4ff";
      ctx.font = "10px monospace";
      ctx.fillText(pt.name.split("_")[0], pt.x + 8, pt.y + 3);
    });
  }

  // View G: Hardware & Supermix-Aether
  $("#btn-quantum-opt")?.addEventListener("click", async () => {
    $("#hardware-action-box").textContent = "Running simulated quantum-inspired tunneling annealing...";
    const res = await api("/api/hardware/quantum-optimize", { method: "POST", body: JSON.stringify({ iterations: 12 }) });
    $("#hardware-action-box").textContent =
      `Quantum-Inspired Annealing (physical QPU=${res.physical_qpu_connected}): ` +
      `best_obj=${res.best_objective}, params=${JSON.stringify(res.best_params)}`;
    announce("Quantum-inspired hyperparameter optimization completed.");
  });

  $("#btn-parallel-sweep")?.addEventListener("click", async () => {
    $("#hardware-action-box").textContent = "Executing parallel distributed worker sweep...";
    const res = await api("/api/hardware/parallel-sweep", { method: "POST", body: JSON.stringify({ steps: 20, mode: "thread" }) });
    $("#hardware-action-box").textContent =
      `Parallel Sweep (${res.execution_mode}, workers=${res.workers}, trials=${res.trials_evaluated}): ` +
      `wall=${res.wall_time_ms}ms, speedup=${res.speedup_factor}x, best=${res.best_config?.label} (F=${res.best_config?.fitness})`;
    announce(`Parallel sweep completed across ${res.workers} workers (${res.speedup_factor}x speedup).`);
  });

  $("#btn-pytorch-pass")?.addEventListener("click", async () => {
    const res = await api("/api/hardware/tensor-pass", { method: "POST", body: JSON.stringify({ precision: "float32" }) });
    $("#hardware-action-box").textContent =
      `PyTorch Tensor Pass (device=${res.device}, precision=${res.precision}): ` +
      `neurons=${res.neurons}, synapses=${res.active_synapses}, drive_norm=${res.drive_norm}, latency=${res.latency_ms}ms`;
    announce(`PyTorch tensor pass executed on ${res.device}.`);
  });

  $("#btn-hyperion-schedule")?.addEventListener("click", async () => {
    const res = await api("/api/hardware/hyperion-schedule", {
      method: "POST",
      body: JSON.stringify({ workload_type: "dual_soma", complexity: 1.5 }),
    });
    $("#hardware-action-box").textContent =
      `HYPERION-20 Scheduler (${res.total_service_classes} service classes): assigned=${res.assigned_service_class} ` +
      `(domain=${res.domain}, backend=${res.target_backend}, cost=${res.estimated_cost_units}, dispatches=${res.total_dispatches})`;
    await refreshHardware();
    announce(`HYPERION-20 dispatched workload to ${res.assigned_service_class}.`);
  });

  $("#btn-tether-audit")?.addEventListener("click", async () => {
    const res = await api("/api/hardware/tether-audit", { method: "POST", body: JSON.stringify({}) });
    const donors = (res.donor_reports || [])
      .map((d) => `${d.model_id}: fidelity=${d.tether_fidelity} (rel=${d.relevant_intervention_effect}, leak=${d.irrelevant_leakage})`)
      .join(" | ");
    $("#hardware-action-box").textContent =
      `${res.protocol}: status=${res.ensemble_status}, consensus_action=${res.consensus_action}, ` +
      `FAULTLINE_risk=${res.faultline_correlated_risk} · Donors: ${donors}`;
    announce(`Supermix-Tether & FAULTLINE audit complete (${res.ensemble_status}).`);
  });

  async function refreshHardware() {
    const data = await api("/api/hardware");
    const hw = data.hardware || {};
    const neuro = hw.neuromorphic_simulation || {};
    const hyp = hw.hyperion_20 || {};
    renderKV("#hardware-kv", [
      ["Active Backend", hw.active_backend],
      ["GPU Available", hw.gpu_available ? "Yes" : "No (CPU fallback)"],
      ["CPU Usage", `${hw.cpu_percent}% (${hw.cpu_count} cores)`],
      ["Process RAM", `${hw.process_memory_mb} MB`],
      ["Neuro Mode", neuro.simulated ? "SIMULATED AER" : "Hardware"],
      ["Physical Neuro Device", neuro.physical_device_connected ? "Connected" : "None"],
      ["Sparse Spike Ratio", neuro.sparse_activation_ratio ?? 0],
      ["Simulated SynOps", `${neuro.synops_executed ?? 0} (~${neuro.estimated_energy_pj ?? 0} pJ)`],
      ["HYPERION-20 Classes", `${hyp.total_service_classes ?? 20} (${hyp.total_dispatches ?? 0} dispatches)`],
    ]);

    $("#models-tbody").innerHTML = (data.supermix_aether?.models || [])
      .map(
        (m) => `<tr>
          <th scope="row">${escapeHtml(m.model_id)}</th>
          <td data-label="Backend">${escapeHtml(m.backend_type)}</td>
          <td data-label="Health">${m.healthy ? "Healthy" : "Disabled"}</td>
          <td data-label="Calls">${m.calls}</td>
          <td data-label="Avg Latency">${m.avg_latency_ms} ms</td>
        </tr>`
      )
      .join("");
  }

  // View H: Timeline & Morpheus
  $("#btn-advance-stage")?.addEventListener("click", async () => {
    const res = await api("/api/timeline/advance-stage", {
      method: "POST",
      body: JSON.stringify({ organism_id: state.selectedOrganismId }),
    });
    await refreshTimeline();
    await refreshALife();
    announce(`Advanced ${res.organism_id} to developmental stage ${res.current_stage}.`);
  });

  $("#btn-morpheus-run").addEventListener("click", async () => {
    const res = await api("/api/morpheus/damage-recover", {
      method: "POST",
      body: JSON.stringify({
        organism_id: state.selectedOrganismId,
        neuron_lesion_ratio: 0.3,
        synapse_sever_ratio: 0.3,
      }),
    });
    drawRecoveryCurve(res.recovery_curve || []);
    $("#morpheus-summary").textContent =
      `Before=${res.performance_before} · Damaged=${res.performance_damaged} · After=${res.performance_after} · Recovery(t)=${res.final_recovery_ratio}`;
    await refreshTimeline();
    announce(`Morpheus damage & recovery complete (Recovery ratio ${res.final_recovery_ratio}).`);
  });

  $("#btn-molt-assay")?.addEventListener("click", async () => {
    const qs = state.selectedOrganismId ? `?organism_id=${encodeURIComponent(state.selectedOrganismId)}` : "";
    const res = await api(`/api/morpheus/molt-assay${qs}`, { method: "POST" });
    const box = $("#molt-assay-box");
    if (box) {
      const p = res.placements || {};
      const rows = Object.entries(p)
        .map(([k, v]) => `${k}: pre=${v.pre_lesion_performance} → lesion=${v.post_lesion_performance} → regrowth+shift=${v.post_regrowth_world_shift_performance} (ret=${v.transfer_retention_ratio})`)
        .join("<br>");
      box.innerHTML =
        `<strong>${escapeHtml(res.protocol)}</strong> · Best Placement: <strong>${escapeHtml(res.best_trace_placement)}</strong><br>` +
        rows;
    }
    announce(`MOLT trace-placement assay complete (best placement: ${res.best_trace_placement}).`);
  });

  function drawRecoveryCurve(curve) {
    const canvas = $("#recovery-canvas");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!curve.length) return;
    const pad = 34;
    const maxR = Math.max(1.1, ...curve.map((c) => c.recovery_ratio));
    ctx.strokeStyle = "#00e5ff";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    curve.forEach((pt, idx) => {
      const x = pad + (idx / Math.max(1, curve.length - 1)) * (canvas.width - 2 * pad);
      const y = canvas.height - pad - (pt.recovery_ratio / maxR) * (canvas.height - 2 * pad);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    curve.forEach((pt, idx) => {
      const x = pad + (idx / Math.max(1, curve.length - 1)) * (canvas.width - 2 * pad);
      const y = canvas.height - pad - (pt.recovery_ratio / maxR) * (canvas.height - 2 * pad);
      ctx.fillStyle = idx === 0 ? "#ffb020" : "#26e3a8";
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#eef4ff";
      ctx.font = "11px monospace";
      ctx.fillText(`${pt.recovery_ratio.toFixed(2)}`, x - 12, y - 10);
    });
  }

  async function refreshTimeline() {
    const qs = state.selectedOrganismId ? `?organism_id=${encodeURIComponent(state.selectedOrganismId)}` : "";
    const data = await api(`/api/timeline${qs}`);
    $("#stage-readout").textContent = `Organism ${data.organism_id} · Current Stage: ${data.current_stage}`;

    const strip = $("#stage-progress-strip");
    if (strip && Array.isArray(data.all_stages)) {
      const curIdx = data.all_stages.indexOf(data.current_stage);
      strip.innerHTML = data.all_stages
        .map((st, idx) => {
          const cls = idx === curIdx ? "stage-pill is-active" : idx < curIdx ? "stage-pill is-done" : "stage-pill";
          const shortLbl = st.replace("STAGE_", "S").replace("_", " ");
          return `<div role="listitem" class="${cls}">${escapeHtml(shortLbl)}</div>`;
        })
        .join("");
    }

    if (data.comparison_first_vs_latest) {
      const c = data.comparison_first_vs_latest;
      $("#checkpoint-diff-box").textContent =
        `Diff (${c.checkpoint_a} → ${c.checkpoint_b}): ΔNeurons=${c.delta_neurons}, ΔSynapses=${c.delta_synapses}, ΔMemories=${c.delta_memories}, ΔFitness=${c.delta_fitness}`;
    }
    $("#checkpoints-list").innerHTML = (data.checkpoints || [])
      .slice()
      .reverse()
      .map(
        (cp) =>
          `<li>[age=${cp.age}] <strong>${escapeHtml(cp.label)}</strong> · ${escapeHtml(cp.stage)} · F=${cp.fitness} · N=${cp.num_neurons}, E=${cp.num_synapses}, M=${cp.memory_count}</li>`
      )
      .join("");
  }

  // View I: Frontier Research Pack Lab
  function renderKuramotoPhaseRing(data) {
    const canvas = $("#frontier-phase-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    const r = Math.min(w, h) * 0.38;

    ctx.strokeStyle = "rgba(110, 168, 254, 0.25)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    const phases = data.phases || [];
    phases.forEach((theta, idx) => {
      const px = cx + Math.cos(theta) * r;
      const py = cy + Math.sin(theta) * r;
      ctx.fillStyle = idx % 2 === 0 ? "#00e5ff" : "#26e3a8";
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, Math.PI * 2);
      ctx.fill();
    });

    const orderR = Number(data.order_parameter_R || 0);
    const meanPsi = Number(data.mean_phase_rad || 0);
    ctx.strokeStyle = "#ffb020";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(meanPsi) * r * orderR, cy + Math.sin(meanPsi) * r * orderR);
    ctx.stroke();

    const meta = $("#frontier-kuramoto-meta");
    if (meta) {
      meta.textContent = `R(t) = ${orderR.toFixed(4)} · PLI = ${Number(data.phase_locking_index || 0).toFixed(4)} · N = ${phases.length}`;
    }
  }

  function renderFrontierResultsTable(resultsList) {
    const tbody = $("#frontier-results-tbody");
    if (!tbody) return;
    tbody.innerHTML = (resultsList || [])
      .map((r) => {
        const st = r.vs_baseline || {};
        const badgeCls = r.verdict === "improved" ? "ltp-badge" : r.verdict === "regressed" ? "ltd-badge" : "struct-badge";
        return `<tr>
          <th scope="row">${escapeHtml(r.module_id)}</th>
          <td>${escapeHtml(r.metric_name)}</td>
          <td>${Number(r.baseline_mean).toFixed(4)}</td>
          <td><strong>${Number(r.candidate_mean).toFixed(4)}</strong></td>
          <td>${Number(r.capacity_matched_control_mean).toFixed(4)}</td>
          <td>[${Number(st.ci_95_low || 0).toFixed(4)}, ${Number(st.ci_95_high || 0).toFixed(4)}]</td>
          <td>${Number(st.e_value || 1).toFixed(2)}</td>
          <td><span class="${badgeCls}">${escapeHtml(String(r.verdict || "").toUpperCase())}</span></td>
        </tr>`;
      })
      .join("");
  }

  async function refreshFrontier() {
    const reg = await api("/api/frontier/registry");
    const sel = $("#frontier-module-select");
    if (sel && !sel.options.length) {
      sel.innerHTML = (reg.modules || [])
        .map((m) => `<option value="${escapeHtml(m.module_id)}">[${escapeHtml(m.category)}] ${escapeHtml(m.title)}</option>`)
        .join("");
    }
    const spec = reg.hook_status?.spec || {};
    if ($("#chk-hook-enabled")) $("#chk-hook-enabled").checked = Boolean(spec.enabled);
    if ($("#chk-hook-kuramoto")) $("#chk-hook-kuramoto").checked = Boolean(spec.enable_kuramoto);
    if ($("#chk-hook-fastweights")) $("#chk-hook-fastweights").checked = Boolean(spec.enable_fast_weights);
    if ($("#chk-hook-assoc")) $("#chk-hook-assoc").checked = Boolean(spec.enable_associative_memory);
    if ($("#chk-hook-ctm")) $("#chk-hook-ctm").checked = Boolean(spec.enable_ctm_synchrony);
    if ($("#chk-hook-continuum")) $("#chk-hook-continuum").checked = Boolean(spec.enable_continuum_memory);
    if ($("#chk-hook-btsp")) $("#chk-hook-btsp").checked = Boolean(spec.enable_btsp);
    if ($("#chk-hook-dendritic")) $("#chk-hook-dendritic").checked = Boolean(spec.enable_dendritic);
    if ($("#frontier-hook-telemetry")) {
      $("#frontier-hook-telemetry").textContent = formatFrontierHookTelemetry(reg.hook_status);
    }
  }

  $("#btn-frontier-run")?.addEventListener("click", async () => {
    const moduleId = $("#frontier-module-select")?.value || "ctm_synchrony";
    announce(`Running capacity-matched frontier experiment: ${moduleId}...`);
    const res = await api("/api/frontier/run", {
      method: "POST",
      body: JSON.stringify({ module_id: moduleId, seeds: [42, 101, 202, 303, 404] }),
    });
    renderFrontierResultsTable([res]);
    $("#frontier-receipt-pre").textContent = JSON.stringify(res, null, 2);
    announce(`Frontier experiment ${moduleId} finished: verdict=${res.verdict}, receipt=${res.receipt_sha256.slice(0, 12)}.`);
  });

  $("#btn-frontier-suite")?.addEventListener("click", async () => {
    announce("Running all-19 capacity-matched frontier sweep...");
    const res = await api("/api/frontier/suite", {
      method: "POST",
      body: JSON.stringify({ seeds: [42, 101, 202, 303, 404] }),
    });
    renderFrontierResultsTable(res.results || []);
    $("#frontier-receipt-pre").textContent = JSON.stringify(
      {
        modules_executed: res.modules_executed,
        improved_count: res.improved_count,
        receipts: (res.results || []).map((r) => ({ module: r.module_id, verdict: r.verdict, sha256: r.receipt_sha256.slice(0, 16) })),
      },
      null,
      2
    );
    announce(`Frontier sweep complete: ${res.improved_count} / ${res.modules_executed} modules strictly improved.`);
  });

  $("#btn-frontier-kuramoto")?.addEventListener("click", async () => {
    const res = await api("/api/frontier/kuramoto-step", {
      method: "POST",
      body: JSON.stringify({ organism_id: state.selectedOrganismId, steps: 24, coupling_k: 2.8 }),
    });
    renderKuramotoPhaseRing(res);
    $("#frontier-receipt-pre").textContent = JSON.stringify(res, null, 2);
    announce(`Kuramoto synchronization updated: R(t)=${res.order_parameter_R}.`);
  });

  $("#btn-frontier-nir")?.addEventListener("click", async () => {
    const qs = state.selectedOrganismId ? `?organism_id=${encodeURIComponent(state.selectedOrganismId)}` : "";
    const res = await api(`/api/frontier/nir-export${qs}`, { method: "POST" });
    $("#frontier-receipt-pre").textContent = JSON.stringify(res, null, 2);
    announce(`Exported validated NIR v${res.validation?.nir_version} graph (${res.validation?.node_count} nodes, ${res.validation?.edge_count} edges).`);
  });

  $("#btn-frontier-hook-apply")?.addEventListener("click", async () => {
    const res = await api("/api/frontier/hook-config", {
      method: "POST",
      body: JSON.stringify({
        enabled: Boolean($("#chk-hook-enabled")?.checked),
        enable_kuramoto: Boolean($("#chk-hook-kuramoto")?.checked),
        enable_fast_weights: Boolean($("#chk-hook-fastweights")?.checked),
        enable_associative_memory: Boolean($("#chk-hook-assoc")?.checked),
        enable_ctm_synchrony: Boolean($("#chk-hook-ctm")?.checked),
        enable_continuum_memory: Boolean($("#chk-hook-continuum")?.checked),
        enable_btsp: Boolean($("#chk-hook-btsp")?.checked),
        enable_dendritic: Boolean($("#chk-hook-dendritic")?.checked),
      }),
    });
    if ($("#frontier-hook-telemetry")) {
      $("#frontier-hook-telemetry").textContent = formatFrontierHookTelemetry(res.hook_status);
    }
    $("#frontier-receipt-pre").textContent = JSON.stringify(res, null, 2);
    announce(`Updated neutral frontier engine hook (active=${res.hook_status?.active}).`);
  });

  document.querySelectorAll(".cv-auto-panel").forEach((panel) => {
    panel.addEventListener("contentvisibilityautostatechange", (ev) => {
      panel.dataset.cvSkipped = String(Boolean(ev.skipped));
    });
  });

  async function refreshCurrentView() {
    if (state.currentView === "command") await refreshCommandCentre();
    else if (state.currentView === "connectome") await refreshConnectome();
    else if (state.currentView === "memory") await refreshMemory();
    else if (state.currentView === "alife") await refreshALife();
    else if (state.currentView === "world") await refreshWorld();
    else if (state.currentView === "research") await refreshResearch();
    else if (state.currentView === "hardware") await refreshHardware();
    else if (state.currentView === "timeline") await refreshTimeline();
    else if (state.currentView === "frontier") await refreshFrontier();
  }

  async function refreshAll() {
    await refreshALife();
    await refreshCommandCentre();
    if (state.currentView !== "command" && state.currentView !== "alife") {
      await refreshCurrentView();
    }
  }

  refreshAll().catch((err) => announce(`Error loading observatory data: ${err.message}`));
})();
