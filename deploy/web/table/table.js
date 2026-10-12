// ARTEMIS table — LiveKit client + HUD state.
// No framework; plain DOM. Token + room name come from URL params.
//
// S1a adds: scene-strip handling, PC stat cards (placeholder data —
// wired to Google Sheets in S1b), DataChannel dispatch for
// "artemis.sheet" / "artemis.scene" events.

(function () {
  "use strict";

  const qs = new URLSearchParams(window.location.search);
  const token = qs.get("t") || qs.get("token") || "";
  const roomName = qs.get("room") || window.location.pathname.split("/").pop() || "";
  const serverUrl = qs.get("server") || "wss://atlas-sjsu.duckdns.org/livekit";

  const $ = (id) => document.getElementById(id);

  const state = {
    room: null,
    localIdentity: null,
    micOn: false,
    startedAt: Date.now(),
    // Characters are keyed by `sheet:row_id`; S1b replaces
    // PLACEHOLDER_CHARS with live Sheets data.
    characters: {},
    scene: { name: "TRANSIT · DAY 184", flags: "NOMINAL" },
  };

  // Placeholder roster so the stats card has something to render
  // before S1b. Matches the pre-gens in Beyond_the_Heliopause
  // Appendix E (values are illustrative, not canonical).
  const PLACEHOLDER_CHARS = [
    { id: "imogen",  name: "IMOGEN ROTH",    role: "Expedition Lead",     san: 70, san_max: 75, hp: 11, hp_max: 11, luck: 55, luck_max: 70, mp: 14, mp_max: 14, status: "ok" },
    { id: "sully",   name: "ERIK SULLIVAN",  role: "Chief Engineer",      san: 62, san_max: 62, hp: 14, hp_max: 14, luck: 50, luck_max: 70, mp: 12, mp_max: 12, status: "ok" },
    { id: "asta",    name: "ASTA NORDQUIST", role: "Medical Officer",     san: 68, san_max: 72, hp: 11, hp_max: 11, luck: 60, luck_max: 70, mp: 14, mp_max: 14, status: "ok" },
    { id: "arlo",    name: "ARLO VANCE",     role: "Surface Ops / EVA",   san: 55, san_max: 55, hp: 15, hp_max: 15, luck: 45, luck_max: 70, mp: 11, mp_max: 11, status: "ok" },
    { id: "saoirse", name: "SAOIRSE KELLEHER", role: "Rad / Atmo Chem",   san: 65, san_max: 70, hp: 11, hp_max: 11, luck: 40, luck_max: 60, mp: 13, mp_max: 13, status: "shaken" },
  ];

  // ── tick clock ──────────────────────────────────────────────
  setInterval(() => {
    const d = new Date();
    $("clock").textContent = d.toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }, 500);

  function log(text, klass) {
    const li = document.createElement("li");
    const t = new Date().toLocaleTimeString("en-GB", { hour12: false });
    li.textContent = `${t}  ${text}`;
    if (klass) li.className = klass;
    const list = $("event-log");
    list.appendChild(li);
    while (list.children.length > 20) list.removeChild(list.firstChild);
    list.scrollTop = list.scrollHeight;
  }

  function renderParticipants() {
    const list = $("participants");
    list.innerHTML = "";
    if (!state.room) return;
    const all = [state.room.localParticipant, ...state.room.remoteParticipants.values()];
    for (const p of all) {
      const li = document.createElement("li");
      const isYou = p.identity === state.localIdentity;
      const isArtemis = p.identity === "artemis";
      const isActive = p.isSpeaking;
      li.textContent =
        (isArtemis ? "◆ " : isYou ? "▸ " : "· ") +
        p.identity +
        (isYou ? "  (you)" : "") +
        (isArtemis ? "  (ai)" : "");
      if (isYou) li.classList.add("you");
      if (isArtemis) li.classList.add("artemis");
      if (isActive) li.classList.add("active");
      list.appendChild(li);
    }
  }

  // ── PC stats card ───────────────────────────────────────────
  //
  // Renders one mini-card per character with SAN / HP / Luck / MP
  // bars. Color-coded by status. In S1b, state.characters is driven
  // by DataChannel events of kind "artemis.sheet" — Keeper's Google
  // Sheet is the source of truth.
  //
  // Identity-gated views (S1a extension):
  //   keeper:*     → full crew status, optional KEEPER badge
  //   player:<id>  → character card pinned at top with YOU badge,
  //                  others rendered dimmer below
  //   guest        → card hidden entirely (body[data-view="guest"])

  function computeView() {
    if (!state.localIdentity) return "guest";
    const idx = state.localIdentity.indexOf(":");
    if (idx < 0) return "guest";
    const prefix = state.localIdentity.slice(0, idx);
    const subject = state.localIdentity.slice(idx + 1);
    if (prefix === "keeper") return "keeper";
    if (prefix === "player" && subject && state.characters[subject]) {
      return "player";
    }
    return "guest";
  }

  function mySubjectId() {
    if (!state.localIdentity) return null;
    const idx = state.localIdentity.indexOf(":");
    return idx < 0 ? null : state.localIdentity.slice(idx + 1);
  }

  function renderStats() {
    const view = computeView();
    document.body.dataset.view = view;
    const grid = $("stats-grid");
    grid.innerHTML = "";

    // Guest view: stats card is hidden via CSS. Nothing else to do.
    if (view === "guest") return;

    const allChars = Object.values(state.characters);
    if (!allChars.length) {
      const empty = document.createElement("div");
      empty.className = "loading";
      empty.textContent = "no crew data";
      grid.appendChild(empty);
      return;
    }

    // Visibility rule (CoC-convention: sheet is private):
    //   keeper  → all crew
    //   player  → ONLY their own card
    const mine = mySubjectId();
    const visible =
      view === "keeper"
        ? allChars.slice().sort((a, b) =>
            (a.name || a.id).localeCompare(b.name || b.id),
          )
        : allChars.filter((c) => c.id === mine);

    if (!visible.length) {
      const empty = document.createElement("div");
      empty.className = "loading";
      empty.textContent = "no character assigned to this identity";
      grid.appendChild(empty);
      return;
    }

    for (const c of visible) {
      grid.appendChild(buildPcCard(c));
    }
  }

  // Absolute scale per stat — bar track width represents this cap.
  // Actual value = bright fill, max = dim extension to max, rest =
  // empty track. Shows at a glance how much headroom this PC has
  // vs the theoretical CoC 7e cap.
  const ABS_MAX = {
    san: 100,
    hp: 20,
    luck: 100,
    mp: 20,
  };

  function buildPcCard(c) {
    const card = document.createElement("div");
    card.className = "pc-card";
    card.dataset.status = c.status || "ok";

    const head = document.createElement("div");
    head.className = "pc-head";
    const name = document.createElement("div");
    name.className = "pc-name";
    name.textContent = c.name || c.id;
    const role = document.createElement("div");
    role.className = "pc-role";
    role.textContent = c.role || "";
    head.appendChild(name);
    head.appendChild(role);

    const bars = document.createElement("div");
    bars.className = "pc-bars";
    addBar(bars, "SAN",  "san",  c.san,  c.san_max,  ABS_MAX.san);
    addBar(bars, "HP",   "hp",   c.hp,   c.hp_max,   ABS_MAX.hp);
    addBar(bars, "LUCK", "luck", c.luck, c.luck_max, ABS_MAX.luck);
    addBar(bars, "MP",   "mp",   c.mp,   c.mp_max,   ABS_MAX.mp);

    card.appendChild(head);
    card.appendChild(bars);
    return card;
  }

  function addBar(parent, label, klass, value, max, absMax) {
    const v = Number(value) || 0;
    const m = Number(max) || 1;
    const cap = Math.max(absMax || 100, m);
    // Three regions on the fixed-scale track:
    //   0 .. pctActual  → bright fill    (current value)
    //   pctActual .. pctMax → dim segment (remaining capacity)
    //   pctMax .. 100   → empty track    (beyond this PC's max)
    const pctActual = Math.max(0, Math.min(100, (v / cap) * 100));
    const pctMax    = Math.max(0, Math.min(100, (m / cap) * 100));

    const lbl = document.createElement("span");
    lbl.className = "bar-label";
    lbl.textContent = label;

    const track = document.createElement("span");
    track.className = "bar-track";

    const reserve = document.createElement("span");
    reserve.className = "bar-reserve " + klass;
    reserve.style.left = pctActual + "%";
    reserve.style.width = Math.max(0, pctMax - pctActual) + "%";

    const fill = document.createElement("span");
    fill.className = "bar-fill " + klass;
    fill.style.width = pctActual + "%";

    // Order matters: reserve first (behind), fill on top.
    track.appendChild(reserve);
    track.appendChild(fill);

    const val = document.createElement("span");
    val.className = "bar-val";
    val.textContent = `${v} / ${m}`;

    parent.appendChild(lbl);
    parent.appendChild(track);
    parent.appendChild(val);
  }

  // Merge an update into state.characters. Accepts either a single
  // character dict or an array.
  function applySheetUpdate(payload) {
    const rows = Array.isArray(payload) ? payload : [payload];
    for (const r of rows) {
      if (!r || !r.id) continue;
      state.characters[r.id] = { ...state.characters[r.id], ...r };
    }
    renderStats();
  }

  // ── Scene strip ─────────────────────────────────────────────

  function applyScene(scene) {
    if (!scene) return;
    state.scene = { ...state.scene, ...scene };
    if (scene.name) $("scene-name").textContent = String(scene.name).toUpperCase();
    if (scene.flags !== undefined) {
      const flagsEl = $("scene-flags");
      flagsEl.textContent = String(scene.flags).toUpperCase();
    }
  }

  // ── Artifacts ───────────────────────────────────────────────

  async function loadArtifacts() {
    const list = $("artifact-list");
    list.innerHTML = "";
    try {
      const res = await fetch("/artifacts/manifest.json?session=" + encodeURIComponent(roomName));
      const items = await res.json();
      if (!items.length) {
        const li = document.createElement("li");
        li.className = "loading";
        li.textContent = "no artifacts available";
        list.appendChild(li);
        return;
      }
      for (const it of items) {
        const li = document.createElement("li");
        li.className = "artifact";
        const left = document.createElement("span");
        left.textContent = it.title;
        const a = document.createElement("a");
        a.href = it.href;
        a.target = "_blank";
        a.rel = "noopener";
        a.textContent = "download";
        li.appendChild(left);
        li.appendChild(a);
        list.appendChild(li);
      }
    } catch (err) {
      const li = document.createElement("li");
      li.className = "loading";
      li.textContent = "artifact service unavailable";
      list.appendChild(li);
    }
  }

  // ── LiveKit connect ─────────────────────────────────────────

  async function connect() {
    if (!token) {
      log("no token in URL — pass ?t=<jwt>", "err");
      $("info-status").textContent = "NO TOKEN";
      return;
    }
    if (!window.LivekitClient) {
      log("livekit-client not loaded", "err");
      return;
    }

    $("info-room").textContent = roomName || "(from token)";
    $("info-status").textContent = "CONNECTING";

    const room = new LivekitClient.Room({
      adaptiveStream: true,
      dynacast: true,
    });
    state.room = room;

    room.on(LivekitClient.RoomEvent.ParticipantConnected, (p) => {
      log("join: " + p.identity);
      renderParticipants();
    });
    room.on(LivekitClient.RoomEvent.ParticipantDisconnected, (p) => {
      log("leave: " + p.identity);
      renderParticipants();
    });
    room.on(LivekitClient.RoomEvent.ActiveSpeakersChanged, (speakers) => {
      renderParticipants();
      const names = speakers.map((s) => s.identity);
      const flagsEl = $("scene-flags");
      if (names.includes("artemis")) {
        flagsEl.textContent = "ARTEMIS · SPEAKING";
      } else if (names.length) {
        flagsEl.textContent = "HEARING · " + names[0].toUpperCase();
      } else {
        flagsEl.textContent = String(state.scene.flags || "NOMINAL").toUpperCase();
      }
    });
    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, pub, p) => {
      log("track: " + p.identity + "/" + track.kind);
      // ARTEMIS video tile is supplied by the meeting client; we only
      // attach her audio so this console can carry session sound when
      // it's opened without a separate meeting client.
      if (track.kind === "audio") {
        const el = track.attach();
        el.autoplay = true;
        document.body.appendChild(el);
      }
    });
    room.on(LivekitClient.RoomEvent.DataReceived, (payload, p) => {
      try {
        const parsed = JSON.parse(new TextDecoder().decode(payload));
        switch (parsed.kind) {
          case "artemis.say":
            log("ARTEMIS: " + parsed.text);
            break;
          case "artemis.sheet":
            applySheetUpdate(parsed.rows || parsed);
            log("stats updated from sheet");
            break;
          case "artemis.scene":
            applyScene(parsed);
            log("scene: " + (parsed.name || ""));
            break;
          case "artemis.chat":
            // Avatar bridge forwards each NATS chat-out / broadcast as
            // kind=artemis.chat so every browser sees it. Service-side
            // identity gating already filtered who the payload is for
            // (chat.out.<id> or broadcast), so we render unconditionally.
            applyChatMessage(parsed.msg || parsed);
            break;
        }
      } catch (_) {
        /* ignore non-JSON data */
      }
    });
    room.on(LivekitClient.RoomEvent.Disconnected, (reason) => {
      log("disconnected: " + (reason || ""));
      $("info-status").textContent = "DISCONNECTED";
      $("info-status").classList.remove("critical");
      $("info-status").classList.add("warn");
    });

    try {
      await room.connect(serverUrl, token);
      state.localIdentity = room.localParticipant.identity;
      $("info-identity").textContent = state.localIdentity;
      const statusEl = $("info-status");
      statusEl.textContent = "CONNECTED";
      statusEl.classList.remove("warn", "critical");
      log("connected as " + state.localIdentity);
      renderParticipants();
      // Re-render stats + chat-modes now that we know who we are —
      // view mode (player / keeper / guest) is identity-derived.
      renderStats();
      renderChatModes();
      $("chat-status").textContent = "LIVE";
      try {
        await room.localParticipant.setMicrophoneEnabled(true);
        state.micOn = true;
        $("mic-btn").textContent = "MIC : ON";
        $("mic-btn").classList.add("active");
      } catch (_) {
        log("mic not available (permission denied?)");
      }
    } catch (err) {
      log("connect failed: " + err.message, "err");
      const statusEl = $("info-status");
      statusEl.textContent = "ERROR";
      statusEl.classList.add("critical");
    }
  }

  // ── Chat card (S1e) ────────────────────────────────────────
  //
  // Transport contract:
  //   outbound (browser → avatar bridge → NATS)
  //     DataChannel payload = {kind: "artemis.chat.in", msg: ChatMessage}
  //     the avatar re-publishes on agi.rh.artemis.chat.in.<from_id>.
  //
  //   inbound (NATS → avatar bridge → DataChannel)
  //     DataChannel payload = {kind: "artemis.chat", msg: ChatMessage}
  //     every browser receives; the chat service has already done the
  //     identity-gating when it chose which subject to publish to.
  //
  // For the smoke-test / offline case (no avatar bridge running),
  // player messages show locally so the UI still feels alive.

  const CHAT_MODES_BY_VIEW = {
    player: [
      { kind: "player_to_artemis", label: "→ ARTEMIS" },
    ],
    keeper: [
      { kind: "keeper_to_all",    label: "→ ALL" },
      { kind: "keeper_to_player", label: "→ PLAYER…" },
    ],
    guest: [],
  };

  function renderChatModes() {
    const view = computeView();
    const sel = $("chat-mode");
    if (!sel) return;
    sel.innerHTML = "";
    const modes = CHAT_MODES_BY_VIEW[view] || [];
    for (const m of modes) {
      const opt = document.createElement("option");
      opt.value = m.kind;
      opt.textContent = m.label;
      sel.appendChild(opt);
    }
  }

  function chatWhoLabel(msg) {
    if (msg.kind === "artemis_to_player" || msg.kind === "artemis_to_all") {
      return "ARTEMIS";
    }
    if (msg.from_id === state.localIdentity) return "YOU";
    // Strip the "player:" / "keeper:" prefix for readability.
    const idx = (msg.from_id || "").indexOf(":");
    return (idx < 0 ? msg.from_id : msg.from_id.slice(idx + 1)).toUpperCase();
  }

  function applyChatMessage(msg) {
    if (!msg || !msg.kind || !msg.body) return;
    const logEl = $("chat-log");
    // Remove the placeholder hint on first real message.
    const hint = logEl.querySelector(".chat-hint");
    if (hint) hint.remove();

    const li = document.createElement("li");
    li.className = "kind-" + msg.kind;
    const who = document.createElement("span");
    who.className = "chat-who";
    who.textContent = chatWhoLabel(msg);
    const body = document.createElement("span");
    body.className = "chat-body";
    body.textContent = msg.body;
    li.appendChild(who);
    li.appendChild(body);
    logEl.appendChild(li);

    // Keep the scrollback bounded on the client; the server has the
    // authoritative transcript.
    while (logEl.children.length > 200) logEl.removeChild(logEl.firstChild);
    logEl.scrollTop = logEl.scrollHeight;
  }

  async function sendChatMessage(body) {
    const view = computeView();
    if (view === "guest") return;
    const modes = CHAT_MODES_BY_VIEW[view] || [];
    if (!modes.length) return;

    const selectedKind = $("chat-mode").value || modes[0].kind;
    let toId = null;
    if (selectedKind === "player_to_artemis") {
      toId = "artemis";
    } else if (selectedKind === "keeper_to_player") {
      // Prompt the Keeper for a target on the fly — keeps the UI tiny
      // for S1e. The full keeper portal (S1h) will have a proper
      // per-player input row.
      const target = window.prompt("whisper to which player? (e.g. imogen)", "");
      if (!target) return;
      toId = target.startsWith("player:") ? target : "player:" + target;
    }

    const msg = {
      kind: selectedKind,
      session_id: roomName || "default",
      from_id: state.localIdentity || "guest",
      to_id: toId,
      body: body,
      ts: Date.now() / 1000,
      corr_id: Math.random().toString(36).slice(2, 14),
    };

    // Optimistically render our own message so the UI doesn't feel
    // laggy while the round-trip happens.
    applyChatMessage(msg);

    if (state.room) {
      const payload = new TextEncoder().encode(
        JSON.stringify({ kind: "artemis.chat.in", msg }),
      );
      try {
        await state.room.localParticipant.publishData(payload, { reliable: true });
        $("chat-status").textContent = "LIVE";
      } catch (err) {
        $("chat-status").textContent = "SEND FAILED";
        log("chat send failed: " + err.message, "err");
      }
    } else {
      $("chat-status").textContent = "NOT CONNECTED";
    }
  }

  const chatForm = $("chat-form");
  if (chatForm) {
    chatForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const input = $("chat-input");
      const body = (input.value || "").trim();
      if (!body) return;
      input.value = "";
      await sendChatMessage(body);
    });
  }

  $("mic-btn").addEventListener("click", async () => {
    if (!state.room) return;
    state.micOn = !state.micOn;
    await state.room.localParticipant.setMicrophoneEnabled(state.micOn);
    $("mic-btn").textContent = state.micOn ? "MIC : ON" : "MIC : OFF";
    $("mic-btn").classList.toggle("active", state.micOn);
  });

  $("leave-btn").addEventListener("click", async () => {
    if (state.room) await state.room.disconnect();
    state.room = null;
    renderParticipants();
  });

  // ── kickoff ─────────────────────────────────────────────────
  document.addEventListener("DOMContentLoaded", () => {
    // Seed with placeholder roster; S1b will overwrite from Sheets.
    applySheetUpdate(PLACEHOLDER_CHARS);
    applyScene(state.scene);
    loadArtifacts();
    connect();
  });
})();
