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
    addBar(bars, "SAN",  "san",  c.san,  c.san_max);
    addBar(bars, "HP",   "hp",   c.hp,   c.hp_max);
    addBar(bars, "LUCK", "luck", c.luck, c.luck_max);
    addBar(bars, "MP",   "mp",   c.mp,   c.mp_max);

    card.appendChild(head);
    card.appendChild(bars);
    return card;
  }

  function addBar(parent, label, klass, value, max) {
    const v = Number(value) || 0;
    const m = Number(max) || 1;
    const pct = Math.max(0, Math.min(100, (v / m) * 100));
    const lbl = document.createElement("span");
    lbl.className = "bar-label";
    lbl.textContent = label;
    const track = document.createElement("span");
    // Stat class on the track so the "remaining to max" portion can be
    // tinted with the stat's dim color (stacked actual|max visual).
    track.className = "bar-track " + klass;
    const fill = document.createElement("span");
    fill.className = "bar-fill " + klass;
    fill.style.width = pct + "%";
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
      $("avatar-status").textContent =
        names.length === 0 ? "IDLE" :
        names.includes("artemis") ? "ARTEMIS · SPEAKING" :
        "HEARING · " + names[0];
    });
    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, pub, p) => {
      log("track: " + p.identity + "/" + track.kind);
      if (p.identity === "artemis" && track.kind === "video") {
        const el = $("artemis-video");
        track.attach(el);
        el.dataset.connected = "true";
        $("avatar-placeholder").hidden = true;
      }
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
      // Re-render stats now that we know who we are — view mode
      // (player / keeper / guest) is identity-derived.
      renderStats();
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
      $("avatar-status").textContent = "FAIL";
    }
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
    $("artemis-video").removeAttribute("data-connected");
    $("avatar-placeholder").hidden = false;
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
