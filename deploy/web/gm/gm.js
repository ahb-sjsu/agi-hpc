// ARTEMIS GM Portal — Keeper cockpit · S1h
//
// Pulls everything over HTTP from /api/* on atlas-sjsu.duckdns.org.
// Auth: JWT passed via ?t= on first load; stashed in sessionStorage.

(function () {
  "use strict";

  const qs = new URLSearchParams(window.location.search);
  const urlToken = qs.get("t") || qs.get("token") || "";
  if (urlToken) sessionStorage.setItem("gm_token", urlToken);
  const TOKEN = sessionStorage.getItem("gm_token") || "";

  const $ = (id) => document.getElementById(id);

  // Session id derived from the URL (or default). The keeper can
  // override in the scene panel footer.
  const initialSession =
    qs.get("session") ||
    window.location.pathname.split("/").filter(Boolean).pop() ||
    "default";
  $("session-id").value = initialSession;

  const state = {
    session: initialSession,
    scene: { name: "—", flags: "—" },
  };

  // ── fetch wrapper ─────────────────────────────────────────────
  async function api(method, path, body) {
    const opts = {
      method,
      headers: { Authorization: "Bearer " + TOKEN },
    };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(path, opts);
    if (res.status === 401) {
      document.body.innerHTML =
        '<div style="padding:40px;color:#e83838;font-family:monospace">' +
        "KEEPER AUTH REQUIRED<br>pass ?t=&lt;keeper-jwt&gt; on the URL</div>";
      throw new Error("unauthorized");
    }
    if (!res.ok) throw new Error("HTTP " + res.status);
    return res.json();
  }

  // ── CREW panel ────────────────────────────────────────────────
  async function loadCrew() {
    try {
      const data = await api("GET", "/api/sheet?name=characters");
      renderCrew(data.rows || []);
    } catch (err) {
      if (err.message !== "unauthorized") console.warn("crew fetch:", err);
    }
  }

  function renderCrew(rows) {
    const body = $("crew-body");
    body.innerHTML = "";
    $("crew-count").textContent = rows.length + " · " + initialSession;
    for (const r of rows) {
      const card = document.createElement("div");
      card.className = "crew-row";
      card.dataset.status = r.status || "ok";

      const hd = document.createElement("div");
      hd.className = "crew-hd";
      const nm = document.createElement("div");
      nm.className = "crew-name";
      nm.textContent = r.name || r.id;
      const rl = document.createElement("div");
      rl.className = "crew-role";
      rl.textContent = r.role || r.occupation || "";
      hd.appendChild(nm);
      hd.appendChild(rl);
      card.appendChild(hd);

      const stats = document.createElement("div");
      stats.className = "crew-stats";
      for (const [k, label] of [
        ["san", "SAN"],
        ["hp", "HP"],
        ["luck", "LCK"],
        ["mp", "MP"],
        ["str", "STR"],
        ["con", "CON"],
        ["siz", "SIZ"],
        ["dex", "DEX"],
        ["app", "APP"],
        ["int", "INT"],
        ["pow", "POW"],
        ["edu", "EDU"],
      ]) {
        if (r[k] === undefined) continue;
        const cell = document.createElement("span");
        const maxKey = k + "_max";
        const maxPart = r[maxKey] !== undefined ? " / " + r[maxKey] : "";
        cell.innerHTML = label + "<b>" + r[k] + maxPart + "</b>";
        stats.appendChild(cell);
      }
      card.appendChild(stats);

      if (r.skills && r.skills.length) {
        const s = document.createElement("div");
        s.className = "crew-skills";
        s.innerHTML =
          "<b>skills</b>" +
          r.skills.map((sk) => `${sk.n || sk.name}:${sk.v ?? sk.value ?? ""}`).join(" · ");
        card.appendChild(s);
      }
      if (r.equipment && r.equipment.length) {
        const e = document.createElement("div");
        e.className = "crew-equip";
        e.innerHTML =
          "<b>equip</b>" +
          r.equipment.map((it) => it.n || it.name || "").filter(Boolean).join(" · ");
        card.appendChild(e);
      }
      if (r.notes) {
        const n = document.createElement("div");
        n.className = "crew-notes";
        n.innerHTML = "<b>notes</b>" + r.notes;
        card.appendChild(n);
      }
      body.appendChild(card);
    }
    if (!rows.length) {
      body.innerHTML =
        '<div class="wiki-offline">no crew loaded — is atlas-artemis-sheets running?</div>';
    }
  }

  // ── CHAT panel ────────────────────────────────────────────────
  async function loadChat() {
    try {
      const data = await api(
        "GET",
        `/api/chat/recent?session=${encodeURIComponent(state.session)}&limit=200`,
      );
      renderChat(data.messages || []);
    } catch (err) {
      if (err.message !== "unauthorized") console.warn("chat fetch:", err);
    }
  }

  function renderChat(msgs) {
    const list = $("chat-list");
    list.innerHTML = "";
    $("chat-count").textContent = msgs.length + " msgs";
    for (const m of msgs) list.appendChild(chatLi(m));
    list.scrollTop = list.scrollHeight;
  }

  function chatLi(m) {
    const li = document.createElement("li");
    li.className = "kind-" + m.kind;
    const meta = document.createElement("span");
    meta.className = "chat-meta";
    const t = new Date(m.ts * 1000).toLocaleTimeString("en-GB", { hour12: false });
    const fromShort = (m.from_id || "").split(":").slice(-1)[0];
    meta.textContent = `${t} ${fromShort}`;
    const body = document.createElement("span");
    body.textContent = m.body;
    li.appendChild(meta);
    li.appendChild(body);
    return li;
  }

  async function searchChat(q) {
    try {
      const data = await api(
        "GET",
        `/api/chat/search?q=${encodeURIComponent(q)}&session=${encodeURIComponent(state.session)}`,
      );
      renderChat(data.hits || []);
    } catch (err) {
      if (err.message !== "unauthorized") console.warn("search:", err);
    }
  }

  $("search-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = $("search-q").value.trim();
    if (!q) return loadChat();
    searchChat(q);
  });

  $("compose-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const mode = $("compose-mode").value;
    const to = $("compose-to").value.trim();
    const body = $("compose-body").value.trim();
    if (!body) return;
    try {
      if (mode === "whisper") {
        if (!to) {
          alert("whisper needs a target");
          return;
        }
        await api("POST", "/api/chat/whisper", {
          session: state.session, to_id: to, body,
        });
      } else {
        await api("POST", "/api/chat/broadcast", {
          session: state.session, body,
        });
      }
      $("compose-body").value = "";
      loadChat();
    } catch (err) {
      if (err.message !== "unauthorized") alert("send failed: " + err.message);
    }
  });

  // ── WIKI panel ────────────────────────────────────────────────
  $("wiki-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const q = $("wiki-q").value.trim();
    if (!q) return;
    try {
      const data = await api(
        "GET",
        `/api/wiki/search?q=${encodeURIComponent(q)}&limit=20`,
      );
      renderWiki(data.hits || []);
    } catch (err) {
      if (err.message !== "unauthorized") console.warn("wiki:", err);
    }
  });

  function renderWiki(hits) {
    const list = $("wiki-list");
    list.innerHTML = "";
    if (!hits.length) {
      list.innerHTML =
        '<li class="wiki-offline">no hits (or wiki service offline)</li>';
      return;
    }
    for (const h of hits) {
      const li = document.createElement("li");
      const title = document.createElement("span");
      title.className = "wiki-title";
      title.textContent = h.title || h.id || "(untitled)";
      const snip = document.createElement("span");
      snip.className = "wiki-snip";
      snip.textContent = (h.snippet || h.body || "").slice(0, 400);
      const ops = document.createElement("div");
      ops.className = "wiki-ops";
      const q2 = document.createElement("button");
      q2.textContent = "→ CHAT";
      q2.addEventListener("click", () => {
        $("compose-body").value = `${h.title ? h.title + ": " : ""}${snip.textContent}`;
        $("compose-body").focus();
      });
      ops.appendChild(q2);
      li.appendChild(title);
      li.appendChild(snip);
      li.appendChild(ops);
      list.appendChild(li);
    }
  }

  // ── SCENE panel ───────────────────────────────────────────────
  async function loadScene() {
    try {
      const data = await api("GET", "/api/scene");
      state.scene = data;
      $("scene-name").value = data.name || "";
      $("scene-flags").value = data.flags || "";
      $("gm-scene").textContent =
        (data.name || "—").toUpperCase() + " · " + (data.flags || "").toUpperCase();
    } catch (err) {
      if (err.message !== "unauthorized") console.warn("scene:", err);
    }
  }

  $("scene-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    try {
      await api("POST", "/api/scene", {
        name: $("scene-name").value.trim(),
        flags: $("scene-flags").value.trim(),
      });
      loadScene();
    } catch (err) {
      if (err.message !== "unauthorized") alert("scene update: " + err.message);
    }
  });

  $("session-id").addEventListener("change", (ev) => {
    state.session = ev.target.value.trim() || "default";
    loadChat();
  });

  // ── boot ──────────────────────────────────────────────────────
  function decodeIdentity() {
    if (!TOKEN) return "—";
    try {
      const parts = TOKEN.split(".");
      const payload = JSON.parse(atob(parts[1].replace(/-/g, "+").replace(/_/g, "/")));
      return payload.identity || payload.sub || "?";
    } catch (_) {
      return "?";
    }
  }
  $("gm-who").textContent = decodeIdentity();

  function bootstrap() {
    loadCrew();
    loadChat();
    loadScene();
    // Refresh crew + chat every 10 s so the portal stays warm without
    // hammering the backend.
    setInterval(loadCrew, 10_000);
    setInterval(loadChat, 10_000);
  }

  document.addEventListener("DOMContentLoaded", bootstrap);
})();
