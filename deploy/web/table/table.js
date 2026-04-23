// ARTEMIS table — LiveKit client + HUD state.
// No framework; plain DOM. Token + room name come from URL params.

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
  };

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
        names.includes("artemis") ? "ARTEMIS SPEAKING" :
        "HEARING " + names[0];
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
        if (parsed.kind === "artemis.say") {
          log("ARTEMIS: " + parsed.text);
        }
      } catch (_) {
        /* ignore non-JSON data */
      }
    });
    room.on(LivekitClient.RoomEvent.Disconnected, (reason) => {
      log("disconnected: " + (reason || ""));
      $("info-status").textContent = "DISCONNECTED";
    });

    try {
      await room.connect(serverUrl, token);
      state.localIdentity = room.localParticipant.identity;
      $("info-identity").textContent = state.localIdentity;
      $("info-status").textContent = "CONNECTED";
      log("connected as " + state.localIdentity);
      renderParticipants();
      // Enable mic by default (can toggle with button).
      try {
        await room.localParticipant.setMicrophoneEnabled(true);
        state.micOn = true;
        $("mic-btn").textContent = "MIC: ON";
        $("mic-btn").classList.add("active");
      } catch (_) {
        log("mic not available (permission denied?)");
      }
    } catch (err) {
      log("connect failed: " + err.message, "err");
      $("info-status").textContent = "ERROR";
      $("avatar-status").textContent = "FAIL";
    }
  }

  $("mic-btn").addEventListener("click", async () => {
    if (!state.room) return;
    state.micOn = !state.micOn;
    await state.room.localParticipant.setMicrophoneEnabled(state.micOn);
    $("mic-btn").textContent = state.micOn ? "MIC: ON" : "MIC: OFF";
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
    loadArtifacts();
    connect();
  });
})();
