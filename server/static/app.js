/* Agent console — vanilla, no build step.
   The browser holds no policy: it sends what you typed, renders what the
   agent said, and relays your Approve/Deny to the server, which is where
   the gate lives. A pending action lives on the server's session, so
   closing this tab loses nothing. */

const el = {
  record: document.getElementById("record"),
  opening: document.getElementById("opening"),
  picker: document.getElementById("agent-picker"),
  standing: document.getElementById("standing"),
  tokens: document.getElementById("meter-tokens"),
  calls: document.getElementById("meter-calls"),
  cost: document.getElementById("meter-cost"),
  activity: document.getElementById("activity"),
  activityText: document.getElementById("activity-text"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  file: document.getElementById("file-input"),
  veil: document.getElementById("dropveil"),
};

let sessionId = null;
let socket = null;
let busy = false;

/* ── helpers ─────────────────────────────────────────────────────────── */

const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/** Deliberately small: bold, inline code, bullets, paragraphs. */
function renderProse(text) {
  const blocks = escapeHtml(text || "").trim().split(/\n{2,}/);
  return blocks.map((block) => {
    const lines = block.split("\n");
    const inline = (s) =>
      s.replace(/`([^`]+)`/g, "<code>$1</code>")
       .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    if (lines.every((l) => /^\s*[-*]\s+/.test(l))) {
      return "<ul>" + lines.map((l) =>
        "<li>" + inline(l.replace(/^\s*[-*]\s+/, "")) + "</li>").join("") + "</ul>";
    }
    return "<p>" + inline(lines.join("<br>")) + "</p>";
  }).join("");
}

function clearOpening() {
  if (el.opening) { el.opening.remove(); el.opening = null; }
}

function place(node) {
  clearOpening();
  el.record.appendChild(node);
  el.record.scrollTop = el.record.scrollHeight;
  return node;
}

function addTurn(who, text, { prose = true } = {}) {
  const wrap = document.createElement("article");
  wrap.className = `turn turn--${who === "you" ? "you" : "agent"}`;
  wrap.innerHTML =
    `<p class="turn__label">${escapeHtml(who)}</p>` +
    `<div class="turn__body">${prose ? renderProse(text) : escapeHtml(text)}</div>`;
  return place(wrap);
}

function addNote(text, bad = false) {
  const note = document.createElement("p");
  note.className = "note" + (bad ? " note--bad" : "");
  note.textContent = text;
  return place(note);
}

function setActivity(text) {
  if (!text) { el.activity.hidden = true; return; }
  el.activityText.textContent = text;
  el.activity.hidden = false;
}

const ACTIVITY_WORDS = {
  web_search: "searching the web",
  web_fetch: "reading a web page",
  read_file: "reading a file",
  read_pdf: "reading a PDF",
  read_docx: "reading a document",
  write_file: "writing a file",
  delete_file: "deleting a file",
};

function describeActivity(event) {
  if (event.kind === "thinking") return "thinking…";
  if (event.kind === "tool_start") return (ACTIVITY_WORDS[event.tool] || event.tool) + "…";
  if (event.kind === "awaiting") return "waiting for your decision";
  if (event.kind === "tool_end") return null;
  return null;
}

function setBudget(budget) {
  if (!budget) return;
  el.tokens.textContent = (budget.tokens ?? 0).toLocaleString();
  el.calls.textContent = budget.tool_calls ?? 0;
  el.cost.textContent = "$" + Number(budget.cost_usd ?? 0).toFixed(4);
}

function setBusy(state) {
  busy = state;
  el.send.disabled = state;
  el.input.disabled = state;
}

/* ── the countersignature card ───────────────────────────────────────── */

/* The gate's reason is written for the audit trail. Lead with what the
   action actually is, in the person's terms, and keep the gate's exact
   wording underneath as the record. */
const ASK = {
  write_file: "Write a file into the agent's folder?",
  delete_file: "Delete this file for good?",
  web_fetch: "Open a web page the agent chose itself?",
  web_search: "Search the web?",
  read_file: "Read this file?",
  read_pdf: "Read this PDF?",
  read_docx: "Read this document?",
};

function addWarrant(pending) {
  const card = document.createElement("section");
  card.className = "warrant";

  const facts = Object.entries(pending.params || {})
    .map(([key, value]) =>
      `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(
        typeof value === "string" ? value : JSON.stringify(value))}</dd>`)
    .join("");

  card.innerHTML =
    `<header class="warrant__head">
       <span>Your signature required</span>
       <span class="warrant__tool">${escapeHtml(pending.tool)}</span>
     </header>
     <p class="warrant__ask">${escapeHtml(
        ASK[pending.tool] || `Run ${pending.tool}?`)}</p>
     <p class="warrant__reason">${escapeHtml(pending.reason)}</p>
     <dl class="warrant__facts">${facts}</dl>
     <div class="warrant__choice">
       <button type="button" class="button button--approve">Approve</button>
       <button type="button" class="button button--deny">Deny</button>
     </div>`;

  const settle = (approved) => {
    card.querySelectorAll("button").forEach((b) => (b.disabled = true));
    card.classList.add("warrant--settled");
    const stamp = document.createElement("span");
    stamp.className = "stamp " + (approved ? "stamp--approved" : "stamp--denied");
    stamp.textContent = approved ? "Approved" : "Denied";
    card.appendChild(stamp);
    send({ type: "decision", approved });
  };

  card.querySelector(".button--approve").addEventListener("click", () => settle(true));
  card.querySelector(".button--deny").addEventListener("click", () => settle(false));
  return place(card);
}

/* ── transport ───────────────────────────────────────────────────────── */

function send(message) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    addNote("Not connected to the agent. Reload the page to reconnect.", true);
    return;
  }
  setBusy(true);
  socket.send(JSON.stringify(message));
}

function handle(message) {
  if (message.type === "activity") {
    const words = describeActivity(message);
    if (words) setActivity(words);
    return;
  }
  if (message.type === "error") {
    setActivity(null); setBusy(false);
    addNote(message.message, true);
    return;
  }
  if (message.type !== "update") return;

  setActivity(null);
  setBudget(message.budget);

  if (message.status === "completed") {
    if (message.text) addTurn(currentAgent(), message.text);
    setBusy(false);
  } else if (message.status === "awaiting_approval") {
    addWarrant(message.pending);
    setBusy(false);
  } else if (message.status === "budget_exceeded") {
    addNote(message.detail + ". Start a new session to continue.", true);
    setBusy(false);
  } else if (message.status === "error") {
    addNote(message.detail, true);
    setBusy(false);
  }
}

function currentAgent() {
  return el.picker.value || "agent";
}

function openStream() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/api/session/${sessionId}/stream`);
  socket.addEventListener("message", (event) => handle(JSON.parse(event.data)));
  socket.addEventListener("close", () => {
    setActivity(null);
    setBusy(false);
  });
}

/* ── session lifecycle ───────────────────────────────────────────────── */

async function startSession(agent) {
  setBusy(true);
  try {
    const response = await fetch("/api/session", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ agent }),
    });
    if (!response.ok) throw new Error((await response.json()).detail);
    const status = await response.json();
    sessionId = status.session_id;
    el.standing.textContent = `${status.agent} · ${status.autonomy}`;
    setBudget(status.budget);
    if (socket) socket.close();
    openStream();
  } catch (error) {
    addNote("Could not start that agent: " + error.message, true);
  } finally {
    setBusy(false);
  }
}

async function loadAgents() {
  try {
    const { agents } = await (await fetch("/api/agents")).json();
    if (!agents.length) {
      el.standing.textContent = "no runnable agents found";
      return;
    }
    el.picker.innerHTML = agents
      .map((a) => `<option value="${escapeHtml(a.name)}">${escapeHtml(a.name)}</option>`)
      .join("");
    await startSession(agents[0].name);
  } catch (error) {
    el.standing.textContent = "server unavailable";
  }
}

/* ── uploads ─────────────────────────────────────────────────────────── */

async function upload(file) {
  if (!sessionId) return;
  const body = new FormData();
  body.append("file", file);
  try {
    const response = await fetch(`/api/session/${sessionId}/upload`, { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail);
    addNote(`Added ${payload.name} to the agent's folder. Ask it to read the file.`);
  } catch (error) {
    addNote(error.message, true);
  }
}

/* ── events ──────────────────────────────────────────────────────────── */

el.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = el.input.value.trim();
  if (!text || busy) return;
  addTurn("you", text, { prose: false });
  el.input.value = "";
  el.input.style.height = "auto";
  setActivity("thinking…");
  send({ type: "message", text });
});

el.input.addEventListener("input", () => {
  el.input.style.height = "auto";
  el.input.style.height = Math.min(el.input.scrollHeight, 144) + "px";
});

el.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.composer.requestSubmit();
  }
});

el.picker.addEventListener("change", () => startSession(el.picker.value));
el.file.addEventListener("change", () => {
  if (el.file.files[0]) upload(el.file.files[0]);
  el.file.value = "";
});

/* The veil is shown only while files are actually over the window.
   dragenter/dragleave fire for every element the pointer crosses, so a
   depth counter tracks the window as a whole — and both handlers apply the
   same "is this a file drag?" test, or the counter drifts and the veil
   strands. Several endings (Esc, dropping outside, tabbing away) fire no
   leave at all, so those are caught explicitly. */
let dragDepth = 0;

function draggingFiles(event) {
  const data = event.dataTransfer;
  if (!data) return false;
  if (data.types && Array.from(data.types).includes("Files")) return true;
  return data.items && Array.from(data.items).some((item) => item.kind === "file");
}

function showVeil(visible) {
  if (!visible) dragDepth = 0;
  el.veil.hidden = !visible;
}

window.addEventListener("dragenter", (event) => {
  if (!draggingFiles(event)) return;
  dragDepth += 1;
  showVeil(true);
});

window.addEventListener("dragover", (event) => {
  if (!draggingFiles(event)) return;
  event.preventDefault();  // required, or the browser refuses the drop
  event.dataTransfer.dropEffect = "copy";
});

window.addEventListener("dragleave", (event) => {
  if (!draggingFiles(event)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) showVeil(false);
});

window.addEventListener("drop", (event) => {
  event.preventDefault();
  showVeil(false);
  if (!event.dataTransfer) return;
  for (const file of event.dataTransfer.files) upload(file);
});

// Endings that fire no dragleave.
window.addEventListener("dragend", () => showVeil(false));
window.addEventListener("blur", () => showVeil(false));
document.addEventListener("visibilitychange", () => {
  if (document.hidden) showVeil(false);
});

loadAgents();
