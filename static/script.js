// Conversation lives in this array only: it is gone on reload, nothing is stored.
const turns = [];
const $ = (id) => document.getElementById(id);

// Webcam / screen captures: Blobs in memory only. Sent with the next question, then dropped.
const shots = [];

function drawShots() {
  const attachments = [...$("images").files].map((f) => {
    const item = document.createElement("span");
    item.className = "attachment";
    item.textContent = f.name;
    return item;
  });
  if ($("video").files[0]) {
    const item = document.createElement("span");
    item.className = "attachment";
    item.textContent = $("video").files[0].name;
    attachments.push(item);
  }
  attachments.push(...shots.map((s, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "attachment capture-attachment";
    b.setAttribute("aria-label", `Remove captured image ${i + 1}`);
    const im = document.createElement("img");
    im.src = s.url;
    im.alt = "";
    const name = document.createElement("span");
    name.textContent = `Capture ${i + 1}`;
    b.append(im, name);
    b.onclick = () => { URL.revokeObjectURL(s.url); shots.splice(shots.indexOf(s), 1); drawShots(); };
    return b;
  }));
  $("shots").replaceChildren(...attachments);
}

function dropShots() {
  shots.forEach((s) => URL.revokeObjectURL(s.url));
  shots.length = 0;
  drawShots();
}

async function capture(getStream) {
  let stream;
  $("cam").disabled = $("scr").disabled = true; // a second click would open a second picker
  try {
    stream = await getStream();
    const v = document.createElement("video");
    v.srcObject = stream;
    v.muted = true;
    await v.play();
    await new Promise((r) => setTimeout(r, 500)); // webcams need a moment to set exposure
    if (!v.videoWidth) throw new Error("no video frame received");
    const c = document.createElement("canvas");
    const k = Math.min(1, 1568 / Math.max(v.videoWidth, v.videoHeight)); // the server shrinks to 1568 anyway, and refuses >5 MB first
    c.width = Math.round(v.videoWidth * k);
    c.height = Math.round(v.videoHeight * k);
    c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
    const blob = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.9));
    if (!blob) throw new Error("could not encode the capture");
    shots.push({ blob, url: URL.createObjectURL(blob) });
    drawShots();
  } catch (err) {
    $("out").textContent = "Capture failed: " + err.message;
    $("out").dataset.error = "true";
  } finally {
    stream?.getTracks().forEach((t) => t.stop()); // turns the camera light / screen-share banner off
    $("cam").disabled = $("scr").disabled = false;
  }
}

$("cam").onclick = () => capture(() => navigator.mediaDevices.getUserMedia({ video: true }));
$("scr").onclick = () => capture(() => navigator.mediaDevices.getDisplayMedia({ video: true }));
$("add-images").onclick = () => $("images").click();
$("add-video").onclick = () => $("video").click();
$("images").onchange = $("video").onchange = drawShots;

function setExpanded(expanded) {
  $("composer-toolbar").hidden = !expanded;
  $("composer-disclaimer").hidden = !expanded;
  $("expand-tools-btn").setAttribute("aria-expanded", String(expanded));
  $("expand-tools-btn").setAttribute("aria-label", expanded ? "Hide attachments and model options" : "Show attachments and model options");
}

$("expand-tools-btn").onclick = () => setExpanded($("composer-toolbar").hidden);
$("text").onfocus = () => setExpanded(true);
$("text").oninput = () => {
  setExpanded(true);
  $("text").style.height = "auto";
  $("text").style.height = Math.min($("text").scrollHeight, 144) + "px";
};
$("text").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("f").requestSubmit();
  } else if (e.key === "Escape" && !$("text").value.trim() && !$("images").files.length && !$("video").files.length && !shots.length) {
    setExpanded(false);
    $("text").blur();
  }
};

const sidebar = $("workspace-sidebar");
const mobileMenu = $("mobile-menu-btn");
function setSidebarOpen(open) {
  sidebar.classList.toggle("is-open", open);
  $("sidebar-backdrop").classList.toggle("is-open", open);
  mobileMenu.setAttribute("aria-expanded", String(open));
  mobileMenu.setAttribute("aria-label", open ? "Close workspace menu" : "Open workspace menu");
}
mobileMenu.onclick = () => setSidebarOpen(!sidebar.classList.contains("is-open"));
$("sidebar-backdrop").onclick = () => setSidebarOpen(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") setSidebarOpen(false); });
document.addEventListener("click", (e) => {
  const hasAttachments = $("images").files.length || $("video").files.length || shots.length;
  if (!hasAttachments && !$("text").value.trim() && !$("f").contains(e.target)) setExpanded(false);
});

function appendUserMessage(text) {
  $("empty-state").hidden = true;
  const message = document.createElement("div");
  message.className = "message message-user";
  message.textContent = text;
  const names = [...$("images").files].map((f) => f.name);
  if ($("video").files[0]) names.push($("video").files[0].name);
  if (shots.length) names.push(`${shots.length} captured image${shots.length === 1 ? "" : "s"}`);
  if (names.length) {
    const attached = document.createElement("div");
    attached.className = "user-attachments";
    attached.textContent = names.join(" · ");
    message.append(attached);
  }
  $("messages").append(message);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function appendAssistantMessage(data) {
  const message = document.createElement("article");
  message.className = "message message-assistant";
  const heading = document.createElement("div");
  heading.className = "message-heading";
  heading.textContent = "ARGUS";
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = data.answer || "";
  message.append(heading, body);
  if (data.sources?.length) {
    const section = document.createElement("section");
    section.className = "message-sources";
    const title = document.createElement("h2");
    title.textContent = "Sources";
    section.append(title);
    for (const source of data.sources) {
      const link = document.createElement("a");
      try {
        const url = new URL(source.url);
        if (url.protocol !== "https:" && url.protocol !== "http:") continue;
        link.href = url.href;
      } catch { continue; }
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title || source.url;
      section.append(link);
    }
    if (section.children.length > 1) message.append(section);
  }
  if (data.note) {
    const note = document.createElement("div");
    note.className = "message-note";
    note.textContent = data.note;
    message.append(note);
  }
  const details = document.createElement("details");
  details.className = "message-details";
  const summary = document.createElement("summary");
  summary.textContent = "Request details";
  const meta = document.createElement("pre");
  meta.textContent = JSON.stringify({ steps: data.steps, degraded: data.degraded, note: data.note });
  details.append(summary, meta);
  message.append(details);
  $("empty-state").hidden = true;
  $("messages").append(message);
  $("messages").scrollTop = $("messages").scrollHeight;
}

function showError(text) {
  const message = document.createElement("div");
  message.className = "message message-error";
  message.setAttribute("role", "alert");
  message.textContent = text;
  $("messages").append(message);
  $("empty-state").hidden = true;
  $("messages").scrollTop = $("messages").scrollHeight;
}

$("f").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("text").value;
  const fd = new FormData();
  fd.append("text", text);
  fd.append("force", $("force").value);
  fd.append("history", JSON.stringify(turns));
  for (const f of $("images").files) fd.append("images", f);
  for (const s of shots) fd.append("images", s.blob, "capture.jpg"); // the server ignores parts without a filename
  if ($("video").files[0]) fd.append("video", $("video").files[0]);
  const attachmentCount = $("images").files.length + shots.length + Number(Boolean($("video").files[0]));
  if (text.trim() || attachmentCount) appendUserMessage(text.trim() || ($("video").files[0] ? "Analyze attached video" : "Analyze attached image"));
  $("btn").disabled = true;
  $("out").textContent = "Working…";
  $("out").removeAttribute("data-error");
  try {
    const r = await fetch("/api/ask", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    appendAssistantMessage(j);
    $("out").textContent = "Response received.";
    dropShots();
    $("text").value = "";
    $("text").style.height = "auto";
    $("images").value = "";
    $("video").value = "";
    drawShots();
    setExpanded(false);
    if (text.trim()) turns.push({ role: "user", content: text }, { role: "assistant", content: j.answer });
    turns.splice(0, turns.length - 20); // the server rejects a history field over ~500 KB, so it must not grow forever
  } catch (err) {
    $("out").textContent = "Error: " + err.message;
    $("out").dataset.error = "true";
    showError(err.message);
  } finally {
    $("btn").disabled = false;
  }
};

$("stats").onclick = async () => {
  try {
    $("out").textContent = JSON.stringify(await (await fetch("/api/stats")).json());
    $("out").removeAttribute("data-error");
  } catch (err) {
    $("out").textContent = "Stats unavailable: " + err.message;
    $("out").dataset.error = "true";
  }
};

$("reset-session-btn").onclick = () => {
  turns.length = 0;
  $("messages").replaceChildren();
  $("empty-state").hidden = false;
  $("text").value = "";
  $("text").style.height = "auto";
  $("images").value = "";
  $("video").value = "";
  $("force").value = "";
  $("out").textContent = "";
  $("out").removeAttribute("data-error");
  dropShots();
  setExpanded(false);
  setSidebarOpen(false);
};
