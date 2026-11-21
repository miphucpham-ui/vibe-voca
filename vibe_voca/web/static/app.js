"use strict";
const $ = (id) => document.getElementById(id);
const TONES = { ngang: "--t-ngang", "huyền": "--t-huyen", "ngã": "--t-nga", "hỏi": "--t-hoi", "sắc": "--t-sac", "nặng": "--t-nang", EN: "--t-en" };
const SLIDERS = ["speed", "semitones", "pitch_range", "comma_ms", "stop_ms", "gap_ms", "texture"];
let voice = null;
let lastUrl = null;

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function wavUrl(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
}
function showError(message) { const e = $("error"); e.textContent = message; e.hidden = !message; }

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);
  return data;
}

function renderVoices(voices) {
  const box = $("voices");
  box.innerHTML = "";
  for (const v of voices) {
    const b = document.createElement("button");
    b.className = "voice" + (v.id === voice ? " on" : "");
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", v.id === voice);
    b.innerHTML = `${v.label}<span class="id">${v.id}</span>`;
    b.onclick = () => { voice = v.id; renderVoices(voices); };
    box.appendChild(b);
  }
}

function settings() {
  const out = { text: $("text").value, voice, punctuate: $("punctuate").checked, polish: $("polish").checked,
                seed: parseInt($("seed").value || "0", 10) };
  for (const s of SLIDERS) out[s] = parseFloat($(s).value);
  return out;
}

function busy(on) {
  $("go").disabled = on; $("random").disabled = on;
  $("go").querySelector(".spinner").hidden = !on;
  $("go").querySelector(".btn-text").textContent = on ? "Đang đọc" : "Đọc";
}

async function speak() {
  showError("");
  if (!$("text").value.trim()) { showError("Nhập câu cần đọc."); return; }
  busy(true);
  try {
    const r = await api("api/synth", settings());
    if (lastUrl) URL.revokeObjectURL(lastUrl);
    lastUrl = wavUrl(r.audio);
    $("audio").src = lastUrl; $("download").href = lastUrl;
    $("result").hidden = false;
    $("audio").play().catch(() => {});
    $("stats").textContent = `${r.seconds.toFixed(2)} s âm thanh · tổng hợp ${r.elapsed.toFixed(2)} s ` +
      `(${Math.round(r.seconds / Math.max(r.elapsed, 1e-3))}× thời gian thực) · ${r.sentences} câu`;
    $("spoken").hidden = !r.spoken; $("spoken").textContent = r.spoken || "";
    $("unknown").hidden = !r.unknown.length;
    $("unknown").textContent = r.unknown.length ? `Từ không đọc được: ${r.unknown.join(", ")}` : "";
    renderPhonemes(r.phonemes);
    renderPlot(r.tracks, r.seconds);
  } catch (e) { showError(e.message); }
  busy(false);
}

async function randomSentence() {
  showError("");
  try {
    const r = await api(`api/random?voice=${encodeURIComponent(voice)}`);
    $("text").value = r.text;
    const ref = $("reference");
    if (r.audio) { $("ref-audio").src = wavUrl(r.audio); $("ref-id").textContent = r.id; ref.hidden = false; }
    else { ref.hidden = true; }
    await speak();
  } catch (e) { showError(e.message); }
}

function renderPhonemes(items) {
  const box = $("phonemes");
  box.innerHTML = "";
  for (const p of items) {
    const d = document.createElement("div");
    d.className = "ph" + (p.unknown ? " unk" : "");
    const color = TONES[p.tone.split("+")[0]];
    d.innerHTML = `<b></b><span class="ipa"></span><span class="tn"></span>`;
    d.querySelector("b").textContent = p.word;
    d.querySelector(".ipa").textContent = p.unknown ? "<unk>" : "/" + p.phones.join(" ") + "/";
    const tn = d.querySelector(".tn");
    tn.textContent = p.tone;
    if (color) tn.style.color = css(color);
    box.appendChild(d);
  }
}

function renderLegend() {
  $("legend").innerHTML = Object.entries(TONES)
    .map(([name, v]) => `<span><i style="background:${css(v)}"></i>${name}</span>`).join("");
}

function renderPlot(tracks, seconds) {
  const svg = $("plot");
  const NS = "http://www.w3.org/2000/svg";
  const width = Math.max(svg.parentElement.clientWidth, Math.min(2400, seconds * 160));
  const height = 260, left = 44, right = 12, top = 34, bottom = 26;
  svg.setAttribute("width", width); svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = "";
  const all = tracks.flatMap((t) => t.hz.filter((x) => x !== null)).sort((a, b) => a - b);
  if (!all.length) return;
  const lo = all[Math.floor(all.length * 0.01)] * 0.85, hi = all[Math.ceil(all.length * 0.99) - 1] * 1.15;
  const x = (s) => left + (s / Math.max(seconds, 0.1)) * (width - left - right);
  const y = (hz) => top + (1 - (hz - lo) / (hi - lo)) * (height - top - bottom);
  const el = (name, attrs, text) => {
    const e = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    svg.appendChild(e); return e;
  };
  const stepHz = Math.pow(10, Math.floor(Math.log10((hi - lo) / 3)));
  for (let v = Math.ceil(lo / (stepHz * 2)) * stepHz * 2; v < hi; v += stepHz * 2) {
    el("line", { x1: left, x2: width - right, y1: y(v), y2: y(v), class: "grid-line" });
    el("text", { x: left - 6, y: y(v) + 4, "text-anchor": "end", class: "axis" }, Math.round(v));
  }
  for (let s = 0; s <= seconds; s += seconds > 12 ? 2 : 1) {
    el("text", { x: x(s), y: height - 8, "text-anchor": "middle", class: "axis" }, `${s}s`);
  }
  const path = (t, a, b) => {
    let d = "", pen = false;
    for (let i = a; i < b && i < t.hz.length; i++) {
      const v = t.hz[i];
      if (v === null) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${x(t.start + i * t.step).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    }
    return d;
  };
  for (const t of tracks) {
    el("path", { d: path(t, 0, t.hz.length), class: "base" });
    for (const w of t.words) {
      for (const s of w.spans) {
        const d = path(t, s.a, s.b);
        if (d) el("path", { d, class: "tone", stroke: css(TONES[s.tone] || "--t-en") });
      }
      if (w.b > w.a) {
        el("line", { x1: x(t.start + w.a * t.step), x2: x(t.start + w.a * t.step), y1: top - 4, y2: height - bottom, class: "grid-line" });
        el("text", { x: x(t.start + ((w.a + w.b) / 2) * t.step), y: top - 10, "text-anchor": "middle", class: "word" }, w.word);
      }
    }
  }
}

async function init() {
  for (const s of SLIDERS) {
    const input = $(s), out = $(`${s}-v`);
    const show = () => { out.textContent = input.value; };
    input.addEventListener("input", show); show();
  }
  $("go").onclick = speak;
  $("random").onclick = randomSentence;
  $("text").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) speak(); });
  renderLegend();
  try {
    const info = await api("api/info");
    $("checkpoint").textContent = info.checkpoint.replace(".pt", "");
    voice = info.default_voice;
    renderVoices(info.voices);
    $("random").hidden = !info.test_sets.length;
    $("text").value = info.examples[0];
    const chips = $("examples");
    for (const ex of info.examples) {
      const c = document.createElement("button");
      c.className = "chip"; c.textContent = ex; c.title = ex;
      c.onclick = () => { $("text").value = ex; };
      chips.appendChild(c);
    }
    if (!info.punctuation_available) {
      $("punctuate").disabled = true;
      $("punctuate").parentElement.title = "Chưa tải mô hình vibert-capu (scripts/setup.sh)";
    }
  } catch (e) { showError("Không kết nối được server: " + e.message); }
}
init();
