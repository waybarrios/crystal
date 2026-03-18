/* ============================================================
   CRYSTAL project page: interactions
   Architecture adapted from the MoDA project page.
   ============================================================ */
(function () {
  "use strict";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $  = (s, c = document) => c.querySelector(s);
  const $$ = (s, c = document) => Array.from(c.querySelectorAll(s));
  const SVGNS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };

  /* ---------- 1. KaTeX ---------- */
  function renderMath() {
    if (!window.katex) return;
    $$(".eq[data-tex]").forEach((el) => {
      try { window.katex.render(el.dataset.tex, el, { displayMode: true, throwOnError: false }); }
      catch (e) { /* keep raw text */ }
    });
  }

  /* ---------- 2. Hero refractor ----------
     Real benchmark exercises refract through the crystal into graded
     reasoning steps: ✓ matched (green), ✗ contradiction (red),
     ○ omitted reference step (grey). Three real examples cycle, telling
     CRYSTAL's story: a lucky guess, a sound chain, and cherry-picking. */
  const ICON = { ok: "✓", miss: "✗", ghost: "○" };
  const EXERCISES = [
    {
      img: "teaser_sample0.jpg", src: "RealWorldQA · perception",
      q: "Which of the 3 consoles is the smallest?", gt: "C (the middle one)",
      f1: 0.15, verdict: "lucky guess",
      steps: [
        { t: "Three consoles sit on the desk", s: "ok" },
        { t: "The middle one looks larger", s: "miss" },
        { t: "Compare their relative sizes", s: "ghost" },
        { t: "Most powerful, by its size", s: "miss" },
        { t: "Answer: C, the middle console", s: "ok" },
      ],
    },
    {
      img: "dataset_example_sample3.jpg", src: "RealWorldQA · perception",
      q: "What color are the traffic lights?", gt: "Green",
      f1: 0.86, verdict: "sound chain",
      steps: [
        { t: "Locate the traffic signals", s: "ok" },
        { t: "Both signals are lit green", s: "ok" },
        { t: "No red or amber is visible", s: "ok" },
        { t: "Answer: Green", s: "ok" },
      ],
    },
    {
      img: "example_scienceqa_3807.jpg", src: "ScienceQA · food web",
      q: "Which organism in this web is a producer?", gt: "Bilberry",
      f1: 0.55, verdict: "correct, skips steps",
      steps: [
        { t: "Producers make their own energy", s: "ok" },
        { t: "The animals here are consumers", s: "ghost" },
        { t: "Lichen & bear sedge are producers too", s: "ghost" },
        { t: "Bilberry is a plant → producer", s: "ok" },
        { t: "Answer: Bilberry", s: "ok" },
      ],
    },
  ];
  const F1COL = (v) => (v < 0.4 ? "var(--miss)" : v < 0.7 ? "var(--amber)" : "var(--match)");

  function buildRefractor() {
    const fig = $("#refractor"); if (!fig) return;
    const rx = $("#rx"), rays = $("#rxRays"), imgEl = $("#rxImg"), srcEl = $("#rxSrc"),
      qEl = $("#rxQ"), gtEl = $("#rxGT"), stepsEl = $("#rxSteps"), scoreEl = $("#rxScore"), dotsEl = $("#rxDots");
    let cur = 0, timer = null;

    EXERCISES.forEach((_, i) => {
      const b = document.createElement("button");
      b.className = "rxdot"; b.type = "button"; b.setAttribute("role", "tab");
      b.setAttribute("aria-label", "Example " + (i + 1));
      b.addEventListener("click", () => { show(i); restart(); });
      dotsEl.appendChild(b);
    });
    const dots = $$(".rxdot", dotsEl);

    function drawRays() {
      rays.innerHTML = "";
      const prism = $(".rx__prism"), cryst = $("#rxCrystal");
      if (!prism || prism.offsetWidth === 0 || !cryst) return;   // prism hidden → stacked on mobile
      const box = rx.getBoundingClientRect(), pr = cryst.getBoundingClientRect();
      const ox = pr.right - box.left - 6, oy = pr.top + pr.height / 2 - box.top;
      // beam from image -> crystal
      const im = $("#rxImgwrap").getBoundingClientRect();
      rays.appendChild(svgEl("line", { x1: im.right - box.left, y1: im.top + im.height / 2 - box.top, x2: pr.left - box.left + 6, y2: oy, stroke: "var(--facet)", "stroke-width": 2, opacity: ".5" }));
      // crystal -> each step
      $$(".rxstep", stepsEl).forEach((st, i) => {
        const r = st.getBoundingClientRect();
        const tx = r.left - box.left, ty = r.top + r.height / 2 - box.top;
        const s = EXERCISES[cur].steps[i].s;
        const col = s === "ok" ? "#16a34a" : s === "miss" ? "#dc2626" : "#93a0b8";
        rays.appendChild(svgEl("path", {
          d: `M ${ox} ${oy} C ${ox + 30} ${oy}, ${tx - 34} ${ty}, ${tx - 3} ${ty}`,
          fill: "none", stroke: col, "stroke-width": s === "ghost" ? 1.4 : 1.8,
          "stroke-dasharray": s === "ghost" ? "3 4" : "none", opacity: s === "ghost" ? ".4" : ".72",
        }));
      });
    }

    function show(i) {
      cur = i;
      const ex = EXERCISES[i];
      imgEl.style.opacity = 0;
      setTimeout(() => { imgEl.src = "static/images/" + ex.img; imgEl.alt = ex.q; imgEl.style.opacity = 1; }, reduceMotion ? 0 : 160);
      srcEl.textContent = ex.src;
      qEl.textContent = "“" + ex.q + "”";
      gtEl.textContent = ex.gt;
      scoreEl.innerHTML = `Match F1 <b class="rxf1" style="color:${F1COL(ex.f1)}">${ex.f1.toFixed(2)}</b> · ${ex.verdict}`;
      stepsEl.innerHTML = ex.steps.map((s) =>
        `<div class="rxstep ${s.s}"><span class="ic">${ICON[s.s]}</span><span>${s.t}</span></div>`).join("");
      dots.forEach((d, k) => { d.classList.toggle("is-on", k === i); d.setAttribute("aria-selected", k === i); });
      requestAnimationFrame(() => {
        drawRays();
        if (!reduceMotion) $$(".rxstep", stepsEl).forEach((st, k) => {
          st.style.opacity = 0; st.style.transform = "translateX(10px)";
          setTimeout(() => { st.style.opacity = ""; st.style.transform = ""; }, 90 + k * 95);
        });
      });
    }
    function next() { show((cur + 1) % EXERCISES.length); }
    function restart() { if (timer) clearInterval(timer); if (!reduceMotion) timer = setInterval(next, 5400); }

    window.addEventListener("resize", drawRays);
    fig.addEventListener("mouseenter", () => { if (timer) clearInterval(timer); });
    fig.addEventListener("mouseleave", restart);

    show(0);
    if ("IntersectionObserver" in window && !reduceMotion) {
      const io = new IntersectionObserver((es) => es.forEach((e) => {
        if (e.isIntersecting) restart(); else if (timer) clearInterval(timer);
      }), { threshold: 0.25 });
      io.observe(fig);
    }
  }

  /* ---------- 3. Dataset source bar ---------- */
  const SOURCES = [
    { source: "MathVision",  count: 3040, percent: 47.7, color: "#2563eb" },
    { source: "ScienceQA",   count: 2017, percent: 31.7, color: "#22d3ee" },
    { source: "RealWorldQA", count: 765,  percent: 12.0, color: "#7c3aed" },
    { source: "MMVP",        count: 300,  percent: 4.7,  color: "#16a34a" },
    { source: "PlotQA",      count: 250,  percent: 3.9,  color: "#d97706" },
  ];
  function buildSourceBar() {
    const bar = $("#srcbar"), leg = $("#srclegend");
    if (!bar || !leg) return;
    SOURCES.forEach((s) => {
      const seg = document.createElement("span");
      seg.style.width = s.percent + "%"; seg.style.background = s.color;
      seg.textContent = s.percent >= 8 ? s.percent + "%" : "";
      seg.title = `${s.source}: ${s.count} (${s.percent}%)`;
      bar.appendChild(seg);
      const li = document.createElement("span");
      li.innerHTML = `<i style="background:${s.color}"></i>${s.source} · ${s.count.toLocaleString()}`;
      leg.appendChild(li);
    });
  }

  /* ---------- 4. Interactive step-matching demo ---------- */
  const REF = [
    "Three gaming consoles sit on the desk",
    "All three are the same console model",
    "Estimate each console's physical size",
    "The left and right consoles look larger",
    "The middle console is the smallest",
    "Therefore the answer is C (the middle one)",
  ];
  // ref: index of the reference step this predicted step matches (null = no match / contradiction)
  const PRED = [
    { text: "There are three consoles on the desk",        ref: 0 },
    { text: "The middle console looks larger than the rest", ref: null },
    { text: "On a closer look the middle one is smallest",  ref: 4 },
    { text: "So the outer two are the larger consoles",     ref: 3 },
    { text: "The smallest is C, the middle console",        ref: 5 },
  ];

  function lisRatio(seq) {                       // longest non-decreasing subsequence ratio
    if (!seq.length) return 1;
    const tails = [];
    seq.forEach((x) => {
      let lo = 0, hi = tails.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (tails[mid] <= x) lo = mid + 1; else hi = mid; }
      tails[lo] = x;
    });
    return tails.length / seq.length;
  }

  function buildMatchDemo() {
    const refCol = $("#refCol"), predCol = $("#predCol"), svg = $("#matchSvg"), demo = $("#matchdemo");
    if (!refCol || !predCol) return;
    const orderBtn = $("#demoOrder"), replayBtn = $("#demoReplay");

    // build step nodes
    const refNodes = REF.map((t, i) => {
      const d = document.createElement("div"); d.className = "mstep"; d.dataset.ri = i;
      d.innerHTML = `<span>${t}</span>`; refCol.appendChild(d); return d;
    });
    const predNodes = PRED.map((p, i) => {
      const d = document.createElement("div"); d.className = "mstep"; d.dataset.pi = i;
      d.innerHTML = `<span>${p.text}</span>`; predCol.appendChild(d); return d;
    });

    const matched = PRED.filter((p) => p.ref !== null).length;
    const prec = matched / PRED.length;
    const rec = matched / REF.length;
    const f1 = prec + rec ? (2 * prec * rec) / (prec + rec) : 0;
    const order = PRED.filter((p) => p.ref !== null).map((p) => p.ref);   // ref indices in predicted order
    const lis = lisRatio(order);
    const ordF1 = f1 * (0.7 + 0.3 * lis);

    function drawLines() {
      svg.innerHTML = "";
      const wrap = $("#matchCanvas").getBoundingClientRect();
      PRED.forEach((p, pi) => {
        if (p.ref === null) return;
        const r = refNodes[p.ref].getBoundingClientRect();
        const pr = predNodes[pi].getBoundingClientRect();
        const y1 = r.top + r.height / 2 - wrap.top;
        const y2 = pr.top + pr.height / 2 - wrap.top;
        // out-of-order edge highlighted when "Ordered" is on
        const bad = orderOn && !isAscendingUpTo(order, p.ref);
        const path = svgEl("path", {
          d: `M 0 ${y1} C 28 ${y1}, 28 ${y2}, 56 ${y2}`,
          fill: "none", "stroke-width": bad ? 2.4 : 2,
          stroke: bad ? "#d97706" : "#16a34a",
          "stroke-dasharray": bad ? "5 4" : "none", opacity: 0.85,
        });
        svg.appendChild(path);
      });
    }
    // is this ref index part of an ascending run (helper to flag the crossing edge)
    function isAscendingUpTo(seq, refv) {
      // flag edges that go "backwards": a matched ref smaller than a previously matched ref
      const pos = seq.indexOf(refv);
      for (let k = 0; k < pos; k++) if (seq[k] > refv) return false;
      return true;
    }

    let orderOn = false;
    function setStates(animate) {
      predNodes.forEach((d, pi) => {
        d.classList.remove("is-match", "is-miss");
        const p = PRED[pi];
        const tag = d.querySelector(".mstep__tag") || (() => { const s = document.createElement("span"); s.className = "mstep__tag"; d.appendChild(s); return s; })();
        if (p.ref === null) { d.classList.add("is-miss"); tag.textContent = "✗ no match"; tag.style.color = "var(--miss)"; }
        else { d.classList.add("is-match"); tag.textContent = "✓ match"; tag.style.color = "var(--match)"; }
      });
      const covered = new Set(PRED.filter((p) => p.ref !== null).map((p) => p.ref));
      refNodes.forEach((d, ri) => { d.classList.toggle("is-uncov", !covered.has(ri)); d.classList.toggle("is-match", covered.has(ri)); });
      drawLines();
    }

    function setReadout() {
      animateNum($("#roP"), prec);
      animateNum($("#roR"), rec);
      animateNum($("#roF1"), f1);
      animateNum($("#roOrd"), ordF1);          // Ordered F1 is order-penalized, always <= Match F1
      const note = $("#demoNote");
      if (note) {
        const inOrder = Math.round(lis * matched);
        note.innerHTML = orderOn
          ? `Matched steps, sorted by reference position: <b>${inOrder} of ${matched}</b> stay in order (LIS&nbsp;=&nbsp;${inOrder}/${matched}). Ordered F1 scales Match F1 by&nbsp;×${(0.7 + 0.3 * lis).toFixed(3)} &nbsp;→&nbsp; <b>${f1.toFixed(2)} drops to ${ordF1.toFixed(2)}</b>.`
          : `<b>${matched} of ${PRED.length}</b> predicted steps match a reference; <b>${REF.length - matched}</b> reference steps are omitted (recall&nbsp;${rec.toFixed(2)}), the cherry-picking gap. Toggle <b>Ordered</b> to also penalize out-of-sequence steps.`;
      }
    }
    function animateNum(el, target) {
      if (!el) return;
      if (reduceMotion) { el.textContent = target.toFixed(2); return; }
      const t0 = performance.now(), dur = 700;
      const tick = (t) => { const k = Math.max(0, Math.min(1, (t - t0) / dur)); el.textContent = (target * (1 - Math.pow(1 - k, 3))).toFixed(2); if (k < 1) requestAnimationFrame(tick); };
      requestAnimationFrame(tick);
    }

    function play() {
      // reset
      predNodes.forEach((d) => { d.classList.remove("is-match", "is-miss"); const t = d.querySelector(".mstep__tag"); if (t) t.remove(); });
      refNodes.forEach((d) => d.classList.remove("is-uncov", "is-match"));
      svg.innerHTML = "";
      if (reduceMotion) { setStates(); setReadout(); return; }
      // staggered reveal
      predNodes.forEach((d, pi) => setTimeout(() => {
        const p = PRED[pi];
        const tag = document.createElement("span"); tag.className = "mstep__tag";
        if (p.ref === null) { d.classList.add("is-miss"); tag.textContent = "✗ no match"; tag.style.color = "var(--miss)"; }
        else { d.classList.add("is-match"); tag.textContent = "✓ match"; tag.style.color = "var(--match)"; if (refNodes[p.ref]) refNodes[p.ref].classList.add("is-match"); }
        d.appendChild(tag);
        d.style.transform = "translateX(6px)"; requestAnimationFrame(() => { d.style.transition = "transform .4s"; d.style.transform = ""; });
        drawLines();
      }, 350 + pi * 480));
      setTimeout(() => {
        const covered = new Set(PRED.filter((p) => p.ref !== null).map((p) => p.ref));
        refNodes.forEach((d, ri) => { if (!covered.has(ri)) d.classList.add("is-uncov"); });
        setReadout();
      }, 350 + PRED.length * 480);
    }

    if (orderBtn) orderBtn.addEventListener("click", () => {
      orderOn = !orderOn; orderBtn.classList.toggle("is-on", orderOn); orderBtn.setAttribute("aria-pressed", orderOn);
      drawLines(); setReadout();
    });
    if (replayBtn) replayBtn.addEventListener("click", play);
    window.addEventListener("resize", () => { if (svg.childNodes.length) drawLines(); });

    // play once when scrolled into view
    if ("IntersectionObserver" in window && !reduceMotion) {
      const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { play(); io.unobserve(e.target); } }), { threshold: 0.4 });
      io.observe(demo);
    } else { setStates(); setReadout(); }
  }

  /* ---------- 5. Results table ---------- */
  const MODELS = [
    { grp: "Commercial MLLMs", cat: "commercial" },
    { m: "GPT-5",            p: "n/a", acc: 57.99, f1: 0.612, P: 0.925, R: 0.479, st: 5.29,  lis: 0.636, ord: 0.539, cat: "commercial" },
    { m: "GPT-5-mini",       p: "n/a", acc: 55.59, f1: 0.773, P: 0.978, R: 0.669, st: 7.57,  lis: 0.560, ord: 0.670, cat: "commercial" },
    { m: "GPT-5.2 Instant",  p: "n/a", acc: 47.35, f1: 0.564, P: 0.974, R: 0.416, st: 4.64,  lis: 0.648, ord: 0.501, cat: "commercial" },
    { m: "Gemini 2.5 Flash", p: "n/a", acc: 53.95, f1: 0.673, P: 0.701, R: 0.765, st: 17.10, lis: 0.584, ord: 0.579, cat: "commercial" },
    { grp: "Qwen Family", cat: "open" },
    { m: "Qwen3-VL-8B",    p: "8B",  acc: 57.66, f1: 0.659, P: 0.827, R: 0.590, st: 7.37,  lis: 0.624, ord: 0.572, cat: "open" },
    { m: "Qwen3-VL-32B",   p: "32B", acc: 49.22, f1: 0.718, P: 0.819, R: 0.704, st: 10.56, lis: 0.581, ord: 0.617, cat: "open" },
    { m: "Qwen2.5-VL-32B", p: "32B", acc: 47.63, f1: 0.653, P: 0.943, R: 0.524, st: 5.86,  lis: 0.619, ord: 0.572, cat: "open" },
    { m: "Qwen3-VL-2B",    p: "2B",  acc: 34.15, f1: 0.595, P: 0.726, R: 0.535, st: 5.94,  lis: 0.669, ord: 0.515, cat: "open" },
    { m: "Qwen2.5-VL-7B",  p: "7B",  acc: 30.43, f1: 0.475, P: 0.765, R: 0.365, st: 4.07,  lis: 0.717, ord: 0.422, cat: "open" },
    { m: "Qwen2.5-VL-3B",  p: "3B",  acc: 39.85, f1: 0.480, P: 0.898, R: 0.347, st: 3.73,  lis: 0.723, ord: 0.434, cat: "open" },
    { grp: "InternVL Family", cat: "open" },
    { m: "InternVL3.5-38B", p: "38B", acc: 51.21, f1: 0.612, P: 0.892, R: 0.498, st: 5.76, lis: 0.643, ord: 0.538, cat: "open" },
    { m: "InternVL3.5-8B",  p: "8B",  acc: 51.98, f1: 0.530, P: 0.882, R: 0.416, st: 4.96, lis: 0.692, ord: 0.469, cat: "open" },
    { m: "InternVL3.5-4B",  p: "4B",  acc: 37.61, f1: 0.432, P: 0.895, R: 0.325, st: 3.75, lis: 0.775, ord: 0.387, cat: "open" },
    { m: "InternVL3.5-2B",  p: "2B",  acc: 33.02, f1: 0.469, P: 0.725, R: 0.371, st: 3.92, lis: 0.731, ord: 0.415, cat: "open" },
    { m: "InternVL3.5-1B",  p: "1B",  acc: 30.13, f1: 0.330, P: 0.616, R: 0.243, st: 2.51, lis: 0.807, ord: 0.297, cat: "open" },
    { grp: "Other Open-Source", cat: "open" },
    { m: "Gemma3-12B",     p: "12B", acc: 33.83, f1: 0.605, P: 0.838, R: 0.499, st: 5.51, lis: 0.673, ord: 0.534, cat: "open" },
    { m: "Gemma3-4B",      p: "4B",  acc: 28.65, f1: 0.618, P: 0.878, R: 0.506, st: 5.72, lis: 0.668, ord: 0.547, cat: "open" },
    { m: "Llama 3.2-11B",  p: "11B", acc: 24.83, f1: 0.471, P: 0.713, R: 0.379, st: 4.19, lis: 0.726, ord: 0.415, cat: "open" },
    { m: "LLaVA-v1.6-7B",  p: "7B",  acc: 24.66, f1: 0.512, P: 0.961, R: 0.370, st: 3.94, lis: 0.675, ord: 0.459, cat: "open" },
    { m: "MiniCPMv2.6-8B", p: "8B",  acc: 25.54, f1: 0.215, P: 0.709, R: 0.134, st: 1.31, lis: 0.854, ord: 0.186, cat: "open" },
  ];
  const RANK_COLS = ["acc", "f1", "P", "R", "ord"];
  function topThree() {
    const rows = MODELS.filter((r) => !r.grp);
    const ranks = {};
    RANK_COLS.forEach((c) => {
      const sorted = [...rows].sort((a, b) => b[c] - a[c]);
      ranks[c] = [sorted[0]?.m, sorted[1]?.m, sorted[2]?.m];
    });
    return ranks;
  }
  function rankClass(ranks, col, model) {
    const i = ranks[col].indexOf(model);
    return i === 0 ? "r1" : i === 1 ? "r2" : i === 2 ? "r3" : "";
  }
  function fmtAcc(v) { return v.toFixed(2) + "%"; }
  function renderResults(cat) {
    const ranks = topThree();
    const cols = [
      ["acc", "Accuracy", fmtAcc], ["f1", "Match F1", (v) => v.toFixed(3)],
      ["P", "P", (v) => v.toFixed(3)], ["R", "R", (v) => v.toFixed(3)],
      ["st", "Steps", (v) => v.toFixed(2)], ["lis", "LIS", (v) => v.toFixed(3)],
      ["ord", "Ord. F1", (v) => v.toFixed(3)],
    ];
    let head = `<tr><th>Model</th><th class="params">Params</th>${cols.map((c) => `<th>${c[1]}</th>`).join("")}</tr>`;
    let body = "";
    MODELS.forEach((row) => {
      if (cat !== "all" && row.cat !== cat) return;
      if (row.grp) { body += `<tr class="grp"><td colspan="9">${row.grp}</td></tr>`; return; }
      body += `<tr><td class="model">${row.m}</td><td class="params">${row.p}</td>` +
        cols.map((c) => {
          const rc = RANK_COLS.includes(c[0]) ? rankClass(ranks, c[0], row.m) : "";
          return `<td><span class="${rc}">${c[2](row[c[0]])}</span></td>`;
        }).join("") + `</tr>`;
    });
    return `<table class="res"><thead>${head}</thead><tbody>${body}</tbody></table>`;
  }
  function buildResults() {
    const wrap = $("#tablewrap");
    if (!wrap) return;
    wrap.innerHTML = renderResults("all");
    $$(".tab").forEach((tab) => tab.addEventListener("click", () => {
      $$(".tab").forEach((x) => { x.classList.remove("is-active"); x.setAttribute("aria-selected", "false"); });
      tab.classList.add("is-active"); tab.setAttribute("aria-selected", "true");
      wrap.innerHTML = renderResults(tab.dataset.tab);
    }));
  }

  /* ---------- 6. Headline cards ---------- */
  const HEADLINE = [
    { n: "20", t: "MLLMs evaluated", s: "4 commercial + 16 open-source" },
    { n: "0.773", t: "Best Match F1", s: "GPT-5-mini · acc 55.6%" },
    { n: "19/20", t: "Cherry-pick", s: "precision ≫ recall" },
    { n: "<60%", t: "Steps in order", s: "no competitive model" },
  ];
  function buildHeadline() {
    const el = $("#headline"); if (!el) return;
    el.innerHTML = HEADLINE.map((h) =>
      `<div class="hcard"><span class="hcard__delta">${h.n}</span><span class="hcard__bench">${h.t}</span><span class="hcard__fam">${h.s}</span></div>`).join("");
  }

  /* ---------- 7. Cherry-picking bars ---------- */
  const CHERRY = [
    { m: "GPT-5",            P: 0.925, R: 0.479 },
    { m: "GPT-5-mini",       P: 0.978, R: 0.669 },
    { m: "Qwen3-VL-32B",     P: 0.819, R: 0.704 },
    { m: "Qwen2.5-VL-3B",    P: 0.898, R: 0.347 },
    { m: "InternVL3.5-8B",   P: 0.882, R: 0.416 },
    { m: "LLaVA-v1.6-7B",    P: 0.961, R: 0.370 },
    { m: "MiniCPMv2.6-8B",   P: 0.709, R: 0.134 },
    { m: "Gemini 2.5 Flash", P: 0.701, R: 0.765 },
  ];
  function buildCherry() {
    const el = $("#cherrybars"); if (!el) return;
    el.innerHTML = CHERRY.map((c) =>
      `<div class="cbar" data-p="${c.P}" data-r="${c.R}">
         <span class="cbar__lab">${c.m}</span>
         <div class="cbar__track"><span class="cbar__p" title="Precision ${c.P}"></span><span class="cbar__r" title="Recall ${c.R}"></span></div>
       </div>`).join("");
    const fill = (bar) => {
      bar.querySelector(".cbar__p").style.width = (bar.dataset.p * 100) + "%";
      bar.querySelector(".cbar__r").style.width = (bar.dataset.r * 100) + "%";
    };
    const bars = $$(".cbar", el);
    if (reduceMotion || !("IntersectionObserver" in window)) { bars.forEach(fill); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { fill(e.target); io.unobserve(e.target); } }), { threshold: 0.5 });
    bars.forEach((b) => io.observe(b));
  }

  /* ---------- 8. GRPO tables ---------- */
  const GRPO_ROWS = [
    { grp: "Qwen2.5-VL-3B" },
    { l: "Baseline",       acc: "39.85", f1: "0.480", P: "0.898", R: "0.347", ord: "0.434" },
    { l: "CPR-Curriculum", acc: "47.52", f1: "0.633", P: "0.963", R: "0.493", ord: "0.560", ours: true },
    { l: "Δ",              acc: "+7.67", f1: "+0.153", P: "+0.065", R: "+0.146", ord: "+0.126", delta: true },
    { grp: "InternVL3.5-4B" },
    { l: "Baseline",       acc: "37.61", f1: "0.432", P: "0.895", R: "0.325", ord: "0.387" },
    { l: "CPR-Curriculum", acc: "45.76", f1: "0.833", P: "0.903", R: "0.811", ord: "0.719", ours: true },
    { l: "Δ",              acc: "+8.15", f1: "+0.401", P: "+0.008", R: "+0.486", ord: "+0.332", delta: true },
  ];
  function buildGrpoTables() {
    const mt = $("#grpotable"); if (!mt) return;
    const cell = (v, d) => d ? `<span class="${v.startsWith("+") ? "up" : "down"}">${v}</span>` : v;
    const body = GRPO_ROWS.map((r) => {
      if (r.grp) return `<tr class="grp"><td colspan="6">${r.grp}</td></tr>`;
      const cls = r.ours ? "is-ours" : r.delta ? "delta" : "";
      return `<tr class="${cls}"><td class="model">${r.l}</td><td>${cell(r.acc, r.delta)}</td><td>${cell(r.f1, r.delta)}</td><td>${cell(r.P, r.delta)}</td><td>${cell(r.R, r.delta)}</td><td>${cell(r.ord, r.delta)}</td></tr>`;
    }).join("");
    mt.innerHTML = `<table class="cmp"><thead><tr><th>Configuration</th><th>Acc</th><th>Match F1</th><th>Prec</th><th>Rec</th><th>Ord. F1</th></tr></thead><tbody>${body}</tbody></table>`;
  }

  /* ---------- 9. Qualitative gallery ---------- */
  const GALLERY = [
    {
      f: "crystal_ex1_sample1265.jpg", src: "MathVision · Sample 1265", n: 12, more: 2, gt: "A",
      q: "Anna follows the arrow and turns at each crossing: right, left, left, right, left, left. What does she reach at the next crossing?",
      steps: [
        "Anna starts in the direction of an arrow.",
        "At each crossing she turns either right or left.",
        "First crossing: turn right.",
        "Second crossing: left.",
        "Third crossing: left again.",
        "Fourth crossing: right.",
        "Fifth crossing: left.",
        "Sixth crossing: left again.",
        "Find what she reaches at the next crossing after this sequence.",
        "Answer is multiple-choice (A to E).",
      ],
    },
    {
      f: "crystal_ex3_sample1947.jpg", src: "MathVision · Sample 1947", n: 13, more: 3, gt: "B (60°)",
      q: "In triangle PSQ, ∠QPS = 20° and PQ = PR = QS, with line QR splitting it. How big is angle RQS?",
      steps: [
        "∠QPS = 20°.",
        "Line QR splits triangle PSQ into PQR and RQS (R on PS).",
        "PQ = QS, so triangle PQS is isosceles with base PS.",
        "Base angles are equal: ∠QPS = ∠QSP = 20°.",
        "Triangle sum: ∠PQS = 180° − 20° − 20° = 140°.",
        "PQ = PR, so triangle PQR is isosceles with base QR.",
        "∠QPR = ∠QPS = 20°, since R lies on PS.",
        "Base angles ∠PQR = ∠PRQ = (180° − 20°)/2 = 80°.",
        "At Q the angle splits: ∠PQS = ∠PQR + ∠RQS.",
        "Substitute ∠PQS = 140° and ∠PQR = 80°.",
      ],
    },
    {
      f: "crystal_ex4_sample6193.jpg", src: "PlotQA · Sample 6193", n: 8, more: 0, gt: "3",
      q: "How many bars are there on the 4th tick from the left?",
      steps: [
        "The question asks how many bars sit at the fourth x-axis tick.",
        "X-axis labels in order: Indonesia, Israel, Italy, Jamaica, then repeat.",
        "The fourth tick is the first 'Jamaica' label.",
        "The chart has three series (2005, 2006, 2007), each a colored bar.",
        "At the Jamaica tick there is one bar per series.",
        "Counting gives three bars at the fourth tick.",
        "Check that the total (3) matches the expected value.",
        "Return the numeric value: 3.",
      ],
    },
    {
      f: "crystal_ex5_sample4306.jpg", src: "ScienceQA · Sample 4306", n: 8, more: 0, gt: "Africa",
      q: "Which of these continents does the equator intersect? (North America, Africa, Europe)",
      steps: [
        "The equator is the line at 0° latitude.",
        "Lines of latitude, including the equator, circle the Earth.",
        "The equator splits Earth into the Northern and Southern Hemispheres.",
        "Africa spans both hemispheres.",
        "Europe lies entirely north of the equator.",
        "North America lies entirely north of the equator.",
        "So the equator crosses Africa.",
        "Select Africa among the options.",
      ],
    },
    {
      f: "crystal_ex9_sample714.jpg", src: "RealWorldQA · Sample 714", n: 9, more: 0, gt: "Yes",
      q: "Is there a stop sign facing us?",
      steps: [
        "Scene: a nighttime residential street, curbs and sidewalks on the right.",
        "Locate a red octagonal sign on a pole near the right curb ahead.",
        "It has a white border and white letters consistent with 'STOP'.",
        "The sign face points toward the camera (frontal octagon), not edge-on.",
        "It sits at the near-right corner of our approach lane.",
        "Lighting lets the sign face be inspected from the camera.",
        "No other stop-sign faces could be confused with it.",
        "A red octagon with white letters facing us means a stop sign faces us.",
        "Answer: Yes.",
      ],
    },
    {
      f: "crystal_ex11_sample5866.jpg", src: "MMVP · Sample 5866", n: 9, more: 0, gt: "A) Yes",
      q: "Is the shark's belly visible in this image?",
      steps: [
        "A single shark is centered against blue water.",
        "It is viewed from slightly below, exposing its underside.",
        "The ventral surface looks pale against the darker top.",
        "The lower body from snout to tail is unobstructed.",
        "The undersides of the pectoral fins are visible.",
        "Lighting separates the pale belly from the dorsal side.",
        "No objects or shadows hide the underside.",
        "These cues indicate the belly is in view.",
        "Select option A.",
      ],
    },
  ];
  function buildGallery() {
    const el = $("#gallery"); if (!el) return;
    el.innerHTML = GALLERY.map((g) => {
      const steps = g.steps.map((s) => `<li>${s}</li>`).join("");
      const more = g.more ? `<span class="excard__more">+${g.more} more step${g.more > 1 ? "s" : ""}</span>` : "";
      return `<figure class="excard">
        <div class="excard__media"><img loading="lazy" src="static/images/${g.f}" alt="${g.src}" /><span class="excard__src">${g.src}</span></div>
        <div class="excard__body">
          <p class="excard__q">${g.q}</p>
          <span class="excard__gt">Ground truth&nbsp;·&nbsp;<b>${g.gt}</b></span>
          <span class="excard__steplab">Reference reasoning · ${g.n} steps</span>
          <ol class="excard__steps">${steps}</ol>
          ${more}
        </div>
      </figure>`;
    }).join("");
  }

  /* ---------- 10. Scroll reveal ---------- */
  function buildReveal() {
    const els = $$(".reveal");
    if (reduceMotion || !("IntersectionObserver" in window)) { els.forEach((e) => e.classList.add("is-in")); return; }
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (en.isIntersecting) { en.target.classList.add("is-in"); io.unobserve(en.target); }
    }), { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
    els.forEach((e) => io.observe(e));
  }

  /* ---------- 11. Count-up stats ---------- */
  function buildCounters() {
    const els = $$(".keystats__n[data-count]");
    if (reduceMotion || !("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((entries) => entries.forEach((en) => {
      if (!en.isIntersecting) return;
      const el = en.target, target = +el.dataset.count;
      let cur = 0; const t0 = performance.now(), dur = 1100;
      const tick = (t) => { const k = Math.max(0, Math.min(1, (t - t0) / dur)); cur = Math.round(target * (1 - Math.pow(1 - k, 3))); el.textContent = cur.toLocaleString(); if (k < 1) requestAnimationFrame(tick); };
      requestAnimationFrame(tick); io.unobserve(el);
    }), { threshold: 0.6 });
    els.forEach((e) => io.observe(e));
  }

  /* ---------- 12. Nav scrolled state ---------- */
  function buildNav() {
    const nav = $(".nav"); if (!nav) return;
    const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 24);
    onScroll(); window.addEventListener("scroll", onScroll, { passive: true });
  }

  /* ---------- 13. Copy BibTeX ---------- */
  function buildCopy() {
    const btn = $("#copyBib"), pre = $("#bibtex"); if (!btn || !pre) return;
    btn.addEventListener("click", async () => {
      const txt = pre.innerText;
      try { await navigator.clipboard.writeText(txt); }
      catch (e) { const r = document.createRange(); r.selectNode(pre); const s = getSelection(); s.removeAllRanges(); s.addRange(r); try { document.execCommand("copy"); } catch (_) {} s.removeAllRanges(); }
      const label = $("span", btn); btn.classList.add("is-copied"); if (label) label.textContent = "Copied";
      setTimeout(() => { btn.classList.remove("is-copied"); if (label) label.textContent = "Copy"; }, 1800);
    });
  }

  /* ---------- init ---------- */
  function init() {
    renderMath(); buildRefractor(); buildSourceBar(); buildMatchDemo();
    buildResults(); buildHeadline(); buildCherry(); buildGrpoTables();
    buildGallery(); buildReveal(); buildCounters(); buildNav(); buildCopy();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
