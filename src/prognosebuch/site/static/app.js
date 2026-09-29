// Tabs (day and window switches) and the chart tooltip. The page works without this script.
(function () {
  document.querySelectorAll('[role="tablist"]').forEach(function (list) {
    var tabs = Array.prototype.slice.call(list.querySelectorAll('[role="tab"]'));
    function select(tab) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        var panel = document.getElementById(t.getAttribute("aria-controls"));
        if (on) panel.removeAttribute("data-hidden"); else panel.setAttribute("data-hidden", "");
      });
      var title = document.getElementById("day-title");
      if (title && tab.dataset.title) title.textContent = tab.dataset.title;
    }
    tabs.forEach(function (tab, i) {
      tab.tabIndex = tab.getAttribute("aria-selected") === "true" ? 0 : -1;
      tab.addEventListener("click", function () { select(tab); });
      tab.addEventListener("keydown", function (e) {
        var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        if (!d) return;
        var next = tabs[(i + d + tabs.length) % tabs.length];
        next.focus(); select(next);
      });
    });
  });

  var de = document.documentElement.lang === "de";
  function fmt(x) {
    if (x === null || x === undefined) return "–";
    var s = x.toFixed(1).replace("-", "−");
    return de ? s.replace(".", ",") : s;
  }
  document.querySelectorAll("figure.ladder[data-tooltip]").forEach(function (fig) {
    var rows = JSON.parse(fig.dataset.tooltip);
    var tip = fig.querySelector(".tip");
    var last = null;
    function show(rect, clientX) {
      var r = rows[+rect.dataset.i];
      if (!r) return;
      if (last) last.classList.remove("on");
      rect.classList.add("on"); last = rect;
      var html = "<b>" + r[0] + "</b><br>" + (de ? "Median " : "Median ") + fmt(r[2]) +
        "<br>P10–P90 " + fmt(r[1]) + " – " + fmt(r[3]);
      if (r[4] !== null) html += "<br>" + (de ? "Tatsächlich " : "Actual ") + fmt(r[4]);
      tip.innerHTML = html + " <span>€/MWh</span>";
      tip.hidden = false;
      var box = fig.getBoundingClientRect();
      var x = clientX - box.left;
      var w = tip.offsetWidth;
      tip.style.left = Math.max(0, Math.min(box.width - w, x + 12)) + "px";
      tip.style.top = "0px";
    }
    function onPointer(e) {
      var el = document.elementFromPoint(e.clientX, e.clientY);
      if (el && el.classList.contains("c-hit")) show(el, e.clientX);
    }
    fig.addEventListener("pointermove", onPointer);
    fig.addEventListener("pointerdown", onPointer);
    fig.addEventListener("pointerleave", function () {
      tip.hidden = true; if (last) last.classList.remove("on"); last = null;
    });
  });
})();
