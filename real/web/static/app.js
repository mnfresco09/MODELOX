(() => {
  const state = {
    bootstrap: null,
    selectedTf: null,
    controlsBuilt: false,
    applyingSelectors: false,
    mainChart: null,
    candleSeries: null,
    overlaySeries: [],
    subCharts: [],
  };

  const el = {
    marketMode: document.getElementById("marketMode"),
    statAsset: document.getElementById("statAsset"),
    statTf: document.getElementById("statTf"),
    statPrice: document.getElementById("statPrice"),
    statEquity: document.getElementById("statEquity"),
    assetSelect: document.getElementById("assetSelect"),
    strategySelect: document.getElementById("strategySelect"),
    tfButtons: document.getElementById("tfButtons"),
    applySelectors: document.getElementById("applySelectors"),
    chartContainer: document.getElementById("chartContainer"),
    subPanels: document.getElementById("subPanels"),
    positionsBody: document.getElementById("positionsBody"),
    balAvailable: document.getElementById("balAvailable"),
    balEquity: document.getElementById("balEquity"),
    orderSide: document.getElementById("orderSide"),
    orderType: document.getElementById("orderType"),
    orderMargin: document.getElementById("orderMargin"),
    orderLeverage: document.getElementById("orderLeverage"),
    estQty: document.getElementById("estQty"),
    estNotional: document.getElementById("estNotional"),
    btnBuy: document.getElementById("btnBuy"),
    btnSell: document.getElementById("btnSell"),
    orderStatus: document.getElementById("orderStatus"),
    systemStatus: document.getElementById("systemStatus"),
    apiError: document.getElementById("apiError"),
  };

  function fmt(v, d = 2) {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    return n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  function setStatus(msg, isError = false) {
    el.orderStatus.textContent = msg || "";
    el.orderStatus.style.color = isError ? "#c15a5a" : "#a6b1bc";
  }

  function setSystemStatus(msg, isError = false) {
    if (el.systemStatus) el.systemStatus.textContent = msg || "";
    if (el.apiError) {
      el.apiError.textContent = isError ? (msg || "") : "";
    }
  }

  function chartBaseOptions() {
    return {
      layout: { background: { type: "solid", color: "#101923" }, textColor: "#a6b1bc" },
      grid: {
        vertLines: { color: "rgba(95,107,120,0.22)" },
        horzLines: { color: "rgba(95,107,120,0.22)" },
      },
      rightPriceScale: { borderColor: "#5f6b78" },
      timeScale: { borderColor: "#5f6b78", rightOffset: 5, barSpacing: 10, minBarSpacing: 5 },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    };
  }

  function initMainChart() {
    if (state.mainChart) return;
    state.mainChart = LightweightCharts.createChart(el.chartContainer, {
      ...chartBaseOptions(),
      width: el.chartContainer.clientWidth,
      height: el.chartContainer.clientHeight,
    });
    state.candleSeries = state.mainChart.addCandlestickSeries({
      upColor: "#4ea46e",
      downColor: "#c15a5a",
      borderUpColor: "#4ea46e",
      borderDownColor: "#c15a5a",
      wickUpColor: "#4ea46e",
      wickDownColor: "#c15a5a",
    });

    state.mainChart.timeScale().applyOptions({ rightOffset: 6, barSpacing: 10 });

    window.addEventListener("resize", () => {
      if (state.mainChart) {
        state.mainChart.resize(el.chartContainer.clientWidth, el.chartContainer.clientHeight);
      }
      state.subCharts.forEach((it) => {
        it.chart.resize(it.container.clientWidth, it.container.clientHeight);
      });
    });
  }

  function clearOverlays() {
    if (!state.mainChart) return;
    state.overlaySeries.forEach((s) => {
      try { state.mainChart.removeSeries(s); } catch (e) {}
    });
    state.overlaySeries = [];
  }

  function clearSubPanels() {
    state.subCharts.forEach((it) => {
      try { it.chart.remove(); } catch (e) {}
    });
    state.subCharts = [];
    el.subPanels.innerHTML = "";
  }

  function renderSubPanels(subPanels) {
    clearSubPanels();
    if (!Array.isArray(subPanels) || subPanels.length === 0) return;

    subPanels.forEach((panel) => {
      const wrap = document.createElement("div");
      wrap.className = "sub-panel";
      const title = document.createElement("div");
      title.className = "sub-panel-title";
      title.textContent = panel.title || "INDICATOR";
      wrap.appendChild(title);
      el.subPanels.appendChild(wrap);

      const chart = LightweightCharts.createChart(wrap, {
        ...chartBaseOptions(),
        width: wrap.clientWidth,
        height: wrap.clientHeight,
      });

      (panel.series || []).forEach((ser) => {
        const s = ser.type === "histogram"
          ? chart.addHistogramSeries({ color: ser.color || "#8a949b" })
          : chart.addLineSeries({ color: ser.color || "#8a949b", lineWidth: 1 });
        s.setData((ser.data || []).map((d) => ({ time: d.time, value: d.value })));

        if (ser.bounds && ser.type !== "histogram") {
          ["upper", "lower", "mid", "hi", "lo"].forEach((k) => {
            if (ser.bounds[k] === undefined || ser.bounds[k] === null) return;
            const b = chart.addLineSeries({
              color: k === "upper" || k === "hi" ? "rgba(193,90,90,0.6)" : k === "lower" || k === "lo" ? "rgba(78,164,110,0.6)" : "rgba(209,181,124,0.5)",
              lineWidth: 1,
              lineStyle: LightweightCharts.LineStyle.Dashed,
            });
            b.setData((ser.data || []).map((d) => ({ time: d.time, value: ser.bounds[k] })));
          });
        }
      });

      chart.timeScale().fitContent();
      state.subCharts.push({ chart, container: wrap });
    });
  }

  function renderChart(chartPayload) {
    initMainChart();
    clearOverlays();

    const candles = (chartPayload && chartPayload.candles) || [];
    if (!candles.length) {
      setSystemStatus("Sin velas para el contexto actual. Esperando datos...", true);
      return;
    }
    state.candleSeries.setData(candles);

    const overlays = chartPayload?.indicators?.overlays || [];
    overlays.forEach((ov) => {
      const s = state.mainChart.addLineSeries({
        color: ov.color || "#4f86b6",
        lineWidth: 1,
      });
      s.setData((ov.data || []).map((d) => ({ time: d.time, value: d.value })));
      state.overlaySeries.push(s);
    });

    state.mainChart.timeScale().fitContent();
    renderSubPanels(chartPayload?.indicators?.sub_panels || []);
  }

  function renderPositions(rows) {
    const html = (rows || []).map((r) => {
      const pnlCls = Number(r.pnl) >= 0 ? "pnl-pos" : "pnl-neg";
      return `
      <tr>
        <td>${r.symbol || "-"}</td>
        <td>${r.side || "-"}</td>
        <td>${r.strategy || "-"}</td>
        <td>${fmt(r.entry_price)}</td>
        <td>${fmt(r.current_price)}</td>
        <td class="${pnlCls}">${fmt(r.pnl)}</td>
        <td><button class="close-btn" data-close-key="${r.close_key}">Cerrar</button></td>
      </tr>`;
    }).join("");

    el.positionsBody.innerHTML = html || `<tr><td colspan="7">Sin posiciones activas</td></tr>`;

    Array.from(el.positionsBody.querySelectorAll(".close-btn")).forEach((b) => {
      b.addEventListener("click", async () => {
        const key = b.getAttribute("data-close-key");
        if (!key) return;
        const res = await fetch("/api/order/close", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ close_key: key }),
        });
        const data = await res.json();
        setStatus(data.ok ? "Posición cerrada" : (data.error || "Error cerrando posición"), !data.ok);
      });
    });
  }

  function renderSelectors(payload) {
    const selected = payload.selected || {};
    state.selectedTf = selected.timeframe;

    if (!state.controlsBuilt) {
      el.assetSelect.innerHTML = (payload.assets || []).map((a) => `<option value="${a}">${a}</option>`).join("");

      el.strategySelect.innerHTML = (payload.strategies || []).map((s) => `<option value="${s.id}">${s.name}</option>`).join("");

      el.tfButtons.innerHTML = (payload.timeframes || []).map((tf) => {
        return `<button class="tf-btn" data-tf="${tf}">${tf}</button>`;
      }).join("");

      Array.from(el.tfButtons.querySelectorAll(".tf-btn")).forEach((btn) => {
        btn.addEventListener("click", () => {
          state.selectedTf = btn.getAttribute("data-tf");
          Array.from(el.tfButtons.querySelectorAll(".tf-btn")).forEach((x) => x.classList.remove("active"));
          btn.classList.add("active");
        });
      });
      state.controlsBuilt = true;
    }

    el.assetSelect.value = selected.asset || "";
    if (selected.strategy_id !== undefined && selected.strategy_id !== null) {
      el.strategySelect.value = String(selected.strategy_id);
    }
    Array.from(el.tfButtons.querySelectorAll(".tf-btn")).forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-tf") === selected.timeframe);
    });
  }

  function renderHeader(payload) {
    const s = payload.selected || {};
    const balance = payload.balance || {};
    el.marketMode.textContent = (balance.currency || "USDT") === "VST" ? "DEMO" : "REAL";
    el.statAsset.textContent = s.asset || "-";
    el.statTf.textContent = s.timeframe || "-";
    el.statPrice.textContent = fmt(payload.last_price, 4);
    el.statEquity.textContent = `${balance.currency || "USDT"} ${fmt(balance.equity)}`;
    el.balAvailable.textContent = `${balance.currency || "USDT"} ${fmt(balance.available)}`;
    el.balEquity.textContent = `${balance.currency || "USDT"} ${fmt(balance.equity)}`;

    const lev = Number(payload.selected_leverage || 1);
    el.orderLeverage.value = `${lev}x`;

    const margin = Number(el.orderMargin.value || 0);
    const price = Number(payload.last_price || 0);
    const qty = margin > 0 && price > 0 ? (margin * lev) / price : 0;
    const notional = margin > 0 ? margin * lev : 0;
    el.estQty.textContent = qty > 0 ? fmt(qty, 6) : "-";
    el.estNotional.textContent = notional > 0 ? fmt(notional, 2) : "-";
  }

  function renderAll(payload) {
    if (!payload || !payload.ok) {
      setSystemStatus(payload?.error || "Snapshot inválido", true);
      return;
    }
    state.bootstrap = payload;
    renderHeader(payload);
    renderSelectors(payload);
    renderChart(payload.chart || {});
    renderPositions(payload.positions || []);
    if (payload.error) {
      setSystemStatus(payload.error, true);
    } else {
      setSystemStatus(`Stream activo | ctx ${payload.context_version || 1} | ${new Date().toLocaleTimeString()}`);
    }
  }

  async function applySelectors() {
    if (state.applyingSelectors) return;
    state.applyingSelectors = true;
    el.applySelectors.disabled = true;
    const asset = el.assetSelect.value;
    const strategyId = Number(el.strategySelect.value || 0);
    const timeframe = state.selectedTf || (state.bootstrap?.selected?.timeframe) || "1m";
    const res = await fetch("/api/selectors", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asset, timeframe, strategy_id: strategyId }),
    });
    const data = await res.json();
    setStatus(data.ok ? "Selectores actualizados" : (data.error || "Error actualizando selectores"), !data.ok);
    state.applyingSelectors = false;
    el.applySelectors.disabled = false;
  }

  async function sendOrder(side) {
    const margin = Number(el.orderMargin.value || 0);
    const orderType = el.orderType.value || "MARKET";
    if (!(margin > 0)) {
      setStatus("Margen inválido", true);
      return;
    }
    const res = await fetch("/api/order/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ side, margin, order_type: orderType }),
    });
    const data = await res.json();
    if (data.ok) {
      const q = data?.filled?.quantity;
      setStatus(`Orden ${side} enviada | qty ${fmt(q, 6)}`);
    } else {
      setStatus(data.error || "Error enviando orden", true);
    }
  }

  function startSSE() {
    const es = new EventSource("/api/stream");
    es.addEventListener("snapshot", (evt) => {
      try {
        const payload = JSON.parse(evt.data);
        renderAll(payload);
      } catch (e) {}
    });
    es.onerror = () => {
      setSystemStatus("SSE desconectado. Reintentando...", true);
      es.close();
      setTimeout(startSSE, 1500);
    };
  }

  async function boot() {
    const res = await fetch("/api/bootstrap");
    const payload = await res.json();
    renderAll(payload);

    el.applySelectors.addEventListener("click", applySelectors);
    el.btnBuy.addEventListener("click", () => sendOrder("LONG"));
    el.btnSell.addEventListener("click", () => sendOrder("SHORT"));
    el.orderMargin.addEventListener("input", () => {
      if (!state.bootstrap) return;
      renderHeader(state.bootstrap);
    });

    startSSE();
  }

  boot().catch(() => setStatus("No se pudo iniciar dashboard", true));
})();
