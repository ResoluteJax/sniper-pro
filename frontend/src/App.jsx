import React, { useState, useEffect, useRef } from 'react';
import { createChart, ColorType, CandlestickSeries } from 'lightweight-charts';
import { Activity, TrendingUp, TrendingDown, DollarSign, Clock, Trophy, Zap, Brain } from 'lucide-react';

const Dashboard = () => {
  const [data, setData] = useState(null);
  const chartContainerRef = useRef(null);
  const chartInstance = useRef(null);
  const seriesInstance = useRef(null);
  const ws = useRef(null);

  useEffect(() => {
    // Se estiver rodando local, usa localhost. Se estiver na nuvem, usa a variável de ambiente.
const socketUrl = import.meta.env.VITE_WS_URL || "ws://127.0.0.1:8000/ws";
ws.current = new WebSocket(socketUrl);
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      setData(message);
      if (seriesInstance.current && message.last_candle) {
        try { seriesInstance.current.update(message.last_candle); } catch (e) {}
      }
    };
    return () => { if (ws.current) ws.current.close(); };
  }, []);

  useEffect(() => {
    if (!chartContainerRef.current || !data || !data.chart_data) return;
    if (chartInstance.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: { background: { type: ColorType.Solid, color: 'transparent' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#334155' }, horzLines: { color: '#334155' } },
      width: chartContainerRef.current.clientWidth,
      height: 400,
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: '#334155' },
      rightPriceScale: { borderColor: '#334155' },
    });

    const newSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e', downColor: '#ef4444', borderUpColor: '#22c55e', borderDownColor: '#ef4444', wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    });

    try {
      const sorted = [...data.chart_data].sort((a, b) => a.time - b.time);
      const unique = sorted.filter((v, i, a) => a.findIndex(t => (t.time === v.time)) === i);
      newSeries.setData(unique);
      chart.timeScale().fitContent();
    } catch (e) {}

    chartInstance.current = chart;
    seriesInstance.current = newSeries;

    const handleResize = () => chart.applyOptions({ width: chartContainerRef.current.clientWidth });
    window.addEventListener('resize', handleResize);
    return () => { window.removeEventListener('resize', handleResize); chart.remove(); chartInstance.current = null; };
  }, [data]);

  if (!data) return <div className="min-h-screen bg-slate-900 flex items-center justify-center text-white"><Activity className="animate-spin mr-2"/> Carregando Sniper Auto-Evolutivo...</div>;

  const isLong = data.status.includes("COMPRA");
  const isShort = data.status.includes("VENDA");
  const isTraining = data.training?.is_training;

  return (
    <div className="min-h-screen bg-slate-900 p-6 font-sans text-slate-100">
      
      {/* HEADER */}
      <header className="flex justify-between items-center mb-6 border-b border-slate-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2 text-white">
            <Activity className="text-blue-500" /> SNIPER PRO <span className="text-xs bg-purple-600 px-2 py-1 rounded text-white font-mono">V5 AUTO-LEARN</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1 flex items-center gap-2">
            <Brain size={14} className={isTraining ? "text-yellow-400 animate-pulse" : "text-slate-500"}/>
            {isTraining ? "IA APRENDENDO COM O MERCADO..." : `Cérebro Estável • Geração ${data.training?.generation}`}
          </p>
        </div>
        <div className="text-right text-yellow-400 font-mono text-xl flex items-center gap-2">
           <Clock size={20}/> {data.uptime}
        </div>
      </header>

      {/* CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
         <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
            <div className="text-slate-400 text-sm mb-1">Saldo Total</div>
            <div className="text-3xl font-bold font-mono">${data.balance.toFixed(2)}</div>
         </div>
         <div className="bg-slate-800 p-4 rounded-xl border border-slate-700">
            <div className="text-slate-400 text-sm mb-1">PnL Aberto</div>
            <div className={`text-3xl font-bold font-mono ${data.pnl_open >= 0 ? 'text-green-400' : 'text-red-400'}`}>
               {data.pnl_open >= 0 ? "+" : ""}{data.pnl_open.toFixed(2)}
            </div>
         </div>
         <div className={`bg-slate-800 p-4 rounded-xl border-l-4 border-slate-700 ${isLong ? 'border-l-green-500' : isShort ? 'border-l-red-500' : ''}`}>
            <div className="text-slate-400 text-sm mb-1">Status Atual</div>
            <div className="text-2xl font-bold">{data.status}</div>
         </div>
         <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 relative overflow-hidden">
            <div className="text-slate-400 text-sm mb-1">Evolução Neural</div>
            <div className="text-2xl font-bold font-mono flex items-center gap-2">
              <Zap className={isTraining ? "text-yellow-400" : "text-slate-600"} />
              {isTraining ? "TREINANDO" : "PRONTO"}
            </div>
            <div className="text-xs text-slate-500 mt-1">Última: {data.training?.last_evolution}</div>
         </div>
      </div>

      {/* ÁREA PRINCIPAL */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg flex flex-col">
           <h2 className="text-lg font-bold mb-4 text-slate-300">Gráfico Real-Time</h2>
           <div ref={chartContainerRef} className="w-full relative" style={{ height: '450px' }}>
             {(!data.chart_data || data.chart_data.length === 0) && (
               <div className="absolute inset-0 flex items-center justify-center text-slate-500">Carregando dados...</div>
             )}
           </div>
        </div>

        <div className="bg-slate-800 p-4 rounded-xl border border-slate-700 h-[530px] flex flex-col">
           <h2 className="text-lg font-bold mb-4 text-slate-300">Histórico de Operações</h2>
           <div className="flex-1 overflow-auto">
             <table className="w-full text-sm text-slate-400">
               <thead className="text-xs text-slate-200 uppercase bg-slate-700 sticky top-0">
                 <tr>
                   <th className="px-2 py-3 text-left">Hora</th>
                   <th className="px-2 py-3 text-left">Tipo</th>
                   <th className="px-2 py-3 text-right">PnL ($)</th>
                   <th className="px-2 py-3 text-right">Motivo</th>
                 </tr>
               </thead>
               <tbody>
                 {data.trades_history.length === 0 ? (
                   <tr><td colSpan="4" className="text-center py-10 text-slate-600">Aguardando operações...</td></tr>
                 ) : (
                   data.trades_history.map((t, i) => (
                     <tr key={i} className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors">
                       <td className="px-2 py-3 text-slate-500 font-mono text-xs">{t.exit_time}</td>
                       <td className="px-2 py-3">
                         <span className={`px-2 py-1 rounded text-xs font-bold ${t.type === 'COMPRA' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                           {t.type}
                         </span>
                       </td>
                       <td className={`px-2 py-3 text-right font-mono font-bold ${t.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                         {t.pnl >= 0 ? "+" : ""}{t.pnl.toFixed(2)}
                       </td>
                       <td className="px-2 py-3 text-right text-xs text-slate-400">{t.result}</td>
                     </tr>
                   ))
                 )}
               </tbody>
             </table>
           </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;