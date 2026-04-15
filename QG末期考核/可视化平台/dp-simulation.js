import React, { useState, useMemo } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine
} from 'recharts';

const DPDashboard = () => {
  // === 1. 状态管理：定义核心超参数 ===
  const [epsilon, setEpsilon] = useState(1.0);
  const [h, setH] = useState(8);
  const [k, setK] = useState(1.5);
  const [b, setB] = useState(1.0);
  const [N, setN] = useState(1000);

  // === 2. 核心物理引擎：根据参数实时推演数据 ===
  // 图1：结构效用增益 (维数灾难与样本免疫推演)
  const chart1Data = useMemo(() => {
    const data = [];
    for (let x = 4; x <= 14; x++) {
      const ourError = Math.max(0.001, (0.015 * x + 0.15 * Math.exp(-0.4 * Math.pow(x - 12, 2))) / epsilon + (k * b) / Math.sqrt(N));
      const safePathError = Math.max(0.005, (0.02 * x) / epsilon + (k * b) / Math.sqrt(N * 0.8));
      data.push({ height: x, Our: ourError.toFixed(4), SafePath: safePathError.toFixed(4) });
    }
    return data;
  }, [epsilon, k, b, N]);

  // 图2：层级分配特性 (拉格朗日最优分配 vs 几何衰减 vs 均分)
  const chart2Data = useMemo(() => {
    const data = [];
    let sumLi = 0;
    let sumOur = 0;
    for (let j = 1; j <= h; j++) {
      sumLi += Math.pow(0.6, j);
      sumOur += Math.pow(j, 1.5);
    }
    for (let i = 1; i <= h; i++) {
      const seqPT = epsilon / h;
      const li = epsilon * Math.pow(0.6, i) / sumLi;
      const our = epsilon * Math.pow(i, 1.5) / sumOur;
      data.push({ depth: i, SeqPT: seqPT.toFixed(3), Li: li.toFixed(3), Our: our.toFixed(3) });
    }
    return data;
  }, [epsilon, h]);

  // 图3：统计分布特征 (CDF 收敛速度推演)
  const chart3Data = useMemo(() => {
    const data = [];
    const lambdaOur = 15 * epsilon * Math.sqrt(N / 1000) / (k * 0.5 + 1);
    const lambdaBase = lambdaOur * 0.5;
    for (let x = 0.01; x <= 0.20; x += 0.01) {
      const ourCDF = Math.min(1, 1 - Math.exp(-lambdaOur * x));
      const baseCDF = Math.min(1, 1 - Math.exp(-lambdaBase * x));
      data.push({ errorBound: x.toFixed(2), Our: ourCDF.toFixed(4), Baseline: baseCDF.toFixed(4) });
    }
    return data;
  }, [epsilon, k, N]);

  // === 3. UI 渲染 ===
  const sliderStyle = "w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer";
  const labelStyle = "block text-sm font-semibold text-gray-700 mb-1";

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif', backgroundColor: '#f8fafc', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto', backgroundColor: 'white', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
        
        {/* 标题区 */}
        <h1 style={{ textAlign: 'center', fontSize: '24px', fontWeight: 'bold', color: '#1e293b', marginBottom: '20px' }}>
          差分隐私前缀树 (DP-Tree) 预算-效用-结构 交互仿真平台
        </h1>

        {/* 控制面板区 */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', marginBottom: '30px', padding: '20px', backgroundColor: '#f1f5f9', borderRadius: '8px' }}>
          <div style={{ flex: '1 1 18%' }}>
            <label className={labelStyle}>隐私预算 (ε): {epsilon.toFixed(1)}</label>
            <input type="range" min="0.1" max="3.0" step="0.1" value={epsilon} onChange={e => setEpsilon(parseFloat(e.target.value))} style={{ width: '100%' }} />
          </div>
          <div style={{ flex: '1 1 18%' }}>
            <label className={labelStyle}>前缀树高度 (h): {h}</label>
            <input type="range" min="4" max="14" step="1" value={h} onChange={e => setH(parseInt(e.target.value))} style={{ width: '100%' }} />
          </div>
          <div style={{ flex: '1 1 18%' }}>
            <label className={labelStyle}>阈值乘数 (k): {k.toFixed(1)}</label>
            <input type="range" min="0.5" max="3.0" step="0.1" value={k} onChange={e => setK(parseFloat(e.target.value))} style={{ width: '100%' }} />
          </div>
          <div style={{ flex: '1 1 18%' }}>
            <label className={labelStyle}>阈值偏置 (b): {b.toFixed(1)}</label>
            <input type="range" min="0" max="5" step="0.5" value={b} onChange={e => setB(parseFloat(e.target.value))} style={{ width: '100%' }} />
          </div>
          <div style={{ flex: '1 1 18%' }}>
            <label className={labelStyle}>查询样本量 (N): {N}</label>
            <input type="range" min="100" max="10000" step="100" value={N} onChange={e => setN(parseInt(e.target.value))} style={{ width: '100%' }} />
          </div>
        </div>

        {/* 图表展示区 */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '30px' }}>
          
          {/* Chart 1: 结构效用增益 */}
          <div style={{ height: '350px', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '10px' }}>
            <h3 style={{ textAlign: 'center', fontSize: '16px', marginBottom: '10px', color: '#334155' }}>结构效用增益 (相对误差 vs 树高)</h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={chart1Data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="height" label={{ value: '前缀树高度', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <ReferenceLine x={h} stroke="red" strokeDasharray="3 3" label={{ position: 'top', value: '当前树高 h' }} />
                <Line type="monotone" dataKey="Our" stroke="#DC143C" strokeWidth={3} name="Our Algorithm" dot={{ r: 4 }} />
                <Line type="monotone" dataKey="SafePath" stroke="#4169E1" strokeWidth={2} name="SafePath" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Chart 2: 层级分配特性 */}
          <div style={{ height: '350px', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '10px' }}>
            <h3 style={{ textAlign: 'center', fontSize: '16px', marginBottom: '10px', color: '#334155' }}>隐私预算层级分配策略对比</h3>
            <ResponsiveContainer width="100%" height="90%">
              <BarChart data={chart2Data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="depth" label={{ value: '前缀树层级 (Depth)', position: 'insideBottomRight', offset: -5 }} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="SeqPT" fill="#228B22" name="SeqPT (均匀分配)" />
                <Bar dataKey="Li" fill="#8A2BE2" name="Li's Alg (几何衰减)" />
                <Bar dataKey="Our" fill="#DC143C" name="Our Alg (拉格朗日倾斜)" />
              </BarChart>
            </ResponsiveContainer>
          </div>

        </div>

        {/* Chart 3: 统计分布特征 */}
        <div style={{ height: '350px', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '10px' }}>
          <h3 style={{ textAlign: 'center', fontSize: '16px', marginBottom: '10px', color: '#334155' }}>查询结果误差分布 CDF</h3>
          <ResponsiveContainer width="100%" height="90%">
            <LineChart data={chart3Data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="errorBound" label={{ value: '误差容忍度 (Sanity Bound)', position: 'insideBottomRight', offset: -5 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="Our" stroke="#DC143C" strokeWidth={3} name="Our Algorithm" dot={false} />
              <Line type="monotone" dataKey="Baseline" stroke="#4169E1" strokeWidth={2} name="Baseline" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

      </div>
    </div>
  );
};

export default DPDashboard;