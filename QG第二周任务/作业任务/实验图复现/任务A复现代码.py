import numpy as np
import matplotlib.pyplot as plt


# 车辆标签与颜色映射：Leader(粉), i(蓝), i+1(黑), i+2(红)
FOLLOWER_LABELS = ["Vehicle i", "Vehicle i+1", "Vehicle i+2"]
FOLLOWER_COLORS = ["tab:blue", "black", "tab:red"]
LEADER_COLOR = "deeppink"

# 相对Leader的期望纵向间距（m），横向期望为0（即收敛到 y=50 轨道）
DESIRED_LONG_GAP = np.array([15.0, 30.0, 45.0])


def _generate_spread_initial_positions(rng, n_followers, min_dist=8.0, max_trials=5000):
    """随机生成且彼此有最小间距的初始位置。"""
    for _ in range(max_trials):
        cand = np.column_stack(
            [
                rng.uniform(0.0, 18.0, size=n_followers),
                rng.uniform(40.0, 70.0, size=n_followers),
            ]
        )
        ok = True
        for i in range(n_followers):
            for j in range(i + 1, n_followers):
                if np.linalg.norm(cand[i] - cand[j]) < min_dist:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return cand
    return cand


def _first_settling_time(t, signal_abs, threshold, hold_time=1.0):
    """返回首次进入阈值且持续 hold_time 的时刻，未满足则返回 np.nan。"""
    if len(t) < 2:
        return np.nan
    dt = t[1] - t[0]
    hold_n = max(1, int(np.ceil(hold_time / dt)))
    ok = signal_abs <= threshold
    for i in range(0, len(t) - hold_n + 1):
        if np.all(ok[i : i + hold_n]):
            return float(t[i])
    return np.nan


def run_case(case_name, xlim, ylim, t_end, fault_case=False, seed=42):
    """离散时间二阶模型：2s后启动高增益编队控制，目标在10s内收敛。"""
    rng = np.random.default_rng(seed)

    dt = 0.01
    t = np.arange(0.0, t_end + dt, dt)
    n_steps = len(t)
    n_followers = 3

    # Leader: 速度恒定，vy=0，确保速度曲线是水平线
    leader_pos0 = np.array([5.0, 50.0])
    leader_vel_const = np.array([7.0, 0.0])

    # 初始阶段随机分布且保持一定间隔（X<20, Y in [40,70]）
    follower_pos0 = _generate_spread_initial_positions(rng, n_followers, min_dist=8.0)
    follower_vel0 = np.column_stack(
        [
            rng.uniform(4.0, 9.0, size=n_followers),
            rng.uniform(-2.0, 2.0, size=n_followers),
        ]
    )

    leader_pos = np.zeros((n_steps, 2))
    leader_vel = np.zeros((n_steps, 2))
    follower_pos = np.zeros((n_steps, n_followers, 2))
    follower_vel = np.zeros((n_steps, n_followers, 2))

    leader_pos[0] = leader_pos0
    leader_vel[:] = leader_vel_const
    follower_pos[0] = follower_pos0
    follower_vel[0] = follower_vel0

    # 高增益控制，保证2s~10s快速收敛
    kp_pos = 4.0
    kd_vel = 5.0
    a_max = 28.0
    # Case II 复杂度降低：健康车辆可更早启动控制
    case2_kp_boost = 1.0
    case2_kd_boost = 1.0

    fault_triggered = False
    # 失控车开环恒速：故障触发后保持该速度，不再受通信/控制影响。
    fault_open_loop_vel = np.array([10.0, 5.0])

    for k in range(1, n_steps):
        # Leader恒速更新
        leader_pos[k] = leader_pos[k - 1] + leader_vel_const * dt

        for i in range(n_followers):
            p = follower_pos[k - 1, i]
            v = follower_vel[k - 1, i]
            control_start = 2.0
            if fault_case and i != 0:
                control_start = 1.0
            if fault_case and i == 0:
                control_start = 0.8

            # t < 2s: 初始阶段不施加编队反馈，只保留惯性
            if t[k] < control_start:
                acc = np.zeros(2)
            else:
                target_pos = leader_pos[k] + np.array([-DESIRED_LONG_GAP[i], 0.0])
                target_vel = leader_vel_const

                # Case II: Vehicle i (蓝色, i=0) 在极短时间后失控。
                # 触发后切换为开环恒速，不再依赖 Leader 或通信信息。
                if fault_case and i == 0 and (t[k] > 1.2 or p[0] > 13.0 or fault_triggered):
                    fault_triggered = True
                    v_new = fault_open_loop_vel.copy()
                    p_new = p + v_new * dt
                    follower_vel[k, i] = v_new
                    follower_pos[k, i] = p_new
                    continue
                else:
                    if fault_case and i != 0:
                        kp_eff = kp_pos * case2_kp_boost
                        kd_eff = kd_vel * case2_kd_boost
                    else:
                        kp_eff = kp_pos
                        kd_eff = kd_vel
                    acc = kp_eff * (target_pos - p) + kd_eff * (target_vel - v)

            acc = np.clip(acc, -a_max, a_max)
            v_new = v + acc * dt
            p_new = p + v_new * dt

            follower_vel[k, i] = v_new
            follower_pos[k, i] = p_new

    return {
        "case_name": case_name,
        "t": t,
        "leader_pos": leader_pos,
        "leader_vel": leader_vel,
        "follower_pos": follower_pos,
        "follower_vel": follower_vel,
        "xlim": xlim,
        "ylim": ylim,
        "fault_case": fault_case,
    }


def _draw_initial_topology(ax, leader0, followers0):
    """用虚线展示初始通信拓扑。"""
    nodes = [leader0] + [followers0[i] for i in range(3)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    for a, b in edges:
        xa, ya = nodes[a]
        xb, yb = nodes[b]
        ax.plot([xa, xb], [ya, yb], linestyle="--", color="gray", alpha=0.45, linewidth=1.2)


def _draw_vehicle_trajectory_with_arrows(ax, x, y, color, label, linestyle="--"):
    """绘制轨迹，并用较大的箭头指示运动方向，便于区分车辆。"""
    ax.plot(x, y, color=color, linewidth=2.3, linestyle=linestyle, label=label, zorder=2)

    n = len(x)
    arrow_count = 10
    idx = np.linspace(1, n - 2, arrow_count, dtype=int)
    dx = x[idx + 1] - x[idx - 1]
    dy = y[idx + 1] - y[idx - 1]

    ax.quiver(
        x[idx],
        y[idx],
        dx,
        dy,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        width=0.006,
        headwidth=8.5,
        headlength=10.5,
        headaxislength=8.2,
        alpha=0.95,
        zorder=4,
    )


def _draw_follower_spacing_links(ax, fp, t, fault_case):
    """按时间抽样连接跟随车，直观显示车间间隔。"""
    if fault_case:
        link_pairs = [(1, 2)]
    else:
        link_pairs = [(0, 1), (1, 2)]

    n = len(t)
    sample_idx = np.linspace(0, n - 1, 14, dtype=int)
    for k in sample_idx:
        for i, j in link_pairs:
            ax.plot(
                [fp[k, i, 0], fp[k, j, 0]],
                [fp[k, i, 1], fp[k, j, 1]],
                color="#6f6f6f",
                linestyle="-",
                linewidth=1.2,
                alpha=0.35,
                zorder=1,
            )


def plot_case(sim_data, fig_id_start, file_prefix):
    t = sim_data["t"]
    lp = sim_data["leader_pos"]
    lv = sim_data["leader_vel"]
    fp = sim_data["follower_pos"]
    fv = sim_data["follower_vel"]

    # Fig. Position Trajectories
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    _draw_initial_topology(ax, lp[0], fp[0])
    _draw_follower_spacing_links(ax, fp, t, sim_data["fault_case"])

    _draw_vehicle_trajectory_with_arrows(
        ax,
        lp[:, 0],
        lp[:, 1],
        color=LEADER_COLOR,
        label="Leader",
        linestyle="-",
    )
    ax.scatter(lp[0, 0], lp[0, 1], color=LEADER_COLOR, s=120, marker="o", edgecolor="white", linewidths=0.8, zorder=5)
    ax.scatter(lp[-1, 0], lp[-1, 1], color=LEADER_COLOR, s=175, marker=">", edgecolor="white", linewidths=0.9, zorder=6)

    for i in range(3):
        _draw_vehicle_trajectory_with_arrows(
            ax,
            fp[:, i, 0],
            fp[:, i, 1],
            color=FOLLOWER_COLORS[i],
            label=FOLLOWER_LABELS[i],
            linestyle="--",
        )
        ax.scatter(fp[0, i, 0], fp[0, i, 1], color=FOLLOWER_COLORS[i], s=105, marker="o", edgecolor="white", linewidths=0.8, zorder=5)
        ax.scatter(fp[-1, i, 0], fp[-1, i, 1], color=FOLLOWER_COLORS[i], s=165, marker=">", edgecolor="white", linewidths=0.9, zorder=6)

    # 末时刻队列示意：Case I 展示完整队列，Case II 展示健康车辆队列
    if sim_data["fault_case"]:
        final_order = [2, 1]
    else:
        final_order = [2, 1, 0]

    qx = [fp[-1, idx, 0] for idx in final_order] + [lp[-1, 0]]
    qy = [fp[-1, idx, 1] for idx in final_order] + [lp[-1, 1]]
    ax.plot(qx, qy, color="#2f2f2f", linewidth=2.2, linestyle=":", alpha=0.85, label="Final stable queue", zorder=3)
    if sim_data["fault_case"]:
        y_fault_queue = fp[-1, 0, 1]
        ax.hlines(
            y_fault_queue,
            xmin=sim_data["xlim"][0],
            xmax=sim_data["xlim"][1],
            color=FOLLOWER_COLORS[0],
            linestyle=":",
            linewidth=1.4,
            alpha=0.5,
            label="Fault vehicle stable lane",
            zorder=0,
        )

    ax.axhline(50.0, color="gray", linewidth=1.0, linestyle=":", alpha=0.8)
    ax.set_xlim(sim_data["xlim"])
    ax.set_ylim(sim_data["ylim"])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Fig.{fig_id_start} {sim_data['case_name']} - Position Trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{file_prefix}_position.png", dpi=180)

    # Fig. Gap Trajectories (含Leader基准)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    zero = np.zeros_like(t)
    ax1.plot(t, zero, color=LEADER_COLOR, linewidth=2.4, label="Leader baseline")
    ax2.plot(t, zero, color=LEADER_COLOR, linewidth=2.4, label="Leader baseline")

    for i in range(3):
        long_gap_err = (lp[:, 0] - fp[:, i, 0]) - DESIRED_LONG_GAP[i]
        lat_gap = fp[:, i, 1] - lp[:, 1]
        ax1.plot(t, long_gap_err, color=FOLLOWER_COLORS[i], linewidth=2.0, label=f"{FOLLOWER_LABELS[i]} x-gap")
        ax2.plot(t, lat_gap, color=FOLLOWER_COLORS[i], linewidth=2.0, label=f"{FOLLOWER_LABELS[i]} y-gap")

    for ref in [0.0]:
        ax1.axhline(ref, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.8)
        ax2.axhline(ref, color="#1f77b4", linestyle=":", linewidth=1.2, alpha=0.8)
    ax1.axvline(10.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.axvline(10.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    ax1.set_ylabel("Longitudinal gap error (m)")
    ax2.set_ylabel("Lateral gap (m)")
    ax2.set_xlabel("Time (s)")
    ax1.set_title(f"Fig.{fig_id_start + 1} {sim_data['case_name']} - Gap Trajectories")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{file_prefix}_gaps.png", dpi=180)

    # Fig. Velocity Trajectories (含Leader)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    ax1.plot(t, lv[:, 0], color=LEADER_COLOR, linewidth=2.6, label="Leader vx")
    ax2.plot(t, lv[:, 1], color=LEADER_COLOR, linewidth=2.6, label="Leader vy")

    for i in range(3):
        ax1.plot(t, fv[:, i, 0], color=FOLLOWER_COLORS[i], linewidth=2.0, label=f"{FOLLOWER_LABELS[i]} vx")
        ax2.plot(t, fv[:, i, 1], color=FOLLOWER_COLORS[i], linewidth=2.0, label=f"{FOLLOWER_LABELS[i]} vy")

    ax1.axvline(10.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.axvline(10.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_ylabel("vx (m/s)")
    ax2.set_ylabel("vy (m/s)")
    ax2.set_xlabel("Time (s)")
    ax1.set_title(f"Fig.{fig_id_start + 2} {sim_data['case_name']} - Velocity Trajectories")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{file_prefix}_velocity.png", dpi=180)


def summarize_convergence(sim_data):
    """输出每个跟随车的收敛时间（阈值内首次稳定）。"""
    t = sim_data["t"]
    lp = sim_data["leader_pos"]
    lv = sim_data["leader_vel"]
    fp = sim_data["follower_pos"]
    fv = sim_data["follower_vel"]

    print(f"\n[{sim_data['case_name']}] 收敛统计（阈值: |x-gap err|<0.5, |y-gap|<0.5, |vx-vLx|<0.5, |vy-vLy|<0.5）")
    for i in range(3):
        x_err = np.abs((lp[:, 0] - fp[:, i, 0]) - DESIRED_LONG_GAP[i])
        y_err = np.abs(fp[:, i, 1] - lp[:, 1])
        vx_err = np.abs(fv[:, i, 0] - lv[:, 0])
        vy_err = np.abs(fv[:, i, 1] - lv[:, 1])

        cond = (x_err < 0.5) & (y_err < 0.5) & (vx_err < 0.5) & (vy_err < 0.5)
        idx = np.argmax(cond)
        if cond[idx]:
            print(f"  {FOLLOWER_LABELS[i]}: t = {t[idx]:.2f}s")
        else:
            print(f"  {FOLLOWER_LABELS[i]}: 未在仿真窗口内收敛")


def compute_case_metrics(sim_data, failed_index=None):
    """计算 Table2 指标：四项收敛时间 + 横向振荡幅值。"""
    t = sim_data["t"]
    lp = sim_data["leader_pos"]
    lv = sim_data["leader_vel"]
    fp = sim_data["follower_pos"]
    fv = sim_data["follower_vel"]

    if failed_index is None:
        active_idx = [0, 1, 2]
    else:
        active_idx = [i for i in [0, 1, 2] if i != failed_index]

    long_times = []
    vx_times = []
    lat_times = []
    vy_times = []

    lat_peak_to_peak = []
    for i in [0, 1, 2]:
        lat_gap = fp[:, i, 1] - lp[:, 1]
        lat_peak_to_peak.append(float(np.max(lat_gap) - np.min(lat_gap)))

    for i in active_idx:
        long_err = np.abs((lp[:, 0] - fp[:, i, 0]) - DESIRED_LONG_GAP[i])
        vx_err = np.abs(fv[:, i, 0] - lv[:, 0])
        lat_err = np.abs(fp[:, i, 1] - lp[:, 1])
        vy_err = np.abs(fv[:, i, 1] - lv[:, 1])

        long_times.append(_first_settling_time(t, long_err, threshold=0.5, hold_time=1.0))
        vx_times.append(_first_settling_time(t, vx_err, threshold=0.5, hold_time=1.0))
        lat_times.append(_first_settling_time(t, lat_err, threshold=0.5, hold_time=1.0))
        vy_times.append(_first_settling_time(t, vy_err, threshold=0.5, hold_time=1.0))

    def safe_nanmax(values):
        arr = np.array(values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return np.nan
        return float(np.nanmax(arr))

    return {
        "Longitudinal gap": safe_nanmax(long_times),
        "x-velocity": safe_nanmax(vx_times),
        "Lateral gap": safe_nanmax(lat_times),
        "y-velocity": safe_nanmax(vy_times),
        "Oscillation amplitude": float(np.max(lat_peak_to_peak)),
    }


def plot_table2(metrics_case1, metrics_case2, png_path="Table2.png", csv_path="Table2.csv"):
    rows = [
        "Longitudinal gap",
        "x-velocity",
        "Lateral gap",
        "y-velocity",
        "Oscillation amplitude",
    ]

    table_data = []
    for k in rows:
        v1 = metrics_case1[k]
        v2 = metrics_case2[k]
        s1 = "N/A" if np.isnan(v1) else f"{v1:.2f}"
        s2 = "N/A" if np.isnan(v2) else f"{v2:.2f}"
        table_data.append([k, s1, s2])

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.axis("off")
    ax.set_title("Table 2 Performance Comparison", fontsize=16, pad=16)

    tbl = ax.table(
        cellText=table_data,
        colLabels=["Index", "Case I", "Case II"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.25, 1.7)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Index,Case I,Case II\n")
        for row in table_data:
            f.write(f"{row[0]},{row[1]},{row[2]}\n")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8")
    print("=== 开始多智能体编队仿真（10s收敛目标）===")

    # 场景一：理想协同编队
    case1 = run_case(
        case_name="Case I",
        xlim=(0.0, 150.0),
        ylim=(0.0, 80.0),
        t_end=20.0,
        fault_case=False,
        seed=7,
    )
    plot_case(case1, fig_id_start=4, file_prefix="Fig_caseI")
    summarize_convergence(case1)

    # 场景二：Vehicle i 失控，i+1 和 i+2 保持鲁棒收敛
    case2 = run_case(
        case_name="Case II",
        xlim=(0.0, 110.0),
        ylim=(0.0, 120.0),
        t_end=10.0,
        fault_case=True,
        seed=7,
    )
    plot_case(case2, fig_id_start=7, file_prefix="Fig_caseII")
    summarize_convergence(case2)

    metrics_case1 = compute_case_metrics(case1, failed_index=None)
    metrics_case2 = compute_case_metrics(case2, failed_index=0)
    plot_table2(metrics_case1, metrics_case2, png_path="Table2.png", csv_path="Table2.csv")

    print("\n图像已输出：Fig_caseI_*.png, Fig_caseII_*.png, Table2.png")