import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json

# ================= 配置与参数设置 =================
# 在桌面创建保存结果的文件夹
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "2026QG_Platoon_Results")
if not os.path.exists(desktop_path):
    os.makedirs(desktop_path)

# 控制增益 (文献设定)
beta = 1.0
gamma = 1.0

# 默认初始状态 (根据 Table 1)
# 车辆顺序: Veh i (idx 0), Veh i+1 (idx 1), Veh i+2 (idx 2)
default_params = {
    "num_followers": 3,
    "leader": {"pos": [20.0, 50.0], "vel": [6.0, 0.0]},
    "followers": [
        {"pos": [6.0, 60.0], "vel": [10.0, 5.0], "desired": [-15.0, 0.0]},    # Veh i
        {"pos": [10.0, 40.0], "vel": [8.0, 4.0], "desired": [-10.0, 0.0]},     # Veh i+1
        {"pos": [16.0, 70.0], "vel": [9.0, 3.0], "desired": [-5.0, 0.0]}       # Veh i+2
    ]
}

# ================= 数据输入模块 =================
def load_config():
    print("选择输入方式: [1] 使用文献 Table 1 默认参数  [2] 从 config.json 加载  [3] 手动输入")
    choice = input("请输入选择 (默认1): ").strip()
    if choice == '2':
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"读取 config.json 失败 ({e})，使用默认参数。")
            return default_params
    elif choice == '3':
        try:
            num = int(input("请输入跟随车辆数量: "))
            params = {"num_followers": num, "followers": []}
            lx, ly = map(float, input("请输入 Leader 初始位置 (x y): ").split())
            lvx, lvy = map(float, input("请输入 Leader 初始速度 (vx vy): ").split())
            params["leader"] = {"pos": [lx, ly], "vel": [lvx, lvy]}
            for i in range(num):
                print(f"--- 配置车辆 {i+1} ---")
                x, y = map(float, input(f"车辆 {i+1} 初始位置 (x y): ").split())
                vx, vy = map(float, input(f"车辆 {i+1} 初始速度 (vx vy): ").split())
                dx, dy = map(float, input(f"车辆 {i+1} 期望相对位置 (dx dy): ").split())
                params["followers"].append({"pos": [x, y], "vel": [vx, vy], "desired": [dx, dy]})
            return params
        except Exception as e:
            print(f"输入错误 ({e})，使用默认参数。")
            return default_params
    return default_params

# ================= 动力学仿真核心 =================
def simulate_platoon(params, A, K, max_time=20.0, dt=0.01, tolerance=0.1):
    N = params["num_followers"]
    steps = int(max_time / dt)
    
    # 历史记录
    history_x = np.zeros((N + 1, steps, 2)) # 0..N-1: followers, N: leader
    history_v = np.zeros((N + 1, steps, 2))
    
    # 初始化
    leader_pos = np.array(params["leader"]["pos"])
    leader_vel = np.array(params["leader"]["vel"])
    history_x[N, 0] = leader_pos
    history_v[N, 0] = leader_vel
    
    pos = np.zeros((N, 2))
    vel = np.zeros((N, 2))
    r = np.zeros((N, 2))
    for i in range(N):
        pos[i] = np.array(params["followers"][i]["pos"])
        vel[i] = np.array(params["followers"][i]["vel"])
        r[i] = np.array(params["followers"][i]["desired"])
        history_x[i, 0] = pos[i]
        history_v[i, 0] = vel[i]

    actual_steps = steps
    convergence_time = max_time

    for t in range(1, steps):
        # Leader 更新 (匀速运动)
        leader_pos = leader_pos + leader_vel * dt
        history_x[N, t] = leader_pos
        history_v[N, t] = leader_vel
        
        # 计算各车的控制输入 u_i
        u = np.zeros((N, 2))
        for i in range(N):
            sum_j_pos = np.zeros(2)
            sum_j_vel = np.zeros(2)
            for j in range(N):
                r_ij = r[i] - r[j]
                sum_j_pos += A[i, j] * (pos[i] - pos[j] - r_ij)
                sum_j_vel += A[i, j] * beta * (vel[i] - vel[j])
            
            term_leader = K[i] * ((pos[i] - leader_pos - r[i]) + gamma * (vel[i] - leader_vel))
            
            # 由于 Leader 加速度为 0，即 v_dot_L = 0
            u[i] = 0 - sum_j_pos - sum_j_vel - term_leader
            
        # 欧拉法更新状态
        pos = pos + vel * dt
        vel = vel + u * dt
        
        for i in range(N):
            history_x[i, t] = pos[i]
            history_v[i, t] = vel[i]
            
        # 自适应结束条件：检查位置误差和速度误差是否都在容忍度内
        pos_errors = np.linalg.norm(pos - (leader_pos + r), axis=1)
        vel_errors = np.linalg.norm(vel - leader_vel, axis=1)
        if np.max(pos_errors) < tolerance and np.max(vel_errors) < tolerance:
            # 持续保持该条件一段时间才认为完全稳定，这里简化为立即判定
            actual_steps = t + 1
            convergence_time = t * dt
            break
            
    return history_x[:, :actual_steps, :], history_v[:, :actual_steps, :], convergence_time, dt

# ================= 绘图与可视化 =================
def plot_results(history_x, history_v, dt, case_name, save_dir):
    N_total = history_x.shape[0]
    steps = history_x.shape[1]
    time_arr = np.arange(steps) * dt
    labels = ["Vehicle i", "Vehicle i+1", "Vehicle i+2", "Leader"]
    colors = ['blue', 'black', 'red', 'magenta']
    
    # Fig: Position Trajectories (X vs Y)
    plt.figure(figsize=(8, 6))
    for i in range(N_total):
        plt.plot(history_x[i, :, 0], history_x[i, :, 1], label=labels[i], color=colors[i])
        # 画个箭头表示方向
        plt.scatter(history_x[i, 0, 0], history_x[i, 0, 1], marker='o', s=100, facecolors='none', edgecolors=colors[i])
        plt.scatter(history_x[i, -1, 0], history_x[i, -1, 1], marker='>', s=100, color=colors[i])
    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{case_name}_Position_Trajectories.png'), dpi=300)
    plt.close()

    # Fig: Gaps (Longitudinal and Lateral)
    fig_gaps, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    for i in range(N_total - 1): # Followers
        long_gap = history_x[i, :, 0] - history_x[-1, :, 0]
        lat_gap = history_x[i, :, 1] - history_x[-1, :, 1]
        ax1.plot(time_arr, long_gap, label=labels[i], color=colors[i])
        ax2.plot(time_arr, lat_gap, label=labels[i], color=colors[i])
    # Leader relative to itself is 0
    ax1.plot(time_arr, np.zeros_like(time_arr), label=labels[-1], color=colors[-1])
    ax2.plot(time_arr, np.zeros_like(time_arr), label=labels[-1], color=colors[-1])
    
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('Longitudinal Gap(m)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Lateral Gap(m)')
    ax2.legend()
    ax2.grid(True)
    plt.savefig(os.path.join(save_dir, f'{case_name}_Gap_Trajectories.png'), dpi=300)
    plt.close()

    # Fig: Velocities
    fig_vel, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    for i in range(N_total):
        ax1.plot(time_arr, history_v[i, :, 0], label=labels[i], color=colors[i])
        ax2.plot(time_arr, history_v[i, :, 1], label=labels[i], color=colors[i])
    ax1.set_xlabel('Time(s)')
    ax1.set_ylabel('X-Velocity(m/s)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Y-Velocity(m/s)')
    ax2.legend()
    ax2.grid(True)
    plt.savefig(os.path.join(save_dir, f'{case_name}_Velocity_Trajectories.png'), dpi=300)
    plt.close()

def create_gif(history_x, case_name, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    N_total = history_x.shape[0]
    steps = history_x.shape[1]
    
    ax.set_xlim(0, 150)
    ax.set_ylim(20, 80)
    ax.set_xlabel("X Position(m)")
    ax.set_ylabel("Y Position(m)")
    ax.grid(True)
    
    colors = ['blue', 'black', 'red', 'magenta']
    labels = ["Veh i", "Veh i+1", "Veh i+2", "Leader"]
    
    points = []
    tails = []
    for i in range(N_total):
        p, = ax.plot([], [], 'o', color=colors[i], markersize=10, label=labels[i])
        t, = ax.plot([], [], '-', color=colors[i], alpha=0.5)
        points.append(p)
        tails.append(t)
    ax.legend(loc="lower right")
    
    # 抽帧以加快生成速度 (例如每 20 个 dt 取一帧)
    frame_step = 20
    
    def update(frame):
        idx = min(frame * frame_step, steps - 1)
        for i in range(N_total):
            points[i].set_data([history_x[i, idx, 0]], [history_x[i, idx, 1]])
            tails[i].set_data(history_x[i, max(0, idx-100):idx, 0], history_x[i, max(0, idx-100):idx, 1])
            # 自适应相机跟随 Leader
            ax.set_xlim(history_x[-1, idx, 0] - 50, history_x[-1, idx, 0] + 50)
        return points + tails

    ani = animation.FuncAnimation(fig, update, frames=steps//frame_step, interval=50, blit=True)
    gif_path = os.path.join(save_dir, f'{case_name}_platoon_animation.gif')
    ani.save(gif_path, writer='pillow')
    plt.close()
    print(f"[{case_name}] 动画生成完成: {gif_path}")

# ================= 主函数 =================
if __name__ == "__main__":
    params = load_config()
    print(f"数据保存路径为: {desktop_path}\n")

    # ----- Case I: 全连通拓扑 -----
    print("开始仿真 Case I: Fully Connected Topology ...")
    A_case1 = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    K_case1 = np.array([0, 1, 1]) # Leader 与 i+1(idx 1) 和 i+2(idx 2) 通信
    
    hx1, hv1, t1, dt1 = simulate_platoon(params, A_case1, K_case1)
    plot_results(hx1, hv1, dt1, "Case_I", desktop_path)
    create_gif(hx1, "Case_I", desktop_path)
    
    # ----- Case II: 部分通信中断 -----
    print("\n开始仿真 Case II: Disconnected Topology (i lost link with i+1 & i+2) ...")
    A_case2 = np.array([
        [0, 0, 0],  # Veh i 掉线
        [0, 0, 1],
        [0, 1, 0]
    ])
    K_case2 = np.array([0, 1, 1]) 
    
    # 修改容忍度或放宽最大时间，因为 Veh i 掉队无法收敛
    hx2, hv2, t2, dt2 = simulate_platoon(params, A_case2, K_case2, max_time=15.0)
    plot_results(hx2, hv2, dt2, "Case_II", desktop_path)
    create_gif(hx2, "Case_II", desktop_path)
    
    # ----- 打印复现的 Table 2 -----
    print("\n" + "="*40)
    print("Table 2 Performance comparisons (Reproduced)")
    print(f"{'Index':<15} | {'Case I':<10} | {'Case II':<10}")
    print("-" * 40)
    # 此处取值主要展示自适应捕捉的时间，若是掉队未收敛，取仿真最大时间
    print(f"{'Convergence Time':<15} | {t1:.1f}s        | {t2:.1f}s (Partial)")
    print("=" * 40)
    print(f"所有图像及可视化动图已保存至: {desktop_path}")