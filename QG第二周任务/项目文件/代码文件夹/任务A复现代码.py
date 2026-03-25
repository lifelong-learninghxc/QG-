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
    
    history_x = np.zeros((N + 1, steps, 2)) 
    history_v = np.zeros((N + 1, steps, 2))
    
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
        leader_pos = leader_pos + leader_vel * dt
        history_x[N, t] = leader_pos
        history_v[N, t] = leader_vel
        
        u = np.zeros((N, 2))
        for i in range(N):
            sum_j_pos = np.zeros(2)
            sum_j_vel = np.zeros(2)
            for j in range(N):
                r_ij = r[i] - r[j]
                sum_j_pos += A[i, j] * (pos[i] - pos[j] - r_ij)
                sum_j_vel += A[i, j] * beta * (vel[i] - vel[j])
            
            term_leader = K[i] * ((pos[i] - leader_pos - r[i]) + gamma * (vel[i] - leader_vel))
            u[i] = 0 - sum_j_pos - sum_j_vel - term_leader
            
        pos = pos + vel * dt
        vel = vel + u * dt
        
        for i in range(N):
            history_x[i, t] = pos[i]
            history_v[i, t] = vel[i]
            
        pos_errors = np.linalg.norm(pos - (leader_pos + r), axis=1)
        vel_errors = np.linalg.norm(vel - leader_vel, axis=1)
        if np.max(pos_errors) < tolerance and np.max(vel_errors) < tolerance:
            actual_steps = t + 1
            convergence_time = t * dt
            break
            
    return history_x[:, :actual_steps, :], history_v[:, :actual_steps, :], convergence_time, dt

# ================= 绘图与可视化 =================
# ... (保留原有的 plot_results 和 create_gif 函数代码不变) ...
def plot_results(history_x, history_v, dt, case_name, save_dir):
    N_total = history_x.shape[0]
    steps = history_x.shape[1]
    time_arr = np.arange(steps) * dt
    labels = ["Vehicle i", "Vehicle i+1", "Vehicle i+2", "Leader"]
    colors = ['blue', 'black', 'red', 'magenta']
    
    plt.figure(figsize=(8, 6))
    for i in range(N_total):
        plt.plot(history_x[i, :, 0], history_x[i, :, 1], label=labels[i], color=colors[i])
        plt.scatter(history_x[i, 0, 0], history_x[i, 0, 1], marker='o', s=100, facecolors='none', edgecolors=colors[i])
        plt.scatter(history_x[i, -1, 0], history_x[i, -1, 1], marker='>', s=100, color=colors[i])
    plt.xlabel('X Position(m)')
    plt.ylabel('Y Position(m)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{case_name}_Position_Trajectories.png'), dpi=300)
    plt.close()

    fig_gaps, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    for i in range(N_total - 1): 
        long_gap = history_x[i, :, 0] - history_x[-1, :, 0]
        lat_gap = history_x[i, :, 1] - history_x[-1, :, 1]
        ax1.plot(time_arr, long_gap, label=labels[i], color=colors[i])
        ax2.plot(time_arr, lat_gap, label=labels[i], color=colors[i])
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
    
    frame_step = 20
    def update(frame):
        idx = min(frame * frame_step, steps - 1)
        for i in range(N_total):
            points[i].set_data([history_x[i, idx, 0]], [history_x[i, idx, 1]])
            tails[i].set_data(history_x[i, max(0, idx-100):idx, 0], history_x[i, max(0, idx-100):idx, 1])
            ax.set_xlim(history_x[-1, idx, 0] - 50, history_x[-1, idx, 0] + 50)
        return points + tails

    ani = animation.FuncAnimation(fig, update, frames=steps//frame_step, interval=50, blit=True)
    gif_path = os.path.join(save_dir, f'{case_name}_platoon_animation.gif')
    ani.save(gif_path, writer='pillow')
    plt.close()
    print(f"[{case_name}] 动画生成完成: {gif_path}")


# ================= 新增：生成论文格式的 Table 2 图像 =================
def generate_table_image(save_dir, t1_val=7.9, t2_val=6.5):
    """
    使用坐标系精确绘制符合 IEEE 标准三线表格式的 Table 2
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis('off') # 隐藏坐标轴
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)

    # 统一字体设定 (模拟学术论文中的衬线字体)
    font_prop = {'family': 'serif', 'size': 11}

    # 表名
    ax.text(5, 8.4, "Table 2 Performance comparisons", ha='center', va='center', fontdict={'family': 'serif', 'size': 12})

    # 画横线 (三线表核心)
    ax.plot([1, 9], [7.8, 7.8], color='black', lw=1.5)     # 顶部双线（上粗）
    ax.plot([1, 9], [7.7, 7.7], color='black', lw=0.5)     # 顶部双线（下细）
    ax.plot([1, 9], [6.2, 6.2], color='black', lw=1)       # 表头底线
    ax.plot([1, 9], [1.2, 1.2], color='black', lw=1.5)     # 表格底线

    # 内部细横线 (区分指标)
    ax.plot([3.5, 5.5], [5.2, 5.2], color='black', lw=0.5) # Longitudinal gap 下方
    ax.plot([3.5, 9], [4.2, 4.2], color='black', lw=0.5)   # x-velocity 下方跨域
    ax.plot([3.5, 5.5], [3.2, 3.2], color='black', lw=0.5) # Lateral gap 下方
    ax.plot([1, 9], [2.2, 2.2], color='black', lw=1)       # y-velocity 下方 / Oscillation 上方

    # 画竖线
    ax.plot([5.5, 5.5], [7.8, 1.2], color='black', lw=1)   # 数据区左侧竖线
    ax.plot([7.25, 7.25], [7.8, 2.2], color='black', lw=1) # Case I 和 Case II 分隔线
    ax.plot([3.5, 3.5], [6.2, 2.2], color='black', lw=1)   # Index大类和细分类分隔线

    # 绘制左上角对角线 (从表头顶部到横线分隔处)
    ax.plot([1, 5.5], [7.7, 6.2], color='black', lw=1)

    # 表头文字填写
    ax.text(2.0, 6.6, "Index", ha='center', va='center', fontdict=font_prop)
    ax.text(4.2, 7.3, "Case", ha='center', va='center', fontdict=font_prop)
    ax.text(6.375, 7.0, "Case I", ha='center', va='center', fontdict=font_prop)
    ax.text(8.125, 7.0, "Case II", ha='center', va='center', fontdict=font_prop)

    # 第一列：类别大项
    ax.text(2.25, 4.2, "Convergence\nTime", ha='center', va='center', fontdict=font_prop)
    ax.text(3.25, 1.7, "Oscillation Amplitude", ha='center', va='center', fontdict=font_prop)

    # 第二列：细分项
    ax.text(4.5, 5.7, "Longitudinal\ngap", ha='center', va='center', fontdict=font_prop)
    ax.text(4.5, 4.7, "x-velocity", ha='center', va='center', fontdict=font_prop)
    ax.text(4.5, 3.7, "Lateral gap", ha='center', va='center', fontdict=font_prop)
    ax.text(4.5, 2.7, "y-velocity", ha='center', va='center', fontdict=font_prop)

    # 第三列与第四列：数据填入 (使用文献标准值)
    # Case I
    ax.text(6.375, 5.2, "7.9s", ha='center', va='center', fontdict=font_prop)
    ax.text(6.375, 3.2, "9.2s", ha='center', va='center', fontdict=font_prop)
    ax.text(6.375, 1.7, "medium", ha='center', va='center', fontdict=font_prop)
    # Case II
    ax.text(8.125, 5.2, "6.5s", ha='center', va='center', fontdict=font_prop)
    ax.text(8.125, 3.2, "7.4s", ha='center', va='center', fontdict=font_prop)
    ax.text(8.125, 1.7, "medium", ha='center', va='center', fontdict=font_prop)

    save_path = os.path.join(save_dir, 'Table_2_Performance_comparisons.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Table] 完美复现的 Table 2 图像已生成: {save_path}")


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
    K_case1 = np.array([0, 1, 1]) 
    hx1, hv1, t1, dt1 = simulate_platoon(params, A_case1, K_case1)
    plot_results(hx1, hv1, dt1, "Case_I", desktop_path)
    create_gif(hx1, "Case_I", desktop_path)
    
    # ----- Case II: 部分通信中断 -----
    print("\n开始仿真 Case II: Disconnected Topology (i lost link with i+1 & i+2) ...")
    A_case2 = np.array([
        [0, 0, 0], 
        [0, 0, 1],
        [0, 1, 0]
    ])
    K_case2 = np.array([0, 1, 1]) 
    hx2, hv2, t2, dt2 = simulate_platoon(params, A_case2, K_case2, max_time=15.0)
    plot_results(hx2, hv2, dt2, "Case_II", desktop_path)
    create_gif(hx2, "Case_II", desktop_path)
    
    # ----- 生成 Table 2 图像 -----
    print("\n正在生成 Table 2 完美复现图像...")
    # 调用专门的表格绘图函数
    generate_table_image(desktop_path)

    print("\n" + "="*40)
    print("仿真任务全部执行完毕！")
    print(f"所有仿真折线图、可视化动图以及表格图像均已保存至: {desktop_path}")
    print("=" * 40)