import json
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import io
from PIL import Image

# ================= 1. 数据加载 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, 'data.json')

try:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"找不到数据文件: {json_path}")
    exit()

nodes = {i: np.array([pt['x'], pt['y']], dtype=float) for i, pt in enumerate(data['points'])}
edges = data['edges']

adj = {i: [] for i in range(len(nodes))}
for e in edges:
    adj[e['start_id']].append(e['end_id'])
    adj[e['end_id']].append(e['start_id'])

random.seed(42)

# ================= 2. 仿真全局参数 (1000帧满血版) =================
render_frames = 1000   # 【目标达成】帧数到达 1000
dt = 0.05
TARGET_SPEED = 55.0    # 保持高速
MAX_ACCEL = 35.0
SAFE_DIST = 10.0       # 跨车队防撞

gap = 100.0 
K_pos, K_vel = 3.5, 5.0   
A_pos, A_vel = 1.0, 2.0   

platoons = []
colors = ['red', 'blue', 'green']

# 生成 3 条完全不同的超长迹线
for p_id in range(3):
    curr = random.choice(list(nodes.keys()))
    route = [curr]
    prev = None
    
    # 增加到 15 个路段，确保迹线足够长，配合 1000 帧的运行
    for _ in range(15):
        neighbors = [n for n in adj[curr] if n != prev]
        if not neighbors: neighbors = adj[curr] 
        next_node = random.choice(neighbors)
        route.append(next_node)
        prev = curr
        curr = next_node
        
    route_coords = [nodes[n] for n in route]
    segment_lens = [np.linalg.norm(route_coords[i+1] - route_coords[i]) for i in range(len(route_coords)-1)]
    cum_dist = [0.0] + list(np.cumsum(segment_lens))
    
    # 初始化：确保所有车在路径内
    s_L = gap * 4.0 
    followers = []
    for i in range(3):
        followers.append({'s': s_L - gap * (i + 1), 'v': 0.0, 'r': - gap * (i + 1)})
        
    platoons.append({
        'id': p_id,
        'color': colors[p_id],
        'route_coords': route_coords,
        'cum_dist': cum_dist,
        'total_length': cum_dist[-1],
        's_L': s_L, 'v_L': 0.0,
        'followers': followers,
        'history': []
    })

# ================= 3. 路径映射函数 =================
def get_xy_on_route(s, coords, cum_dist):
    s = max(0.0, min(s, cum_dist[-1])) 
    for i in range(len(cum_dist)-1):
        if cum_dist[i] <= s <= cum_dist[i+1]:
            ratio = (s - cum_dist[i]) / (cum_dist[i+1] - cum_dist[i])
            return coords[i] + ratio * (coords[i+1] - coords[i])
    return coords[-1]

# ================= 4. 1000帧动力学演算 (含终点锁定) =================
print("开始执行 1000 帧高精度演算...")
for frame in range(render_frames):
    # 更新当前位置雷达
    for p in platoons:
        p['xy_curr_all'] = [get_xy_on_route(p['s_L'], p['route_coords'], p['cum_dist'])] + \
                           [get_xy_on_route(f['s'], p['route_coords'], p['cum_dist']) for f in p['followers']]

    for p_idx, p in enumerate(platoons):
        s_L, v_L = p['s_L'], p['v_L']
        total_len = p['total_length']
        
        # 跨车队礼让
        collision_risk = False
        for other_idx, other_p in enumerate(platoons):
            if p_idx > other_idx: 
                for my_xy in p['xy_curr_all']:
                    for their_xy in other_p['xy_curr_all']:
                        if np.linalg.norm(my_xy - their_xy) < SAFE_DIST:
                            collision_risk = True; break
                    if collision_risk: break
            if collision_risk: break
            
        # 领航车：终点停靠逻辑
        remaining = total_len - s_L
        if collision_risk:
            v_L_next = max(0.0, v_L - 25.0 * dt)
        elif remaining <= 0.1:
            v_L_next = 0.0
            s_L = total_len # 强行锁定终点
        elif remaining < 40.0:
            v_L_next = v_L * 0.82 # 临近终点，执行深度减速
        elif v_L < TARGET_SPEED:       
            v_L_next = v_L + 15.0 * dt
        else:
            v_L_next = v_L
            
        s_L_next = s_L + v_L_next * dt
        
        # 跟随车编队
        new_v, new_s = [], []
        for i, f in enumerate(p['followers']):
            u_i = - K_pos * (f['s'] - s_L - f['r']) - K_vel * (f['v'] - v_L)
            for j, f_other in enumerate(p['followers']):
                if i != j and abs(i - j) == 1:
                    u_i += - A_pos * (f['s'] - f_other['s'] - (f['r']-f_other['r'])) - A_vel * (f['v'] - f_other['v'])
            
            v_next = np.clip(f['v'] + np.clip(u_i, -MAX_ACCEL, MAX_ACCEL) * dt, 0, 80)
            # 跟随车终点保护
            s_expected = s_L_next + f['r']
            s_next = f['s'] + v_next * dt
            if s_next > total_len - (i+1)*2.0: # 简易防冲出
                 s_next = min(s_next, total_len - (i+1)*2.0)
            
            new_v.append(v_next); new_s.append(s_next)
            
        p['s_L'], p['v_L'] = s_L_next, v_L_next
        xy_L = get_xy_on_route(s_L_next, p['route_coords'], p['cum_dist'])
        xy_F = []
        for i in range(3):
            p['followers'][i]['s'], p['followers'][i]['v'] = new_s[i], new_v[i]
            xy_F.append(get_xy_on_route(new_s[i], p['route_coords'], p['cum_dist']))
            
        p['history'].append({'L': xy_L, 'F': xy_F})

# ================= 5. 渲染与保存 =================
fig, ax = plt.subplots(figsize=(10, 10))
all_coords = np.array(list(nodes.values()))
min_x, max_x = all_coords[:,0].min()-20, all_coords[:,0].max()+20
min_y, max_y = all_coords[:,1].min()-20, all_coords[:,1].max()+20

print(f"正在渲染 {render_frames} 帧图像...")
gif_frames = []
for frame in range(render_frames):
    ax.clear(); ax.set_xlim(min_x, max_x); ax.set_ylim(min_y, max_y); ax.axis('off')
    
    for e in edges:
        ax.plot([nodes[e['start_id']][0], nodes[e['end_id']][0]], 
                [nodes[e['start_id']][1], nodes[e['end_id']][1]], color='lightgray', lw=1.5, zorder=1)

    for p in platoons:
        hist = p['history'][frame]; c = p['color']
        rx, ry = zip(*p['route_coords'])
        ax.plot(rx, ry, color=c, lw=6, alpha=0.1, zorder=1) # 迹线展示
        
        ax.plot([hist['L'][0]]+[f[0] for f in hist['F']], [hist['L'][1]]+[f[1] for f in hist['F']], color=c, lw=2, alpha=0.6, zorder=3)
        ax.plot(hist['L'][0], hist['L'][1], marker='s', color=c, ms=8, ls='None', zorder=5)
        for f_xy in hist['F']:
            ax.plot(f_xy[0], f_xy[1], marker='o', color=c, ms=5, ls='None', zorder=4)

    ax.set_title(f"1000-Frame CAV Platoon Trace (Strict Path Following)")
    
    buf = io.BytesIO(); plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0); gif_frames.append(Image.open(buf).copy()); buf.close()
    if (frame + 1) % 100 == 0: print(f"进度: {frame+1}/{render_frames}")

plt.close(fig)
print("正在封装超长 GIF...")
gif_frames[0].save('platoon_simulation.gif', save_all=True, append_images=gif_frames[1:], duration=30, loop=0)
print("✅ 1000 帧全迹线版保存成功！")