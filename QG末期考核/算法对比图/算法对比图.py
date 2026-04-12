import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# [模块 1] 环境与排版配置
# ==========================================
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "2026末期考核_隐私预算分布分析")
os.makedirs(desktop_path, exist_ok=True)

print("[*] 启动 Task 2: 隐私预算 (ε) 分布对比 (修复文本遮挡版)...")

# IEEE Transactions 高精度排版规范
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'legend.fontsize': 11,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.linewidth': 1.5,
    'font.family': 'sans-serif'
})

# ==========================================
# [模块 2] 数学推导与真实前缀树模拟引擎
# ==========================================
h = 5
levels = np.arange(1, h + 1)
total_eps = 1.0

# --- 1. SeqPT: 绝对均分 (水平线) ---
seq_mean = np.array([total_eps / h] * h)
seq_std = np.array([0.0] * h)

# --- 2. Li's Algorithm: 对数递增 (左低右高) ---
sigma = 1.0
li_log = np.log(levels + sigma)
li_mean = li_log / np.sum(li_log) * total_eps
li_std = np.array([0.0] * h)

# --- 3. Our Algorithm: 拉格朗日动态优化 (左高右低，带方差) ---
np.random.seed(42)

class SimNode:
    def __init__(self, level):
        self.level = level
        self.children = []
        self.prob = 0.0
        self.eps_raw = 0.0

root = SimNode(0)
current_level = [root]
branching_factors = [5, 4, 3, 2, 1] 
all_nodes = {1:[], 2:[], 3:[], 4:[], 5:[]}

for lv, bf in enumerate(branching_factors, 1):
    next_level = []
    for node in current_level:
        num_children = int(np.random.exponential(scale=bf))
        num_children = max(1, min(num_children, bf * 2)) 
        for _ in range(num_children):
            child = SimNode(lv)
            node.children.append(child)
            next_level.append(child)
            all_nodes[lv].append(child)
    current_level = next_level

total_patterns = sum(len(nodes) for nodes in all_nodes.values())
def calc_prob(node):
    if not node.children:
        node.prob = 1.0 / total_patterns
    else:
        node.prob = (1.0 / total_patterns) + sum(calc_prob(c) for c in node.children)
    return node.prob
calc_prob(root)

def allocate_raw(node):
    if not node.children:
        node.eps_raw = total_eps
        return node.eps_raw
    sum_term = sum(c.prob / (allocate_raw(c)**3) for c in node.children)
    if sum_term > 0:
        node.eps_raw = len(node.children) * (node.prob / sum_term)**(1/3)
    else:
        node.eps_raw = total_eps
    return node.eps_raw
allocate_raw(root)

our_raw_means = []
our_raw_stds = []
for lv in levels:
    eps_values = [n.eps_raw for n in all_nodes[lv]]
    our_raw_means.append(np.mean(eps_values))
    our_raw_stds.append(np.std(eps_values))

our_raw_means = np.array(our_raw_means)
our_raw_stds = np.array(our_raw_stds)
scaling_factor = total_eps / np.sum(our_raw_means) 

our_mean = our_raw_means * scaling_factor
our_std = our_raw_stds * scaling_factor


# ==========================================
# [模块 3] 绘制分布曲线 (零遮挡完美排版)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7))
x = levels
offset = 0.05 

# 1. 绘制 SeqPT
ax.errorbar(x - offset, seq_mean, yerr=seq_std, fmt='-s', color='black', 
            label='SeqPT', capsize=6, markeredgewidth=2, markerfacecolor='none', zorder=3)

# 2. 绘制 Li's Algorithm
ax.errorbar(x, li_mean, yerr=li_std, fmt='-x', color='#8A2BE2', 
            label="Li's Algorithm", capsize=6, markeredgewidth=2, zorder=3)

# 3. 绘制 本文算法 (带显著的误差棒)
ax.errorbar(x + offset, our_mean, yerr=our_std, fmt='-D', color='#DC143C', 
            label='Our Algorithm (Mean ± Std)', capsize=6, markeredgewidth=2, markerfacecolor='none', zorder=3)

# 【核心修复】：动态自适应计算 Y 轴上限，大幅度拔高天花板留出空白区
max_y_value = max(np.max(seq_mean), np.max(li_mean), np.max(our_mean + our_std))
ax.set_ylim(-0.03, max_y_value * 1.50) # 顶部留出 50% 的绝对安全空间！

# 美化图表
ax.set_title('Privacy Budget (ε) Distribution Across Prefix Tree Levels', pad=15, fontweight='bold')
ax.set_xlabel('Prefix Tree Level (Top to Bottom)')
ax.set_ylabel('Allocated Privacy Budget ε')
ax.set_xticks(x)

ax.grid(True, linestyle='--', color='#cccccc', alpha=0.7, zorder=0)
ax.legend(loc='upper right', frameon=True, edgecolor='black')

# 【核心修复】：将文本框移动到"正上方 (Top Center)" 的巨大空白区，实现零遮挡
annotation_text = (
    "Theoretical Analysis:\n"
    "1. SeqPT: Uniform baseline (Constant, Std=0)\n"
    "2. Li's Alg: Heuristic protects leaves (Log-Increase, Std=0)\n"
    "3. Our Alg: Lagrangian optimal protects root bottlenecks\n"
    "   (Decreasing, Std>0 due to dynamic $p_i$ adaptation)"
)
# x=0.45 稍微偏左居中，y=0.96 紧贴图表内部顶部
ax.text(0.42, 0.96, annotation_text, 
        transform=ax.transAxes, fontsize=12, va='top', ha='center', 
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#f9f9f9', edgecolor='black', alpha=0.9), zorder=5)

plt.tight_layout()
save_path = os.path.join(desktop_path, "Task2_Budget_Distribution_NoOverlap.png")
plt.savefig(save_path, bbox_inches='tight')
plt.close()

print(f"✅ 隐私预算分布图 (零遮挡排版版) 渲染完成！")
print(f"✅ 文本已转移至图表正上方真空区，完美解决遮挡问题。输出路径: {save_path}")