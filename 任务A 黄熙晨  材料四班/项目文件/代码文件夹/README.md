# 车辆智能编队控制复现项目 (CAV Platoon Control Simulation)

## 项目简介
本项目是对《Feedback-based platoon control for connected autonomous vehicles under different communication network topologies》论文核心实验的完整代码复现（即**任务A**）。项目模拟了网联自动驾驶车辆（CAVs）在不同通信网络拓扑下的编队控制过程，成功复现了论文中的位置轨迹、间距误差、速度变化以及 Table 2 的对比数据。

## 目录结构说明
根据项目实际归档，文件结构如下：
```text
📦 项目根目录
 ┣ 📂 笔记
 ┃ ┗ 📜 任务A文献阅读任务.md           # 归纳整理的文献阅读与推导笔记
 ┣ 📂 矩阵推导
 ┃ ┣ 🖼️ 1.jpg                        # 论文公式的矩阵表达手工推导过程图
 ┃ ┗ 🖼️ 2.jpg 
 ┃ ┗ 🖼️ 2.jpg  
 ┗ 📂 实验图复现
   ┣ 📜 任务A复现代代码.py            # 核心仿真、绘图及动画生成代码
   ┗ 📂 复现结果                      # 运行代码后生成的图像、表格和动图保存位置
     ┣ 🖼️ Case_I_Gap_Trajectories.png
     ┣ 🖼️ Case_I_Position_Trajectories.png
     ┣ 🖼️ Case_I_Velocity_Trajectories.png
     ┣ 🖼️ Case_I_platoon_animation.gif
     ┣ 🖼️ Case_II_Gap_Trajectories.png
     ┣ 🖼️ Case_II_Position_Trajectories.png
     ┣ 🖼️ Case_II_velocity_Trajectories.png
     ┣ 🖼️ Case_II_platoon_animation.gif
     ┗ 🖼️ Table_2_Performance_comparisons.png