from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

import numpy as np


ArrayLike = np.ndarray


@dataclass
class VectorTransformer:
    """
    n 维向量坐标系变换与几何分析工具。

    约定：
    - 基底矩阵 B 的列向量是各坐标轴在世界坐标中的表示。
    - 世界向量 v = B @ c（c 为该基底下坐标列向量）。
    - 批量输入时，向量按“每行一个向量”组织，形状为 (m, n)。
    """

    B_cur: ArrayLike
    B_target: ArrayLike
    check_cond_threshold: float | None = None

    def __post_init__(self) -> None:
        self.B_cur = self._as_2d_square(self.B_cur, name="B_cur")
        self.B_target = self._as_2d_square(self.B_target, name="B_target")

        if self.B_cur.shape != self.B_target.shape:
            raise ValueError(
                f"B_cur 和 B_target 形状必须一致，收到 {self.B_cur.shape} 与 {self.B_target.shape}"
            )

        self.n = self.B_cur.shape[0]
        self._check_invertible(self.B_cur, name="B_cur")
        self._check_invertible(self.B_target, name="B_target")

        # 可选：病态矩阵提醒（不阻断，仅提示）
        if self.check_cond_threshold is not None:
            cond_cur = np.linalg.cond(self.B_cur)
            cond_tgt = np.linalg.cond(self.B_target)
            if cond_cur > self.check_cond_threshold or cond_tgt > self.check_cond_threshold:
                print(
                    f"[Warning] 检测到较大条件数: cond(B_cur)={cond_cur:.3e}, "
                    f"cond(B_target)={cond_tgt:.3e}"
                )

    @staticmethod
    def _as_2d_square(B: ArrayLike, name: str) -> np.ndarray:
        B = np.asarray(B, dtype=float)
        if B.ndim != 2:
            raise ValueError(f"{name} 必须是二维矩阵，收到 ndim={B.ndim}")
        r, c = B.shape
        if r != c:
            raise ValueError(f"{name} 必须是方阵，收到形状 {B.shape}")
        return B

    @staticmethod
    def _check_invertible(B: np.ndarray, name: str) -> None:
        n = B.shape[0]
        rank = np.linalg.matrix_rank(B)
        if rank < n:
            raise np.linalg.LinAlgError(
                f"{name} 不可逆：rank={rank} < n={n}"
            )

    def _as_batch_vectors(self, c: ArrayLike) -> Tuple[np.ndarray, bool]:
        c = np.asarray(c, dtype=float)
        if c.ndim == 1:
            if c.shape[0] != self.n:
                raise ValueError(
                    f"向量维度不匹配：期望 {self.n}，收到 {c.shape[0]}"
                )
            return c.reshape(1, self.n), True
        if c.ndim == 2:
            if c.shape[1] != self.n:
                raise ValueError(
                    f"批量向量维度不匹配：期望第二维 {self.n}，收到 {c.shape[1]}"
                )
            return c, False
        raise ValueError(f"向量输入必须为一维或二维，收到 ndim={c.ndim}")

    def _to_output_shape(self, x: np.ndarray, was_single: bool) -> np.ndarray:
        return x[0] if was_single else x

    def to_world(self, c_cur: ArrayLike) -> np.ndarray:
        """
        当前坐标 -> 世界坐标
        支持单向量 (n,) 或批量 (m, n)。
        """
        C, was_single = self._as_batch_vectors(c_cur)
        # 每行向量 c_i 对应 world_i = (B_cur @ c_i_col)_T = c_i_row @ B_cur.T
        V = C @ self.B_cur.T
        return self._to_output_shape(V, was_single)

    def transform_coordinates(self, c_cur: ArrayLike) -> np.ndarray:
        """
        当前坐标 c_cur -> 目标坐标 c_target
        核心：先 v = B_cur @ c_cur，再解 B_target @ c_target = v。
        严禁 inv，使用 np.linalg.solve。
        """
        C, was_single = self._as_batch_vectors(c_cur)

        # 批量世界坐标 (m, n)
        V = C @ self.B_cur.T

        # 批量求解：B_target @ X = V.T，得到 X.T 即每行一个目标坐标
        C_target = np.linalg.solve(self.B_target, V.T).T
        return self._to_output_shape(C_target, was_single)

    def projection_lengths(self, c_cur: ArrayLike) -> np.ndarray:
        """
        计算向量在目标坐标系各轴方向上的投影长度（标量投影）。
        对每个轴 b_j：proj_j = <v, b_j> / ||b_j||。
        """
        C, was_single = self._as_batch_vectors(c_cur)
        V = C @ self.B_cur.T  # (m, n)

        axes = self.B_target  # 列向量为轴
        axis_norms = np.linalg.norm(axes, axis=0)  # (n,)
        if np.any(axis_norms == 0):
            raise np.linalg.LinAlgError("目标基底存在零长度轴向量，无法计算投影。")

        # dots[i, j] = <v_i, b_j>
        dots = V @ axes
        proj = dots / axis_norms[None, :]
        return self._to_output_shape(proj, was_single)

    def angles_with_target_axes(self, c_cur: ArrayLike) -> np.ndarray:
        """
        计算向量与目标坐标系各基向量夹角（度）。
        - 通过 np.clip 防止浮点误差导致 arccos 域错误。
        - 零向量夹角未定义，返回 np.nan。
        """
        C, was_single = self._as_batch_vectors(c_cur)
        V = C @ self.B_cur.T  # (m, n)

        axes = self.B_target
        axis_norms = np.linalg.norm(axes, axis=0)  # (n,)
        if np.any(axis_norms == 0):
            raise np.linalg.LinAlgError("目标基底存在零长度轴向量，无法计算夹角。")

        # <v_i, b_j>
        dots = V @ axes  # (m, n)
        v_norms = np.linalg.norm(V, axis=1, keepdims=True)  # (m, 1)

        angles = np.full_like(dots, fill_value=np.nan, dtype=float)
        valid = (v_norms[:, 0] > 0.0)
        if np.any(valid):
            cos_vals = dots[valid] / (v_norms[valid] * axis_norms[None, :])
            cos_vals = np.clip(cos_vals, -1.0, 1.0)
            angles[valid] = np.degrees(np.arccos(cos_vals))

        return self._to_output_shape(angles, was_single)

    def volume_scale(self) -> float:
        """
        目标基底相对于单位空间的体积缩放倍率 |det(B_target)|。
        """
        return float(abs(np.linalg.det(self.B_target)))

    def condition_number(self, which: Literal["cur", "target"] = "target") -> float:
        """
        可选的病态性检查指标。数值越大，问题越病态。
        """
        B = self.B_cur if which == "cur" else self.B_target
        return float(np.linalg.cond(B))

    def analyze(self, c_cur: ArrayLike) -> Dict[str, Any]:
        """
        一次性返回主要分析结果。
        """
        return {
            "target_coordinates": self.transform_coordinates(c_cur),
            "projection_lengths": self.projection_lengths(c_cur),
            "angles_deg": self.angles_with_target_axes(c_cur),
            "volume_scale": self.volume_scale(),
        }


if __name__ == "__main__":
    # 示例：从标准基转换到 3D 旋转 + 缩放基底
    theta = np.deg2rad(35.0)
    Rz = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ],
        dtype=float,
    )
    S = np.diag([2.0, 1.5, 0.5])
    B_target = Rz @ S

    B_cur = np.eye(3)
    transformer = VectorTransformer(B_cur=B_cur, B_target=B_target, check_cond_threshold=1e8)

    # 单向量
    c1 = np.array([1.0, 2.0, 3.0])
    result1 = transformer.analyze(c1)
    print("single vector result:")
    for k, v in result1.items():
        print(f"{k}: {v}")

    # 批量向量（每行一个）
    C_batch = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],  # 零向量，夹角将返回 nan
        ],
        dtype=float,
    )
    result_batch = transformer.analyze(C_batch)
    print("\nbatch result:")
    for k, v in result_batch.items():
        print(f"{k}: {v}")