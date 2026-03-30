"""
Semantic Critic Module
语义审视者模块

This module implements Phase II: Counterfactual Causal Pruning
通过预期-违背范式进行反事实推理，实现幻觉检测和级联剪枝

Key innovations:
1. 计算语义残差（expectation vs. reality）
2. VLM-based causality diagnosis（诊断感知错误的物理原因）
3. 触发认知依赖树的级联删除
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.epistemic_graph import (
    EpistemicNode,
    EpistemicGraph,
    SemanticDistribution,
    NodeType,
    CognitiveDependency
)
from src.eval_utils_gpt_aeqa import call_openai_api, encode_tensor2base64


@dataclass
class FalsificationReport:
    """证伪报告"""
    node_id: str
    semantic_residual: float  # 语义残差值
    is_falsified: bool  # 是否证伪
    physical_cause: Optional[str]  # 物理原因（如"mirror reflection", "glass window"）
    textual_explanation: str  # 文字解释
    cascade_deleted_nodes: List[str]  # 级联删除的节点ID列表
    confidence: float  # 证伪置信度


class SemanticCritic:
    """
    语义审视者
    负责验证Ghost Node的预测，检测幻觉，并触发级联剪枝
    """

    def __init__(self, cfg: dict, epistemic_graph: EpistemicGraph):
        self.cfg = cfg
        self.epistemic_graph = epistemic_graph

        # 配置参数
        self.semantic_residual_threshold = cfg.get("semantic_residual_threshold", 0.5)
        self.enable_vlm_diagnosis = cfg.get("enable_vlm_causality_diagnosis", True)
        self.min_falsification_confidence = cfg.get("min_falsification_confidence", 0.7)

        # 特征提取器（使用CLIP）
        self.clip_model = None
        self.clip_preprocess = None

        # 统计信息
        self.stats = {
            "total_verifications": 0,
            "successful_verifications": 0,
            "falsifications": 0,
            "hallucination_types": {},  # 幻觉类型统计
        }

    def set_clip_model(self, clip_model, clip_preprocess):
        """设置CLIP模型用于特征提取"""
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess

    def verify_ghost_node_arrival(
        self,
        ghost_node: EpistemicNode,
        actual_observation: Dict,
        scene_objects: Dict
    ) -> FalsificationReport:
        """
        验证Ghost Node：当Agent到达预测的Ghost Node时调用

        Args:
            ghost_node: 待验证的Ghost Node
            actual_observation: 实际观测数据
                - rgb_image: RGB图像
                - depth: 深度图
                - detected_objects: 检测到的对象列表
                - semantic_class: 实际观测到的房间类型（如果有）
            scene_objects: 场景对象字典

        Returns:
            FalsificationReport对象
        """

        self.stats["total_verifications"] += 1

        # 步骤1: 计算语义残差
        semantic_residual = self._compute_semantic_residual(
            ghost_node, actual_observation
        )

        # 步骤2: 判断是否证伪
        is_falsified = semantic_residual > self.semantic_residual_threshold

        # 步骤3: 如果证伪，进行因果诊断
        physical_cause = None
        textual_explanation = ""
        cascade_deleted_nodes = []

        if is_falsified:
            # VLM诊断：为什么预测错误？
            if self.enable_vlm_diagnosis:
                physical_cause, textual_explanation = self._diagnose_falsification_cause(
                    ghost_node, actual_observation
                )
            else:
                textual_explanation = "Semantic residual exceeds threshold"

            # 更新幻觉类型统计
            if physical_cause:
                self.stats["hallucination_types"][physical_cause] = (
                    self.stats["hallucination_types"].get(physical_cause, 0) + 1
                )

            # 步骤4: 标记节点为已证伪
            self.epistemic_graph.nodes[ghost_node.node_id].mark_falsified(
                reason=textual_explanation,
                residual=semantic_residual
            )

            # 步骤5: 级联删除依赖节点
            deleted_count = self.epistemic_graph.cascade_delete(ghost_node.node_id)
            cascade_deleted_nodes = self._get_cascade_deleted_node_ids(ghost_node.node_id)

            self.stats["falsifications"] += 1

            print(f"""
[Semantic Critic] FALSIFICATION DETECTED
  Node ID: {ghost_node.node_id}
  Semantic Residual: {semantic_residual:.3f}
  Physical Cause: {physical_cause}
  Explanation: {textual_explanation}
  Cascade Deleted: {len(cascade_deleted_nodes)} nodes
""")

        else:
            # 验证通过：升级为Observed Node
            predicted_class = ghost_node.semantic_dist.categories[
                np.argmax(ghost_node.semantic_dist.probabilities)
            ]
            ghost_node.upgrade_to_observed(
                observed_class=predicted_class,
                confidence=1.0 - semantic_residual
            )

            self.stats["successful_verifications"] += 1
            textual_explanation = f"Verified as {predicted_class}"

            print(f"""
[Semantic Critic] VERIFICATION PASSED
  Node ID: {ghost_node.node_id}
  Predicted: {predicted_class}
  Residual: {semantic_residual:.3f}
""")

        # 生成报告
        report = FalsificationReport(
            node_id=ghost_node.node_id,
            semantic_residual=semantic_residual,
            is_falsified=is_falsified,
            physical_cause=physical_cause,
            textual_explanation=textual_explanation,
            cascade_deleted_nodes=cascade_deleted_nodes,
            confidence=abs(semantic_residual - self.semantic_residual_threshold)
        )

        return report

    def _compute_semantic_residual(
        self,
        ghost_node: EpistemicNode,
        actual_observation: Dict
    ) -> float:
        """
        计算语义残差：预期 vs. 实际观测

        方法1: 基于类别匹配（快速）
        方法2: 基于特征距离（CLIP embedding距离）
        方法3: 基于VLM判断（最准确但慢）
        """

        # 方法1: 类别匹配
        if "semantic_class" in actual_observation:
            actual_class = actual_observation["semantic_class"]
            expected_prob = ghost_node.semantic_dist.get_expected_semantic_score(actual_class)
            residual_category = 1.0 - expected_prob
        else:
            residual_category = 0.5  # 未知

        # 方法2: 特征距离（有CLIP模型）
        if self.clip_model is not None and ghost_node.feature is not None:
            residual_feature = self._compute_feature_residual(
                ghost_node.feature,
                actual_observation.get("rgb_image", None)
            )
        else:
            residual_feature = 0.5

        # 方法3: 对象级别验证
        residual_objects = self._compute_object_residual(
            ghost_node.semantic_dist,
            actual_observation.get("detected_objects", [])
        )

        # 综合残差（加权平均）
        w1 = self.cfg.get("residual_weight_category", 0.5)
        w2 = self.cfg.get("residual_weight_feature", 0.3)
        w3 = self.cfg.get("residual_weight_objects", 0.2)

        total_residual = (
            w1 * residual_category +
            w2 * residual_feature +
            w3 * residual_objects
        )

        return float(total_residual)

    def _compute_feature_residual(
        self,
        expected_feature: torch.Tensor,
        actual_image: Optional[np.ndarray]
    ) -> float:
        """计算特征空间的残差（CLIP embedding距离）"""

        if actual_image is None:
            return 0.5

        try:
            # 提取实际图像的CLIP特征
            from PIL import Image
            actual_pil = Image.fromarray(actual_image)
            actual_tensor = self.clip_preprocess(actual_pil).unsqueeze(0)

            with torch.no_grad():
                actual_feature = self.clip_model.encode_image(actual_tensor)
                actual_feature = F.normalize(actual_feature, dim=-1)

            # 计算余弦相似度
            expected_feature_norm = F.normalize(expected_feature.unsqueeze(0), dim=-1)
            similarity = F.cosine_similarity(expected_feature_norm, actual_feature, dim=-1)

            # 残差 = 1 - 相似度
            residual = 1.0 - float(similarity.item())
            return residual

        except Exception as e:
            print(f"[Semantic Critic] Error computing feature residual: {e}")
            return 0.5

    def _compute_object_residual(
        self,
        expected_semantic_dist: SemanticDistribution,
        actual_objects: List[str]
    ) -> float:
        """
        计算对象级别的残差

        逻辑：检查实际观测到的对象是否符合预期的房间类型
        例如：如果预测是bedroom但观测到toilet，则残差高
        """

        if len(actual_objects) == 0:
            return 0.5  # 无对象，无法判断

        # 房间类型 -> 典型对象的映射
        room_object_association = {
            "bedroom": ["bed", "wardrobe", "nightstand", "pillow", "blanket"],
            "bathroom": ["toilet", "sink", "shower", "bathtub", "mirror"],
            "kitchen": ["refrigerator", "stove", "oven", "sink", "microwave", "cabinet"],
            "living_room": ["sofa", "tv", "coffee_table", "armchair", "lamp"],
            "dining_room": ["dining_table", "chair", "cabinet"],
            "hallway": ["door", "picture", "lamp"],
            "office": ["desk", "chair", "computer", "bookshelf"],
            "closet": ["wardrobe", "shelf", "hanger"],
        }

        # 计算每个预测类别的对象匹配得分
        match_scores = []
        for category, prob in zip(expected_semantic_dist.categories,
                                   expected_semantic_dist.probabilities):
            if category in room_object_association:
                expected_objects = room_object_association[category]
                # 计算实际对象与期望对象的交集
                actual_objects_lower = [obj.lower() for obj in actual_objects]
                matches = sum(1 for exp_obj in expected_objects if exp_obj in actual_objects_lower)
                match_score = matches / max(len(expected_objects), 1)
                match_scores.append(prob * match_score)
            else:
                match_scores.append(0.0)

        # 期望匹配得分（概率加权）
        expected_match = sum(match_scores)

        # 残差 = 1 - 匹配得分
        residual = 1.0 - expected_match
        return residual

    def _diagnose_falsification_cause(
        self,
        ghost_node: EpistemicNode,
        actual_observation: Dict
    ) -> Tuple[str, str]:
        """
        使用VLM诊断证伪的物理原因

        常见幻觉类型：
        1. mirror_reflection: 镜子反射（将镜中景象误认为真实房间）
        2. glass_window: 玻璃窗户（将窗外景象误认为室内空间）
        3. painting_poster: 画作/海报（将2D图像误认为3D空间）
        4. occlusion: 遮挡（部分遮挡导致误判）
        5. lighting_artifact: 光照伪影（阴影、反光等）
        """

        # 编码预期图像和实际图像
        expected_img_base64 = None
        if hasattr(ghost_node, 'feature') and ghost_node.feature is not None:
            # 将tensor转换为numpy
            if isinstance(ghost_node.feature, torch.Tensor):
                expected_img = ghost_node.feature.cpu().numpy()
                if expected_img.dtype != np.uint8:
                    expected_img = (expected_img * 255).astype(np.uint8)
            else:
                expected_img = ghost_node.feature
            expected_img_base64 = encode_tensor2base64(expected_img)

        actual_img = actual_observation.get("rgb_image", None)
        actual_img_base64 = None
        if actual_img is not None:
            actual_img_base64 = encode_tensor2base64(actual_img)

        # 构建VLM提示
        sys_prompt = """You are an expert in visual perception error diagnosis.
Your task is to analyze WHY a semantic prediction failed by identifying the PHYSICAL CAUSE of the perceptual error.

Common causes:
1. mirror_reflection: Mirror reflecting another room's image
2. glass_window: Glass window showing outdoor or another room
3. painting_poster: 2D artwork mistaken for 3D space
4. occlusion: Partial occlusion causing misidentification
5. lighting_artifact: Shadow or reflection artifacts
6. distant_view: Distant view mistaken for nearby space
7. Other: Other causes (specify)

Output format:
{
    "physical_cause": "mirror_reflection",
    "explanation": "The predicted bedroom was actually a mirror reflection of a bedroom. The agent mistook the mirror surface for a real room entrance."
}
"""

        # 构建预测信息
        predicted_class = ghost_node.semantic_dist.categories[
            np.argmax(ghost_node.semantic_dist.probabilities)
        ]
        actual_class = actual_observation.get("semantic_class", "Unknown")

        user_prompt = f"""Analyze the perceptual error:

Expected prediction: {predicted_class}
Actual observation: {actual_class}

Detected objects in actual view: {', '.join(actual_observation.get('detected_objects', [])[:10])}

Please identify the physical cause of this prediction error."""

        contents = []
        if expected_img_base64:
            contents.append(("Expected view (frontier observation):", expected_img_base64))
        if actual_img_base64:
            contents.append(("Actual view (upon arrival):", actual_img_base64))
        contents.append((user_prompt,))

        # 调用VLM
        try:
            response = call_openai_api(sys_prompt, contents)
            if response is None:
                return "unknown", "VLM diagnosis failed"

            # 解析JSON
            import json
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx == -1 or end_idx == 0:
                return "unknown", response

            json_str = response[start_idx:end_idx]
            diagnosis = json.loads(json_str)

            physical_cause = diagnosis.get("physical_cause", "unknown")
            explanation = diagnosis.get("explanation", "")

            return physical_cause, explanation

        except Exception as e:
            print(f"[Semantic Critic] Error in VLM diagnosis: {e}")
            return "unknown", f"Prediction failed: expected {predicted_class}, got {actual_class}"

    def _get_cascade_deleted_node_ids(self, falsified_node_id: str) -> List[str]:
        """
        获取级联删除的节点ID列表（用于报告）
        这个方法应该在cascade_delete之前调用以记录将被删除的节点
        """
        from collections import deque

        deleted_nodes = []
        queue = deque([falsified_node_id])
        visited = set()

        while queue:
            current_id = queue.popleft()

            # 避免重复访问
            if current_id in visited:
                continue
            visited.add(current_id)

            # 获取所有依赖当前节点的子节点
            if current_id in self.epistemic_graph.dependencies:
                for dep in self.epistemic_graph.dependencies[current_id]:
                    child_id = dep.child_id

                    # 如果子节点还未被访问且存在于图中
                    if child_id not in visited and child_id in self.epistemic_graph.nodes:
                        deleted_nodes.append(child_id)
                        queue.append(child_id)

        return deleted_nodes

    def detect_hallucination_in_snapshot(
        self,
        snapshot: object,
        scene_objects: Dict
    ) -> Optional[str]:
        """
        检测快照中是否包含幻觉对象

        应用场景：即使在Observed Node中，也可能存在局部幻觉
        （如：真实的卧室中有一面镜子，镜子中的对象不应该被建图）

        Returns:
            如果检测到幻觉，返回幻觉原因；否则返回None
        """

        # 完整实现需要：
        # 1. 深度一致性检查：镜子、玻璃表面的深度通常异常
        # 2. 表面法线分析：检测高反射率表面
        # 3. 对象位置合理性：检测物理上不可能的对象位置
        # 4. VLM辅助判断：询问VLM是否图像中包含镜子/玻璃等反射物

        # 简化实现：基于启发式规则
        if not hasattr(snapshot, 'cluster') or len(snapshot.cluster) == 0:
            return None

        # 提取快照中的对象类别
        snapshot_objects = []
        for obj_id in snapshot.cluster:
            if obj_id in scene_objects:
                obj_class = scene_objects[obj_id].get("class_name", "").lower()
                snapshot_objects.append(obj_class)

        # 规则1: 检测高反射物体（镜子、玻璃）
        reflective_objects = ["mirror", "glass", "window", "tv_screen", "painting", "picture"]
        has_reflective = any(obj in snapshot_objects for obj in reflective_objects)

        if has_reflective:
            # 如果存在反射物体，标记为潜在幻觉源
            # 但不直接证伪，因为镜子本身是真实存在的对象
            # 真正的幻觉是镜子"背后"的虚假Ghost Node
            return "reflective_surface_detected"

        # 规则2: 检测深度异常
        # （需要深度图数据，在简化版本中跳过）
        if hasattr(snapshot, 'depth') and snapshot.depth is not None:
            depth_data = snapshot.depth
            # 检测深度跳变（可能是反射边界）
            # 简化实现：检测深度标准差
            depth_std = np.std(depth_data[depth_data > 0])
            if depth_std > 2.0:  # 阈值可调
                return "depth_inconsistency_detected"

        # 规则3: 检测物理上不合理的对象组合
        # 例如：在浴室中出现床，可能是镜中卧室的反射
        room_specific_objects = {
            "bathroom": ["toilet", "shower", "bathtub", "sink"],
            "bedroom": ["bed", "wardrobe", "nightstand"],
            "kitchen": ["stove", "oven", "refrigerator"],
        }

        # 检测对象组合是否一致
        detected_room_types = []
        for room_type, typical_objects in room_specific_objects.items():
            if any(obj in snapshot_objects for obj in typical_objects):
                detected_room_types.append(room_type)

        # 如果检测到多种房间类型的特征对象，可能是幻觉
        if len(detected_room_types) > 1:
            return f"inconsistent_room_types_detected: {', '.join(detected_room_types)}"

        # 没有检测到明显幻觉
        return None

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            "verification_success_rate": (
                self.stats["successful_verifications"] /
                max(1, self.stats["total_verifications"])
            ),
            "falsification_rate": (
                self.stats["falsifications"] /
                max(1, self.stats["total_verifications"])
            ),
        }


class ExpectationViolationDetector:
    """
    预期-违背检测器
    用于在导航过程中实时检测预期与观测的不一致
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.violation_threshold = cfg.get("expectation_violation_threshold", 0.6)

    def detect_violation(
        self,
        expected_state: Dict,
        observed_state: Dict
    ) -> Tuple[bool, float]:
        """
        检测预期与实际观测的违背

        Args:
            expected_state: 预期状态（如：应该看到门、走廊等）
            observed_state: 实际观测状态

        Returns:
            (is_violated, violation_score)
        """

        # 检查空间一致性
        spatial_violation = self._check_spatial_violation(expected_state, observed_state)

        # 检查语义一致性
        semantic_violation = self._check_semantic_violation(expected_state, observed_state)

        # 综合违背得分
        violation_score = 0.5 * spatial_violation + 0.5 * semantic_violation

        is_violated = violation_score > self.violation_threshold

        return is_violated, violation_score

    def _check_spatial_violation(self, expected: Dict, observed: Dict) -> float:
        """
        检查空间违背（如：预期是开放空间，实际是墙壁）

        Args:
            expected: 预期状态，包含：
                - navigable: bool, 是否可通行
                - distance: float, 预期距离
                - obstacles: List[str], 预期障碍物
            observed: 实际观测状态，包含相同字段

        Returns:
            违背得分 [0.0, 1.0]，0表示完全一致，1表示完全违背
        """

        violation_score = 0.0
        num_checks = 0

        # 检查1: 可通行性违背
        if "navigable" in expected and "navigable" in observed:
            if expected["navigable"] != observed["navigable"]:
                violation_score += 1.0
            num_checks += 1

        # 检查2: 距离违背
        if "distance" in expected and "distance" in observed:
            expected_dist = expected["distance"]
            observed_dist = observed["distance"]
            # 计算相对误差
            if expected_dist > 0:
                relative_error = abs(expected_dist - observed_dist) / expected_dist
                # 如果误差超过50%，认为有违背
                violation_score += min(relative_error / 0.5, 1.0)
                num_checks += 1

        # 检查3: 障碍物违背
        if "obstacles" in expected and "obstacles" in observed:
            expected_obstacles = set(expected["obstacles"])
            observed_obstacles = set(observed["obstacles"])

            # 计算Jaccard距离（1 - Jaccard相似度）
            if len(expected_obstacles) > 0 or len(observed_obstacles) > 0:
                intersection = len(expected_obstacles & observed_obstacles)
                union = len(expected_obstacles | observed_obstacles)
                jaccard_sim = intersection / union if union > 0 else 0.0
                violation_score += (1.0 - jaccard_sim)
                num_checks += 1

        # 检查4: 深度一致性（如果有深度图）
        if "depth_map" in expected and "depth_map" in observed:
            expected_depth = expected["depth_map"]
            observed_depth = observed["depth_map"]

            # 计算深度均方误差（归一化）
            if expected_depth is not None and observed_depth is not None:
                mse = np.mean((expected_depth - observed_depth) ** 2)
                # 归一化到[0, 1]（假设深度范围在0-10m）
                normalized_mse = min(mse / 10.0, 1.0)
                violation_score += normalized_mse
                num_checks += 1

        # 返回平均违背得分
        return violation_score / max(num_checks, 1)

    def _check_semantic_violation(self, expected: Dict, observed: Dict) -> float:
        """
        检查语义违背（如：预期是卧室特征，实际是厨房特征）

        Args:
            expected: 预期状态，包含：
                - room_type: str, 预期房间类型
                - semantic_distribution: SemanticDistribution, 语义分布
                - expected_objects: List[str], 预期对象列表
            observed: 实际观测状态，包含：
                - room_type: str, 实际房间类型
                - detected_objects: List[str], 检测到的对象

        Returns:
            违背得分 [0.0, 1.0]，0表示完全一致，1表示完全违背
        """

        violation_score = 0.0
        num_checks = 0

        # 检查1: 房间类型违背
        if "room_type" in expected and "room_type" in observed:
            expected_room = expected["room_type"]
            observed_room = observed["room_type"]

            if expected_room != observed_room:
                # 如果房间类型完全不同，违背得分高
                violation_score += 1.0
            num_checks += 1

        # 检查2: 基于语义分布的违背
        if "semantic_distribution" in expected and "room_type" in observed:
            semantic_dist = expected["semantic_distribution"]
            observed_room = observed["room_type"]

            # 计算预期分布中实际房间类型的概率
            expected_prob = semantic_dist.get_expected_semantic_score(observed_room)
            # 违背得分 = 1 - 预期概率
            violation_score += (1.0 - expected_prob)
            num_checks += 1

        # 检查3: 对象级别的违背
        if "expected_objects" in expected and "detected_objects" in observed:
            expected_objects = set([obj.lower() for obj in expected["expected_objects"]])
            detected_objects = set([obj.lower() for obj in observed["detected_objects"]])

            # 计算对象集合的不一致程度
            if len(expected_objects) > 0:
                # 计算检测到的对象中有多少在期望之外
                unexpected_objects = detected_objects - expected_objects
                # 计算期望对象中有多少未被检测到
                missing_objects = expected_objects - detected_objects

                # 违背得分：未预期对象比例 + 缺失对象比例
                unexpected_ratio = len(unexpected_objects) / max(len(detected_objects), 1)
                missing_ratio = len(missing_objects) / len(expected_objects)

                object_violation = 0.5 * unexpected_ratio + 0.5 * missing_ratio
                violation_score += object_violation
                num_checks += 1

        # 检查4: 特征级别的违背（如果有CLIP特征）
        if "expected_feature" in expected and "observed_feature" in observed:
            expected_feat = expected["expected_feature"]
            observed_feat = observed["observed_feature"]

            # 计算特征空间的余弦距离
            if expected_feat is not None and observed_feat is not None:
                # 归一化特征
                expected_feat_norm = F.normalize(
                    torch.tensor(expected_feat).unsqueeze(0), dim=-1
                )
                observed_feat_norm = F.normalize(
                    torch.tensor(observed_feat).unsqueeze(0), dim=-1
                )

                # 余弦相似度
                similarity = F.cosine_similarity(
                    expected_feat_norm, observed_feat_norm, dim=-1
                )

                # 违背得分 = 1 - 相似度
                feature_violation = 1.0 - float(similarity.item())
                violation_score += feature_violation
                num_checks += 1

        # 检查5: 语义一致性规则检查
        # 例如：卧室不应该有厨具，浴室不应该有床
        if "room_type" in expected and "detected_objects" in observed:
            expected_room = expected["room_type"]
            detected_objects = [obj.lower() for obj in observed["detected_objects"]]

            # 不兼容对象映射
            incompatible_objects = {
                "bedroom": ["stove", "oven", "refrigerator", "toilet", "shower"],
                "bathroom": ["bed", "sofa", "dining_table", "stove"],
                "kitchen": ["bed", "toilet", "shower", "bathtub"],
                "living_room": ["toilet", "shower", "bathtub", "stove", "oven"],
            }

            if expected_room in incompatible_objects:
                forbidden_objects = incompatible_objects[expected_room]
                # 检查是否有不兼容对象
                has_incompatible = any(
                    obj in detected_objects for obj in forbidden_objects
                )
                if has_incompatible:
                    violation_score += 0.8  # 严重违背
                    num_checks += 1

        # 返回平均违背得分
        return violation_score / max(num_checks, 1)
