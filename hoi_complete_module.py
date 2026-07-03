"""
HOI系统补充模块 - 完整的动作分类、物理分析和mAP评估
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

# ==================== 完整的117个动作分类 ====================

def get_action_categories():
    """
    将117个动作按照交互特性分类
    """
    
    # 需要接触的动作（人和物体需要有物理接触）
    contact_required = [
        # 手持类
        'hold', 'carry', 'pick_up', 'lift', 'wield', 'swing',
        
        # 穿戴类
        'wear',
        
        # 饮食类
        'eat', 'drink_with', 'sip', 'lick', 'bite',
        
        # 工具使用类
        'cut_with', 'brush_with', 'stab', 'type_on', 'text_on', 'write',
        
        # 操作类
        'adjust', 'assemble', 'open', 'close', 'turn', 'twist', 'operate',
        'control', 'stir', 'squeeze', 'peel', 'break', 'repair', 'install',
        'pack', 'tie', 'zip', 'fill', 'pour',
        
        # 推拉类
        'push', 'pull', 'drag',
        
        # 清洁类
        'clean', 'wash', 'dry', 'wipe',
        
        # 身体接触类
        'hug', 'kiss', 'pet', 'scratch', 'groom',
        
        # 坐/躺类（需要接触）
        'sit_on', 'sit_at', 'lie_on', 'lean_on',
        
        # 站立类（需要接触）
        'stand_on', 'hop_on', 'straddle',
        
        # 骑乘类
        'ride', 'drive', 'row', 'sail',
        
        # 抓握类
        'catch', 'grab', 'grasp',
        
        # 其他接触类
        'milk', 'shear', 'feed', 'toast', 'serve', 'set'
    ]
    
    # 需要距离的动作（人和物体之间需要有一定距离）
    distance_required = [
        # 投掷类
        'throw', 'launch', 'toss',
        
        # 踢类
        'kick',
        
        # 指向类
        'point', 'direct',
        
        # 观察类
        'watch', 'inspect', 'check', 'read', 'observe',
        
        # 追逐类
        'chase', 'hunt', 'follow',
        
        # 释放类
        'release', 'fly',
        
        # 远程交互
        'talk_on', 'call', 'photograph', 'shoot'
    ]
    
    # 可近可远的动作（距离灵活）
    flexible_distance = [
        # 看类
        'look_at', 'smell', 'listen',
        
        # 运动类
        'run', 'walk', 'jump', 'slide', 'race',
        
        # 玩耍类
        'play', 'dribble', 'spin',
        
        # 教学类
        'teach', 'train', 'show',
        
        # 阻挡类
        'block', 'stop_at',
        
        # 挥手类
        'wave', 'greet', 'signal',
        
        # 其他
        'board', 'exit', 'paint', 'park', 'pay', 'sign',
        'tag', 'lose', 'make', 'move', 'blow', 'grind',
        'flip', 'flush', 'herd', 'hit', 'hose', 'light',
        'load', 'no_interaction', 'pick', 'stand_under'
    ]
    
    return {
        'contact_required': contact_required,
        'distance_required': distance_required,
        'flexible_distance': flexible_distance
    }


# ==================== 完整的物理因素分析 ====================

def analyze_physics_constraints_complete(proposal, image_shape=None):
    """
    对所有117个动作进行完整的物理约束分析
    
    Args:
        proposal: HOI提议，包含verb, object_class, human_bbox, object_bbox
        image_shape: 图像尺寸 (height, width)
    
    Returns:
        float: 物理合理性得分 (0-1)
    """
    
    verb = proposal.verb
    object_class = proposal.object_class
    human_bbox = proposal.human_bbox
    object_bbox = proposal.object_bbox
    
    # 计算基本空间关系
    h_cx = (human_bbox[0] + human_bbox[2]) / 2
    h_cy = (human_bbox[1] + human_bbox[3]) / 2
    h_width = human_bbox[2] - human_bbox[0]
    h_height = human_bbox[3] - human_bbox[1]
    
    o_cx = (object_bbox[0] + object_bbox[2]) / 2
    o_cy = (object_bbox[1] + object_bbox[3]) / 2
    o_width = object_bbox[2] - object_bbox[0]
    o_height = object_bbox[3] - object_bbox[1]
    
    # 相对位置
    rel_x = o_cx - h_cx
    rel_y = o_cy - h_cy
    distance = np.sqrt(rel_x**2 + rel_y**2)
    
    # 默认得分
    score = 0.8
    
    # ========== 手部操作类动作 ==========
    if verb in ['hold', 'carry', 'pick_up', 'lift', 'grasp', 'grab']:
        # 物体应该在手部可及范围
        hand_reach = h_height * 0.6  # 手臂长度约为身高的60%
        
        if distance > hand_reach:
            score *= 0.3  # 太远
        if o_cy < h_cy - h_height * 0.8:  # 物体太高
            score *= 0.2
        if o_width > h_width * 2:  # 物体太大
            score *= 0.5
            
    # ========== 脚部动作 ==========
    elif verb in ['kick', 'step_on']:
        # 物体应该在脚部高度和范围
        foot_level = h_cy + h_height * 0.4  # 脚部位置
        
        if abs(o_cy - foot_level) > h_height * 0.2:
            score *= 0.4
        if distance > h_height * 0.5:
            score *= 0.3
            
    # ========== 坐/站类动作 ==========
    elif verb in ['sit_on', 'sit_at']:
        # 物体应该在合适的高度
        sit_height = h_cy + h_height * 0.2  # 坐下时的高度
        
        if o_cy < sit_height - h_height * 0.3:
            score *= 0.4  # 太高
        if o_width < h_width * 0.5:
            score *= 0.3  # 太小无法坐
            
    elif verb in ['stand_on', 'hop_on']:
        # 物体需要足够大和稳定
        if o_width < h_width * 0.3 or o_height < h_height * 0.05:
            score *= 0.2  # 太小无法站立
        if o_cy < h_cy + h_height * 0.3:
            score *= 0.4  # 太高难以攀爬
            
    elif verb == 'lie_on':
        # 物体需要足够大
        if o_width < h_width * 0.8 or o_height < h_height * 0.3:
            score *= 0.3  # 太小无法躺
            
    # ========== 骑乘类动作 ==========
    elif verb in ['ride', 'straddle']:
        # 物体应该在合适的高度和大小
        if object_class in ['bicycle', 'motorcycle', 'horse']:
            if o_height < h_height * 0.3 or o_height > h_height * 0.8:
                score *= 0.4
        else:
            score *= 0.3  # 不适合骑乘的物体
            
    elif verb == 'drive':
        # 需要在车辆内部
        if object_class in ['car', 'truck', 'bus']:
            if not _check_inside(human_bbox, object_bbox):
                score *= 0.3
        else:
            score *= 0.1
            
    # ========== 投掷类动作 ==========
    elif verb in ['throw', 'toss', 'launch']:
        # 需要一定距离，物体不能太大
        if distance < h_height * 0.2:
            score *= 0.5  # 太近
        if o_width > h_width or o_height > h_height * 0.5:
            score *= 0.3  # 物体太大难以投掷
            
    # ========== 推拉类动作 ==========
    elif verb in ['push', 'pull', 'drag']:
        # 物体大小和距离的关系
        if distance > h_height * 0.8:
            score *= 0.4  # 太远
        if o_width > h_width * 3 and o_height > h_height * 2:
            score *= 0.3  # 物体太大难以推动
            
    # ========== 工具使用类 ==========
    elif verb in ['cut_with', 'stab', 'stir']:
        # 需要手持工具
        if object_class in ['knife', 'scissors', 'fork', 'spoon']:
            if distance > h_height * 0.5:
                score *= 0.3
        else:
            score *= 0.2  # 不是合适的工具
            
    elif verb in ['type_on', 'text_on']:
        # 需要在设备前方
        if object_class in ['keyboard', 'laptop', 'cell_phone']:
            if distance > h_height * 0.5:
                score *= 0.3
            if abs(rel_y) > h_height * 0.3:
                score *= 0.5  # 高度不合适
        else:
            score *= 0.1
            
    # ========== 饮食类动作 ==========
    elif verb in ['eat', 'drink_with', 'sip', 'lick']:
        # 物体应该在嘴部附近
        mouth_level = h_cy - h_height * 0.3
        
        if distance > h_height * 0.4:
            score *= 0.3
        if abs(o_cy - mouth_level) > h_height * 0.2:
            score *= 0.5
            
    # ========== 穿戴类动作 ==========
    elif verb == 'wear':
        # 物体大小应该合适
        if object_class in ['tie', 'backpack', 'hat', 'glasses']:
            if o_width > h_width * 1.5:
                score *= 0.4  # 太大
        else:
            score *= 0.3  # 不是可穿戴物品
            
    # ========== 清洁类动作 ==========
    elif verb in ['clean', 'wash', 'dry', 'wipe']:
        # 需要接近物体
        if distance > h_height * 0.6:
            score *= 0.3
            
    # ========== 身体接触类 ==========
    elif verb in ['hug', 'kiss', 'pet']:
        # 需要非常近
        if distance > h_height * 0.3:
            score *= 0.2
        if object_class in ['person', 'dog', 'cat', 'teddy_bear']:
            score *= 1.2  # 合适的对象
        else:
            score *= 0.5
            
    # ========== 操作类动作 ==========
    elif verb in ['open', 'close', 'turn', 'adjust']:
        # 需要在操作范围内
        if distance > h_height * 0.5:
            score *= 0.3
            
    elif verb in ['fill', 'pour']:
        # 需要在上方
        if o_cy > h_cy:
            score *= 0.5  # 物体在下方难以倒入
        if distance > h_height * 0.5:
            score *= 0.3
            
    # ========== 运动类动作 ==========
    elif verb in ['jump', 'hop_on']:
        # 需要空间
        if distance < h_height * 0.1:
            score *= 0.3  # 太近
            
    elif verb in ['run', 'walk', 'race']:
        # 需要移动空间
        if object_class == 'person':
            score *= 1.0  # 可以和人一起跑
        else:
            score *= 0.7
            
    # ========== 观察类动作 ==========
    elif verb in ['watch', 'look_at', 'inspect', 'check', 'read']:
        # 需要合适的距离和角度
        if distance < h_height * 0.1:
            score *= 0.5  # 太近
        if distance > h_height * 3:
            score *= 0.4  # 太远
            
    # ========== 指向类动作 ==========
    elif verb in ['point', 'direct']:
        # 需要一定距离
        if distance < h_height * 0.3:
            score *= 0.5
            
    # ========== 其他特殊动作 ==========
    elif verb == 'stand_under':
        # 物体应该在上方
        if o_cy > h_cy:
            score *= 0.1  # 物体在下方
            
    elif verb == 'board':
        # 进入交通工具
        if object_class in ['airplane', 'bus', 'train', 'boat']:
            if distance > h_height:
                score *= 0.3
        else:
            score *= 0.2
            
    elif verb == 'exit':
        # 离开交通工具
        if object_class in ['car', 'bus', 'train', 'airplane']:
            if not _check_near(human_bbox, object_bbox, threshold=h_height*0.5):
                score *= 0.3
        else:
            score *= 0.2
            
    # 确保得分在[0,1]范围内
    return max(0.0, min(1.0, score))


def _check_inside(bbox1, bbox2):
    """检查bbox1是否在bbox2内部"""
    return (bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and
            bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3])


def _check_near(bbox1, bbox2, threshold=50):
    """检查两个bbox是否接近"""
    c1 = [(bbox1[0]+bbox1[2])/2, (bbox1[1]+bbox1[3])/2]
    c2 = [(bbox2[0]+bbox2[2])/2, (bbox2[1]+bbox2[3])/2]
    dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    return dist < threshold


# ==================== 完整的mAP评估 ====================

class HOIEvaluator:
    """
    HOI检测的mAP评估器
    支持HICO-DET格式的评估
    """
    
    def __init__(self, 
                 verb_classes=None,
                 object_classes=None,
                 iou_threshold=0.5):
        """
        Args:
            verb_classes: 动作类别列表
            object_classes: 物体类别列表  
            iou_threshold: 正样本的IoU阈值
        """
        self.verb_classes = verb_classes or VERB_CLASSES
        self.object_classes = object_classes or OBJECT_CLASSES
        self.iou_threshold = iou_threshold
        
        # 构建HOI类别
        self.hoi_categories = self._build_hoi_categories()
        
    def _build_hoi_categories(self):
        """构建所有可能的HOI类别组合"""
        categories = []
        for verb in self.verb_classes:
            for obj in self.object_classes:
                categories.append(f"{verb}_{obj}")
        return categories
    
    def compute_map(self, predictions, ground_truths):
        """
        计算mAP
        
        Args:
            predictions: Dict[image_id, List[HOIInstance]]
                每个HOIInstance包含:
                - human_bbox: [x1,y1,x2,y2]
                - object_bbox: [x1,y1,x2,y2]
                - verb: str
                - object_class: str
                - confidence: float
                
            ground_truths: Dict[image_id, List[Dict]]
                每个Dict包含:
                - human_bbox: [x1,y1,x2,y2]
                - object_bbox: [x1,y1,x2,y2]
                - verb: str (或verb_id需要转换)
                - object_class: str (或object_id需要转换)
                
        Returns:
            Dict: 包含mAP和详细指标
        """
        
        # 为每个HOI类别计算AP
        ap_scores = {}
        
        for verb in self.verb_classes:
            for obj in self.object_classes:
                hoi_class = f"{verb}_{obj}"
                
                # 收集该类别的所有预测和真值
                class_predictions = []
                class_ground_truths = []
                
                # 收集预测
                for img_id, preds in predictions.items():
                    for pred in preds:
                        if hasattr(pred, 'verb'):  # HOIInstance对象
                            pred_verb = pred.verb
                            pred_obj = pred.object_class
                            if pred_verb == verb and pred_obj == obj:
                                class_predictions.append({
                                    'image_id': img_id,
                                    'human_bbox': pred.human_bbox,
                                    'object_bbox': pred.object_bbox,
                                    'score': pred.confidence
                                })
                        else:  # Dict格式
                            if pred['verb'] == verb and pred['object_class'] == obj:
                                class_predictions.append({
                                    'image_id': img_id,
                                    'human_bbox': pred['human_bbox'],
                                    'object_bbox': pred['object_bbox'],
                                    'score': pred['confidence']
                                })
                
                # 收集真值
                for img_id, gts in ground_truths.items():
                    for gt in gts:
                        gt_verb = gt.get('verb', '')
                        gt_obj = gt.get('object_class', '')
                        
                        # 如果是ID格式，需要转换
                        if isinstance(gt_verb, int):
                            gt_verb = self.verb_classes[gt_verb] if gt_verb < len(self.verb_classes) else ''
                        if isinstance(gt_obj, int):
                            gt_obj = self.object_classes[gt_obj] if gt_obj < len(self.object_classes) else ''
                        
                        if gt_verb == verb and gt_obj == obj:
                            class_ground_truths.append({
                                'image_id': img_id,
                                'human_bbox': gt['human_bbox'],
                                'object_bbox': gt['object_bbox']
                            })
                
                # 计算该类别的AP
                if len(class_ground_truths) > 0:
                    ap = self._compute_ap(class_predictions, class_ground_truths)
                    ap_scores[hoi_class] = ap
        
        # 计算mAP
        if ap_scores:
            mAP = np.mean(list(ap_scores.values()))
        else:
            mAP = 0.0
        
        # 按动作和物体分别计算mAP
        verb_ap = defaultdict(list)
        object_ap = defaultdict(list)
        
        for hoi_class, ap in ap_scores.items():
            verb, obj = hoi_class.split('_', 1)
            verb_ap[verb].append(ap)
            object_ap[obj].append(ap)
        
        verb_map = {v: np.mean(aps) for v, aps in verb_ap.items()}
        object_map = {o: np.mean(aps) for o, aps in object_ap.items()}
        
        return {
            'mAP': mAP,
            'ap_scores': ap_scores,
            'verb_mAP': verb_map,
            'object_mAP': object_map,
            'num_classes_evaluated': len(ap_scores)
        }
    
    def _compute_ap(self, predictions, ground_truths):
        """
        计算单个HOI类别的Average Precision
        """
        if not predictions:
            return 0.0
        
        # 按置信度排序预测
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        # 初始化
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        # 标记已匹配的真值
        gt_matched = defaultdict(set)
        
        for pred_idx, pred in enumerate(predictions):
            pred_img = pred['image_id']
            pred_h_box = pred['human_bbox']
            pred_o_box = pred['object_bbox']
            
            # 找到该图像的所有真值
            img_gts = [gt for gt in ground_truths if gt['image_id'] == pred_img]
            
            if not img_gts:
                fp[pred_idx] = 1
                continue
            
            # 寻找最佳匹配
            best_match_idx = -1
            best_match_score = 0
            
            for gt_idx, gt in enumerate(img_gts):
                # 跳过已匹配的真值
                if gt_idx in gt_matched[pred_img]:
                    continue
                
                # 计算HOI IoU（人和物体IoU的最小值）
                human_iou = self._compute_iou(pred_h_box, gt['human_bbox'])
                object_iou = self._compute_iou(pred_o_box, gt['object_bbox'])
                hoi_iou = min(human_iou, object_iou)
                
                if hoi_iou > best_match_score:
                    best_match_score = hoi_iou
                    best_match_idx = gt_idx
            
            # 判断是否为正样本
            if best_match_score >= self.iou_threshold:
                tp[pred_idx] = 1
                gt_matched[pred_img].add(best_match_idx)
            else:
                fp[pred_idx] = 1
        
        # 计算precision和recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # 计算AP (11-point interpolation)
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap
    
    def _compute_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-10)
    
    def evaluate_from_json(self, pred_json_path, gt_json_path):
        """
        从JSON文件评估
        
        JSON格式:
        [
            {
                "file_name": "image.jpg",
                "hoi_annotation": [
                    {
                        "subject_id": 0,  # 人的索引
                        "object_id": 1,   # 物体的索引
                        "category_id": 88 # HOI类别ID
                    }
                ],
                "annotations": [
                    {
                        "bbox": [x1,y1,x2,y2],
                        "category_id": 1  # COCO类别ID
                    }
                ]
            }
        ]
        """
        import json
        
        # 加载JSON
        with open(pred_json_path, 'r') as f:
            pred_data = json.load(f)
        
        with open(gt_json_path, 'r') as f:
            gt_data = json.load(f)
        
        # 转换格式
        predictions = self._convert_json_to_predictions(pred_data)
        ground_truths = self._convert_json_to_ground_truths(gt_data)
        
        # 计算mAP
        return self.compute_map(predictions, ground_truths)
    
    def _convert_json_to_predictions(self, json_data):
        """转换JSON格式的预测结果"""
        predictions = {}
        
        if isinstance(json_data, list):
            # HICO格式
            for item in json_data:
                img_id = item['file_name']
                predictions[img_id] = []
                
                if 'predictions' in item:
                    for pred in item['predictions']:
                        predictions[img_id].append({
                            'human_bbox': pred['human_bbox'],
                            'object_bbox': pred['object_bbox'],
                            'verb': pred['verb'],
                            'object_class': pred['object_class'],
                            'confidence': pred['confidence']
                        })
        else:
            # 直接的字典格式
            predictions = json_data
        
        return predictions
    
    def _convert_json_to_ground_truths(self, json_data):
        """转换JSON格式的真值标注"""
        ground_truths = {}
        
        for item in json_data:
            img_id = item['file_name']
            ground_truths[img_id] = []
            
            # 获取检测框
            annotations = item['annotations']
            
            # 获取HOI标注
            for hoi in item.get('hoi_annotation', []):
                subject_id = hoi['subject_id']
                object_id = hoi['object_id']
                category_id = hoi['category_id']
                
                # 获取对应的边界框
                human_bbox = annotations[subject_id]['bbox']
                object_bbox = annotations[object_id]['bbox']
                
                # 转换类别ID到动词和物体
                # 这里需要根据实际的映射关系
                verb_id = category_id // len(self.object_classes)
                object_id = category_id % len(self.object_classes)
                
                verb = self.verb_classes[verb_id] if verb_id < len(self.verb_classes) else 'unknown'
                obj = self.object_classes[object_id] if object_id < len(self.object_classes) else 'unknown'
                
                ground_truths[img_id].append({
                    'human_bbox': human_bbox,
                    'object_bbox': object_bbox,
                    'verb': verb,
                    'object_class': obj
                })
        
        return ground_truths


# ==================== 全局变量 ====================

# 动作列表（117个）
VERB_CLASSES = [
    'adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush_with', 'buy',
    'carry', 'catch', 'chase', 'check', 'clean', 'control', 'cook', 'cut',
    'cut_with', 'direct', 'drag', 'dribble', 'drink_with', 'drive', 'dry', 'eat',
    'eat_at', 'exit', 'feed', 'fill', 'flip', 'flush', 'fly', 'greet', 'grind',
    'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose', 'hug', 'hunt', 'inspect',
    'install', 'jump', 'kick', 'kiss', 'lasso', 'launch', 'lick', 'lie_on',
    'lift', 'light', 'load', 'lose', 'make', 'milk', 'move', 'no_interaction',
    'open', 'operate', 'pack', 'paint', 'park', 'pay', 'peel', 'pet', 'pick',
    'pick_up', 'point', 'pour', 'pull', 'push', 'race', 'read', 'release',
    'repair', 'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear',
    'sign', 'sip', 'sit_at', 'sit_on', 'slide', 'smell', 'spin', 'squeeze',
    'stab', 'stand_on', 'stand_under', 'stick', 'stir', 'stop_at', 'straddle',
    'swing', 'tag', 'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast',
    'train', 'turn', 'type_on', 'walk', 'wash', 'watch', 'wave', 'wear',
    'wield', 'zip'
]

# 物体列表（80个）
OBJECT_CLASSES = [
    'airplane', 'apple', 'backpack', 'banana', 'baseball_bat', 'baseball_glove',
    'bear', 'bed', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle', 'bowl',
    'broccoli', 'bus', 'cake', 'car', 'carrot', 'cat', 'cell_phone', 'chair',
    'clock', 'couch', 'cow', 'cup', 'dining_table', 'dog', 'donut', 'elephant',
    'fire_hydrant', 'fork', 'frisbee', 'giraffe', 'hair_drier', 'handbag', 'horse',
    'hot_dog', 'keyboard', 'kite', 'knife', 'laptop', 'microwave', 'motorcycle',
    'mouse', 'orange', 'oven', 'parking_meter', 'person', 'pizza', 'potted_plant',
    'refrigerator', 'remote', 'sandwich', 'scissors', 'sheep', 'sink', 'skateboard',
    'skis', 'snowboard', 'spoon', 'sports_ball', 'stop_sign', 'suitcase',
    'surfboard', 'teddy_bear', 'tennis_racket', 'tie', 'toaster', 'toilet',
    'toothbrush', 'traffic_light', 'train', 'truck', 'tv', 'umbrella', 'vase',
    'wine_glass', 'zebra'
]


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("HOI系统补充模块测试")
    print("=" * 60)
    
    # 测试动作分类
    print("\n1. 动作分类统计:")
    categories = get_action_categories()
    print(f"   需要接触的动作: {len(categories['contact_required'])} 个")
    print(f"   需要距离的动作: {len(categories['distance_required'])} 个")
    print(f"   灵活距离的动作: {len(categories['flexible_distance'])} 个")
    
    # 测试物理分析
    print("\n2. 物理因素分析测试:")
    
    # 创建测试用例
    class TestProposal:
        def __init__(self, verb, obj, h_bbox, o_bbox):
            self.verb = verb
            self.object_class = obj
            self.human_bbox = h_bbox
            self.object_bbox = o_bbox
    
    test_cases = [
        TestProposal('ride', 'bicycle', [100,100,200,300], [150,250,250,350]),
        TestProposal('kick', 'sports_ball', [100,100,200,300], [180,280,220,320]),
        TestProposal('eat', 'apple', [100,100,200,300], [150,120,170,140]),
        TestProposal('sit_on', 'chair', [100,100,200,300], [100,250,200,320])
    ]
    
    for prop in test_cases:
        score = analyze_physics_constraints_complete(prop)
        print(f"   {prop.verb} + {prop.object_class}: {score:.2f}")
    
    # 测试mAP评估
    print("\n3. mAP评估器测试:")
    evaluator = HOIEvaluator()
    print(f"   HOI类别总数: {len(evaluator.hoi_categories)}")
    print(f"   IoU阈值: {evaluator.iou_threshold}")
    
    # 创建模拟数据
    mock_predictions = {
        'image1': [
            {
                'human_bbox': [100,100,200,300],
                'object_bbox': [150,250,250,350],
                'verb': 'ride',
                'object_class': 'bicycle',
                'confidence': 0.9
            }
        ]
    }
    
    mock_ground_truths = {
        'image1': [
            {
                'human_bbox': [95,95,205,305],
                'object_bbox': [145,245,255,355],
                'verb': 'ride',
                'object_class': 'bicycle'
            }
        ]
    }
    
    results = evaluator.compute_map(mock_predictions, mock_ground_truths)
    print(f"   测试mAP: {results['mAP']:.4f}")
    
    print("\n测试完成!")