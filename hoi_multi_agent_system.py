"""
HOI Agent系统 - 完整版本（单文件）
包含：
- LLM驱动的规则生成
- 完整的117个动作支持
- 详细的物理分析
- mAP评估功能
- 多Agent辩论机制
"""

import json
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import torch.nn.functional as F
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    AutoModelForCausalLM, AutoTokenizer,
    BlipProcessor, BlipForConditionalGeneration,
    BlipForImageTextRetrieval
)
import re
from scipy.optimize import linear_sum_assignment
try:
    from ultralytics import YOLO
except:
    print("YOLO not available, will use fallback detection")
    YOLO = None
import warnings
import os
warnings.filterwarnings('ignore')


OFFLINE_RULE_CACHE_PATH = os.path.join(
    os.path.dirname(__file__),
    'offline_llm_rule_cache.json'
)




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


COCO_TO_OUR_CLASSES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic_light',
    11: 'fire_hydrant', 13: 'stop_sign', 14: 'parking_meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports_ball', 38: 'kite', 39: 'baseball_bat',
    40: 'baseball_glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis_racket',
    44: 'bottle', 46: 'wine_glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
    55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot_dog', 59: 'pizza',
    60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted_plant',
    65: 'bed', 67: 'dining_table', 70: 'toilet', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell_phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy_bear', 89: 'hair_drier',
    90: 'toothbrush'
}



@dataclass
class HOIInstance:
    """HOI实例"""
    human_bbox: List[float]
    object_bbox: List[float]
    object_class: str
    verb: str
    confidence: float
    reasoning: Dict = field(default_factory=dict)
    debate_history: List = field(default_factory=list)

@dataclass
class AgentArgument:
    """Agent论据"""
    agent_name: str
    proposal_id: int
    stance: str  
    evidence: Dict
    confidence: float
    reasoning: str
    response_to: Optional[str] = None



def get_action_categories():
    """将117个动作按照交互特性分类"""
    
    
    contact_required = [
        
        'hold', 'carry', 'pick_up', 'lift', 'wield', 'swing',
        
        'wear',
        
        'eat', 'drink_with', 'sip', 'lick',
        
        'cut_with', 'brush_with', 'stab', 'type_on', 'text_on',
        
        'adjust', 'assemble', 'open', 'turn', 'operate',
        'control', 'stir', 'squeeze', 'peel', 'break', 'repair', 'install',
        'pack', 'tie', 'zip', 'fill', 'pour',
        
        'push', 'pull', 'drag',
        
        'clean', 'wash', 'dry',
        
        'hug', 'kiss', 'pet', 'scratch', 'groom',
        
        'sit_on', 'sit_at', 'lie_on',
        
        'stand_on', 'hop_on', 'straddle',
        
        'ride', 'drive', 'row', 'sail',
        
        'catch',
        
        'milk', 'shear', 'feed', 'toast', 'serve', 'set', 'paint',
        'cut', 'grind', 'flip', 'make', 'load', 'stick'
    ]
    
    
    distance_required = [
        
        'throw', 'launch',
        
        'kick',
        
        'point', 'direct',
        
        'watch', 'inspect', 'check', 'read',
        
        'chase', 'hunt',
        
        'release', 'fly',
        
        'talk_on'
    ]
    
    
    flexible_distance = [
        
        'smell',
        
        'run', 'walk', 'jump', 'slide', 'race',
        
        'dribble', 'spin',
        
        'teach', 'train',
        
        'block', 'stop_at',
        
        'wave', 'greet',
        
        'board', 'exit', 'park', 'pay', 'sign',
        'tag', 'lose', 'move', 'blow', 'hit', 'hose', 'light',
        'no_interaction', 'pick', 'stand_under', 'flush',
        'herd', 'buy'
    ]
    
    return {
        'contact_required': contact_required,
        'distance_required': distance_required,
        'flexible_distance': flexible_distance
    }



def analyze_physics_constraints_complete(proposal, image_shape=None):
    """对所有117个动作进行完整的物理约束分析"""
    
    verb = proposal.verb
    object_class = proposal.object_class
    human_bbox = proposal.human_bbox
    object_bbox = proposal.object_bbox
    
    
    h_cx = (human_bbox[0] + human_bbox[2]) / 2
    h_cy = (human_bbox[1] + human_bbox[3]) / 2
    h_width = human_bbox[2] - human_bbox[0]
    h_height = human_bbox[3] - human_bbox[1]
    
    o_cx = (object_bbox[0] + object_bbox[2]) / 2
    o_cy = (object_bbox[1] + object_bbox[3]) / 2
    o_width = object_bbox[2] - object_bbox[0]
    o_height = object_bbox[3] - object_bbox[1]
    
    
    rel_x = o_cx - h_cx
    rel_y = o_cy - h_cy
    distance = np.sqrt(rel_x**2 + rel_y**2)
    
    
    score = 0.8
    
    
    if verb in ['hold', 'carry', 'pick_up', 'pick', 'lift', 'wield']:
        hand_reach = h_height * 0.6
        if distance > hand_reach:
            score *= 0.3
        if o_cy < h_cy - h_height * 0.8:
            score *= 0.2
        if o_width > h_width * 2:
            score *= 0.5
            
    
    elif verb in ['kick']:
        foot_level = h_cy + h_height * 0.4
        if abs(o_cy - foot_level) > h_height * 0.2:
            score *= 0.4
        if distance > h_height * 0.5:
            score *= 0.3
            
    
    elif verb in ['sit_on', 'sit_at']:
        sit_height = h_cy + h_height * 0.2
        if o_cy < sit_height - h_height * 0.3:
            score *= 0.4
        if o_width < h_width * 0.5:
            score *= 0.3
            
    elif verb in ['stand_on', 'hop_on']:
        if o_width < h_width * 0.3 or o_height < h_height * 0.05:
            score *= 0.2
        if o_cy < h_cy + h_height * 0.3:
            score *= 0.4
            
    elif verb == 'lie_on':
        if o_width < h_width * 0.8 or o_height < h_height * 0.3:
            score *= 0.3
            
    elif verb == 'straddle':
        if o_width > h_width * 1.5:
            score *= 0.4
            
    
    elif verb == 'ride':
        if object_class in ['bicycle', 'motorcycle', 'horse']:
            if o_height < h_height * 0.3 or o_height > h_height * 0.8:
                score *= 0.4
        else:
            score *= 0.3
            
    elif verb == 'drive':
        if object_class in ['car', 'truck', 'bus']:
            
            if not (human_bbox[0] >= object_bbox[0] and human_bbox[2] <= object_bbox[2]):
                score *= 0.3
        else:
            score *= 0.1
            
    
    elif verb in ['throw', 'launch']:
        if distance < h_height * 0.2:
            score *= 0.5
        if o_width > h_width or o_height > h_height * 0.5:
            score *= 0.3
            
    
    elif verb in ['push', 'pull', 'drag']:
        if distance > h_height * 0.8:
            score *= 0.4
        if o_width > h_width * 3 and o_height > h_height * 2:
            score *= 0.3
            
    
    elif verb in ['cut_with', 'stab', 'stir']:
        if object_class in ['knife', 'scissors', 'fork', 'spoon']:
            if distance > h_height * 0.5:
                score *= 0.3
        else:
            score *= 0.2
            
    elif verb in ['type_on', 'text_on']:
        if object_class in ['keyboard', 'laptop', 'cell_phone']:
            if distance > h_height * 0.5:
                score *= 0.3
            if abs(rel_y) > h_height * 0.3:
                score *= 0.5
        else:
            score *= 0.1
            
    
    elif verb in ['eat', 'drink_with', 'sip', 'lick']:
        mouth_level = h_cy - h_height * 0.3
        if distance > h_height * 0.4:
            score *= 0.3
        if abs(o_cy - mouth_level) > h_height * 0.2:
            score *= 0.5
            
    
    elif verb == 'wear':
        if object_class in ['tie', 'backpack', 'handbag', 'hat']:
            if o_width > h_width * 1.5:
                score *= 0.4
        else:
            score *= 0.3
            
    
    elif verb in ['clean', 'wash', 'dry']:
        if distance > h_height * 0.6:
            score *= 0.3
            
    
    elif verb in ['hug', 'kiss', 'pet']:
        if distance > h_height * 0.3:
            score *= 0.2
        if object_class in ['person', 'dog', 'cat', 'teddy_bear']:
            score *= 1.2
        else:
            score *= 0.5
            
    
    elif verb in ['open', 'turn', 'adjust', 'operate', 'control']:
        if distance > h_height * 0.5:
            score *= 0.3
            
    elif verb in ['fill', 'pour']:
        if o_cy > h_cy:
            score *= 0.5
        if distance > h_height * 0.5:
            score *= 0.3
            
    
    elif verb in ['jump', 'hop_on']:
        if distance < h_height * 0.1:
            score *= 0.3
            
    elif verb in ['run', 'walk', 'race']:
        if object_class == 'person':
            score *= 1.0
        else:
            score *= 0.7
            
    
    elif verb in ['watch', 'inspect', 'check', 'read']:
        if distance < h_height * 0.1:
            score *= 0.5
        if distance > h_height * 3:
            score *= 0.4
            
    
    elif verb in ['point', 'direct']:
        if distance < h_height * 0.3:
            score *= 0.5
            
    
    elif verb == 'stand_under':
        if o_cy > h_cy:
            score *= 0.1
            
    elif verb == 'board':
        if object_class in ['airplane', 'bus', 'train', 'boat']:
            if distance > h_height:
                score *= 0.3
        else:
            score *= 0.2
            
    elif verb == 'exit':
        if object_class in ['car', 'bus', 'train', 'airplane']:
            if distance > h_height * 0.5:
                score *= 0.3
        else:
            score *= 0.2
            
    elif verb in ['brush_with']:
        if object_class in ['toothbrush', 'hair_drier']:
            if distance > h_height * 0.3:
                score *= 0.3
        else:
            score *= 0.2
            
    elif verb == 'swing':
        if object_class in ['baseball_bat', 'tennis_racket']:
            if distance > h_height * 0.5:
                score *= 0.3
        else:
            score *= 0.3
            
    
    return max(0.0, min(1.0, score))



class ObjectDetector:
    """使用YOLO或OWL-ViT进行目标检测"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print("初始化目标检测器...")
        
        
        self.use_yolo = False
        if YOLO is not None:
            try:
                self.yolo = YOLO('yolov8x.pt')
                self.use_yolo = True
                print("使用YOLOv8进行目标检测")
            except:
                print("YOLO模型加载失败")
        
        
        if not self.use_yolo:
            try:
                self.owl_processor = AutoProcessor.from_pretrained("google/owlvit-base-patch32")
                self.owl_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    "google/owlvit-base-patch32"
                ).to(device)
                self.owl_model.eval()
                print("使用OWL-ViT进行目标检测")
            except:
                print("OWL-ViT不可用，将使用模拟检测")
    
    def detect(self, image_path: str) -> Dict:
        """检测图像中的人和物体"""
        if self.use_yolo:
            return self._detect_with_yolo(image_path)
        elif hasattr(self, 'owl_model'):
            image = Image.open(image_path).convert('RGB')
            return self._detect_with_owl(image)
        else:
            return self._mock_detection()
    
    def _detect_with_yolo(self, image_path: str) -> Dict:
        """使用YOLO检测"""
        results = self.yolo(image_path, conf=0.25)
        
        detections = {
            'humans': [],
            'objects': []
        }
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    bbox = [x1, y1, x2, y2]
                    
                    if cls == 0:  
                        detections['humans'].append({
                            'bbox': bbox,
                            'confidence': conf
                        })
                    else:
                        class_name = self.yolo.names.get(cls, 'object').replace(" ", "_")
                        detections['objects'].append({
                            'bbox': bbox,
                            'class': class_name,
                            'confidence': conf
                        })
        
        return detections
    
    def _detect_with_owl(self, image: Image) -> Dict:
        """使用OWL-ViT检测"""
        
        texts = ["person", "human"] + OBJECT_CLASSES[:30]
        
        inputs = self.owl_processor(
            text=texts,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.owl_model(**inputs)
        
        
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        results = self.owl_processor.post_process_object_detection(
            outputs=outputs,
            threshold=0.2,
            target_sizes=target_sizes
        )[0]
        
        detections = {
            'humans': [],
            'objects': []
        }
        
        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            bbox = box.tolist()
            class_name = texts[label]
            
            if class_name in ["person", "human"]:
                detections['humans'].append({
                    'bbox': bbox,
                    'confidence': float(score)
                })
            else:
                detections['objects'].append({
                    'bbox': bbox,
                    'class': class_name,
                    'confidence': float(score)
                })
        
        return detections
    
    def _mock_detection(self) -> Dict:
        """模拟检测结果用于测试"""
        return {
            'humans': [
                {'bbox': [100, 100, 200, 300], 'confidence': 0.9},
                {'bbox': [300, 150, 400, 350], 'confidence': 0.85}
            ],
            'objects': [
                {'bbox': [150, 200, 250, 280], 'class': 'bicycle', 'confidence': 0.8},
                {'bbox': [50, 300, 150, 400], 'class': 'bench', 'confidence': 0.75}
            ]
        }



class LLMRuleGenerator:
    """使用LLM动态生成交互规则"""
    
    def __init__(self):
        print("初始化LLM规则生成器...")

        
        self.use_llm = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            if not torch.cuda.is_available():
                self.model.to(self.device)
            self.model_name = model_name
            self.use_llm = True
            print(f"使用{model_name}生成规则")
        except Exception as e:
            print(f"LLM加载失败: {e}，使用预定义规则")
            self.model = None
            self.tokenizer = None
        
        self.cached_rules = self._load_complete_rules()
        self.offline_cache_path = OFFLINE_RULE_CACHE_PATH
        self.offline_rules = self._load_offline_rules()
        if self.offline_rules:
            print(f"已加载离线LLM规则缓存: {self.offline_cache_path}")

    def generate_interaction_rules(self, verb: str, object_class: str) -> Dict:
        """生成特定动作和物体的交互规则"""
        offline_rule = self._get_offline_rule(verb, object_class)
        if offline_rule:
            return offline_rule

        if self.use_llm:
            try:
                return self._generate_with_llm(verb, object_class)
            except:
                return self._use_cached_rules(verb, object_class)
        else:
            return self._use_cached_rules(verb, object_class)
    
    def _generate_with_llm(self, verb: str, object_class: str) -> Dict:
        """使用LLM生成规则"""
        system_message = "You are a HOI (Human-Object Interaction) expert. Only reply with a single valid JSON object using double quotes."
        user_message = (
            f"Analyze whether a person can '{verb}' a '{object_class}'. "
            "Return JSON with keys: plausibility (0-1 float), physical_possible (bool), common_sense (bool), "
            "safe (bool), reasoning (string <= 60 chars)."
        )

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"{system_message}\nUser: {user_message}\nAssistant:"

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=768
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.2,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)

                return {
                    'plausibility': float(result.get('plausibility', 0.5)),
                    'is_valid': result.get('plausibility', 0.5) > 0.3,
                    'physical_possible': bool(result.get('physical_possible', True)),
                    'common_sense': bool(result.get('common_sense', True)),
                    'safe': bool(result.get('safe', True)),
                    'reasoning': result.get('reasoning', 'LLM analysis'),
                    'requirements': result.get('requirements', 'Standard'),
                    'source': 'llm'
                }
            else:
                
                return self._parse_llm_fallback(response, verb, object_class)
                
        except Exception as e:
            print(f"LLM生成错误: {e}")
            return self._use_cached_rules(verb, object_class)
    
    def _parse_llm_fallback(self, response: str, verb: str, object_class: str) -> Dict:
        """备用解析LLM响应"""
        response_lower = response.lower()
        
        
        if any(word in response_lower for word in ['high', 'yes', 'possible', 'common', 'normal']):
            plausibility = 0.8
        elif any(word in response_lower for word in ['medium', 'maybe', 'sometimes']):
            plausibility = 0.5
        else:
            plausibility = 0.2
        
        return {
            'plausibility': plausibility,
            'is_valid': plausibility > 0.3,
            'physical_possible': plausibility > 0.2,
            'common_sense': plausibility > 0.5,
            'safe': plausibility > 0.1,
            'reasoning': response[:200] if response else f'Analysis for {verb} + {object_class}',
            'requirements': 'Standard conditions',
            'source': 'llm_fallback'
        }

    def _get_offline_rule(self, verb: str, object_class: str) -> Optional[Dict]:
        """从离线缓存中读取规则"""
        if not self.offline_rules:
            return None

        verb_rules = self.offline_rules.get(verb)
        if not verb_rules:
            with open("verb_problem.txt", "a+") as f:
                f.write(str(verb) + "\n")
            return None
            

        rule = verb_rules.get(object_class)
        if not rule:
            with open("object_class_problem.txt", "a+") as f:
                f.write(str(object_class) + "\n")
            return None

        rule = dict(rule)
        rule.setdefault('source', 'offline_cache')
        return rule

    def _use_cached_rules(self, verb: str, object_class: str) -> Dict:
        """使用预定义的缓存规则"""
        if verb in self.cached_rules and object_class in self.cached_rules[verb]:
            return self.cached_rules[verb][object_class]
        
        
        return {
            'plausibility': 0.5,
            'is_valid': True,
            'physical_possible': True,
            'common_sense': False,
            'safe': True,
            'reasoning': f'Default rule for {verb} + {object_class}',
            'requirements': 'General conditions',
            'source': 'default'
        }

    def _load_offline_rules(self) -> Optional[Dict]:
        """加载离线缓存文件"""
        if not self.offline_cache_path or not os.path.exists(self.offline_cache_path):
            return None

        try:
            with open(self.offline_cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"读取离线缓存失败: {e}")
            return None
    
    def _load_complete_rules(self) -> Dict:
        """加载所有117个动作的完整规则"""
        rules = {}
        
        
        verb_object_rules = {
            'adjust': {'tie': 1.0, 'clock': 0.9, 'chair': 0.8, 'tv': 0.9, 'laptop': 0.8},
            'assemble': {'bicycle': 0.9, 'chair': 0.8, 'dining_table': 0.7},
            'block': {'person': 0.8, 'car': 0.7, 'bicycle': 0.7},
            'blow': {'cake': 0.9, 'hair_drier': 0.3, 'kite': 0.7},
            'board': {'airplane': 0.9, 'bus': 0.95, 'train': 0.95, 'boat': 0.9},
            'break': {'vase': 0.7, 'bowl': 0.6, 'bottle': 0.6},
            'brush_with': {'toothbrush': 1.0, 'hair_drier': 0.2},
            'buy': {obj: 0.9 for obj in OBJECT_CLASSES},
            'carry': {obj: 0.8 if obj not in ['airplane', 'bus', 'train', 'truck', 'elephant'] else 0.1 
                     for obj in OBJECT_CLASSES},
            'catch': {'sports_ball': 1.0, 'frisbee': 0.95, 'baseball_glove': 0.9},
            'chase': {'dog': 0.8, 'cat': 0.8, 'person': 0.7, 'bird': 0.7},
            'check': {'cell_phone': 1.0, 'clock': 0.9, 'laptop': 0.95, 'book': 0.8},
            'clean': {'sink': 0.9, 'toilet': 0.8, 'dining_table': 0.9, 'car': 0.8},
            'control': {'remote': 1.0, 'tv': 0.8, 'laptop': 0.9, 'car': 0.9},
            'cook': {'pizza': 0.8, 'cake': 0.7, 'hot_dog': 0.8, 'sandwich': 0.8},
            'cut': {'cake': 0.9, 'pizza': 0.9, 'apple': 0.9, 'orange': 0.8},
            'cut_with': {'knife': 1.0, 'scissors': 0.95},
            'direct': {'person': 0.9, 'traffic_light': 0.7, 'car': 0.6},
            'drag': {'chair': 0.8, 'suitcase': 0.9, 'backpack': 0.7},
            'dribble': {'sports_ball': 1.0},
            'drink_with': {'cup': 1.0, 'bottle': 0.95, 'wine_glass': 0.95},
            'drive': {'car': 1.0, 'truck': 0.9, 'bus': 0.8, 'motorcycle': 0.9},
            'dry': {'hair_drier': 1.0, 'dog': 0.7, 'cat': 0.6},
            'eat': {'apple': 1.0, 'banana': 1.0, 'sandwich': 1.0, 'pizza': 1.0, 'donut': 1.0, 'cake': 1.0},
            'eat_at': {'dining_table': 1.0, 'couch': 0.5, 'bed': 0.4},
            'exit': {'car': 1.0, 'bus': 1.0, 'train': 1.0, 'airplane': 1.0},
            'feed': {'dog': 1.0, 'cat': 1.0, 'horse': 0.9, 'cow': 0.9},
            'fill': {'cup': 1.0, 'bottle': 1.0, 'bowl': 0.9, 'sink': 0.9},
            'flip': {'skateboard': 0.9, 'book': 0.9},
            'flush': {'toilet': 1.0},
            'fly': {'airplane': 0.9, 'kite': 1.0, 'frisbee': 0.8},
            'greet': {'person': 1.0},
            'grind': {'skateboard': 1.0, 'snowboard': 0.8},
            'groom': {'dog': 1.0, 'cat': 0.9, 'horse': 0.9},
            'herd': {'cow': 1.0, 'sheep': 1.0},
            'hit': {'sports_ball': 1.0, 'baseball_bat': 0.8, 'tennis_racket': 0.8},
            'hold': {obj: 0.9 if obj not in ['airplane', 'bus', 'train', 'elephant'] else 0.1
                     for obj in OBJECT_CLASSES},
            'hop_on': {'bicycle': 1.0, 'motorcycle': 0.9, 'skateboard': 0.9},
            'hose': {'elephant': 0.7, 'car': 0.9},
            'hug': {'person': 1.0, 'teddy_bear': 1.0, 'dog': 0.9, 'cat': 0.8},
            'hunt': {'bird': 0.6, 'bear': 0.3},
            'inspect': {obj: 0.8 for obj in OBJECT_CLASSES},
            'install': {'tv': 0.9, 'microwave': 0.8, 'oven': 0.8},
            'jump': {'skateboard': 0.8, 'surfboard': 0.7},
            'kick': {'sports_ball': 1.0},
            'kiss': {'person': 1.0, 'teddy_bear': 0.7},
            'lasso': {'horse': 0.9, 'cow': 0.9},
            'launch': {'boat': 0.8, 'kite': 0.9},
            'lick': {'donut': 0.8},
            'lie_on': {'bed': 1.0, 'couch': 0.95, 'bench': 0.6},
            'lift': {obj: 0.8 if obj not in ['airplane', 'bus', 'train', 'truck', 'elephant'] else 0.1
                    for obj in OBJECT_CLASSES},
            'light': {'fire_hydrant': 0.3, 'oven': 0.7},
            'load': {'truck': 0.9, 'car': 0.8, 'boat': 0.7},
            'lose': {'sports_ball': 0.7, 'cell_phone': 0.8, 'remote': 0.8},
            'make': {'bed': 0.9, 'sandwich': 0.9, 'pizza': 0.8, 'cake': 0.8},
            'milk': {'cow': 1.0},
            'move': {obj: 0.7 for obj in OBJECT_CLASSES},
            'no_interaction': {obj: 1.0 for obj in OBJECT_CLASSES},
            'open': {'book': 1.0, 'bottle': 0.9, 'refrigerator': 0.95, 'laptop': 0.95},
            'operate': {'laptop': 1.0, 'cell_phone': 1.0, 'remote': 1.0},
            'pack': {'suitcase': 1.0, 'backpack': 1.0},
            'paint': {'vase': 0.8, 'chair': 0.7},
            'park': {'car': 1.0, 'truck': 0.95, 'motorcycle': 0.95},
            'pay': {'person': 0.9, 'parking_meter': 1.0},
            'peel': {'banana': 1.0, 'orange': 1.0},
            'pet': {'dog': 1.0, 'cat': 1.0, 'horse': 0.8},
            'pick': {'apple': 1.0, 'orange': 0.9},
            'pick_up': {obj: 0.8 if obj not in ['airplane', 'bus', 'train', 'truck'] else 0.1
                       for obj in OBJECT_CLASSES},
            'point': {obj: 1.0 for obj in OBJECT_CLASSES},
            'pour': {'bottle': 0.9, 'cup': 0.8, 'wine_glass': 0.8},
            'pull': {'suitcase': 1.0, 'chair': 0.8},
            'push': {'car': 0.7, 'chair': 0.9, 'dining_table': 0.7},
            'race': {'car': 0.9, 'motorcycle': 0.9, 'bicycle': 0.9, 'horse': 0.8},
            'read': {'book': 1.0, 'cell_phone': 0.9, 'laptop': 0.8},
            'release': {'bird': 0.9, 'frisbee': 0.9, 'kite': 0.8},
            'repair': {'bicycle': 0.9, 'motorcycle': 0.8, 'car': 0.8},
            'ride': {'bicycle': 1.0, 'motorcycle': 1.0, 'horse': 0.95},
            'row': {'boat': 1.0},
            'run': {'person': 0.1},
            'sail': {'boat': 1.0},
            'scratch': {'dog': 0.9, 'cat': 0.9},
            'serve': {'pizza': 0.9, 'cake': 0.9, 'sandwich': 0.9},
            'set': {'dining_table': 1.0, 'clock': 0.9},
            'shear': {'sheep': 1.0},
            'sign': {'book': 0.9},
            'sip': {'cup': 1.0, 'wine_glass': 1.0, 'bottle': 0.8},
            'sit_at': {'dining_table': 1.0},
            'sit_on': {'chair': 1.0, 'couch': 1.0, 'bench': 0.95, 'toilet': 1.0},
            'slide': {'skateboard': 0.9, 'surfboard': 0.8, 'snowboard': 0.9},
            'smell': {'pizza': 0.9, 'cake': 0.9, 'donut': 0.8},
            'spin': {'frisbee': 0.9, 'sports_ball': 0.8},
            'squeeze': {'orange': 0.9, 'bottle': 0.8, 'teddy_bear': 0.9},
            'stab': {'knife': 0.8, 'fork': 0.9},
            'stand_on': {'skateboard': 0.9, 'surfboard': 0.9, 'chair': 0.6},
            'stand_under': {'umbrella': 1.0},
            'stick': {'fork': 0.7},
            'stir': {'cup': 0.9, 'bowl': 0.9, 'spoon': 1.0},
            'stop_at': {'stop_sign': 1.0, 'traffic_light': 1.0},
            'straddle': {'bicycle': 0.9, 'motorcycle': 0.9},
            'swing': {'baseball_bat': 1.0, 'tennis_racket': 1.0},
            'tag': {'person': 1.0},
            'talk_on': {'cell_phone': 1.0},
            'teach': {'person': 1.0, 'dog': 0.7},
            'text_on': {'cell_phone': 1.0, 'laptop': 0.8},
            'throw': {'sports_ball': 1.0, 'frisbee': 1.0},
            'tie': {'tie': 1.0, 'boat': 0.8},
            'toast': {'wine_glass': 1.0, 'cup': 0.8},
            'train': {'dog': 1.0, 'horse': 0.9},
            'turn': {'book': 0.9, 'tv': 0.9, 'laptop': 0.8},
            'type_on': {'keyboard': 1.0, 'laptop': 1.0, 'cell_phone': 0.8},
            'walk': {'dog': 1.0, 'horse': 0.6},
            'wash': {'car': 0.9, 'dog': 0.8, 'sink': 0.9},
            'watch': {'tv': 1.0, 'laptop': 0.9, 'cell_phone': 0.8, 'clock': 0.9},
            'wave': {'person': 1.0},
            'wear': {'tie': 1.0, 'backpack': 1.0, 'handbag': 0.9},
            'wield': {'baseball_bat': 1.0, 'tennis_racket': 1.0, 'knife': 0.8},
            'zip': {'backpack': 1.0, 'suitcase': 1.0}
        }
        
        
        for verb, objects_scores in verb_object_rules.items():
            rules[verb] = {}
            for obj in OBJECT_CLASSES:
                if obj in objects_scores:
                    score = objects_scores[obj]
                else:
                    score = 0.2
                
                rules[verb][obj] = {
                    'plausibility': score,
                    'is_valid': score > 0.3,
                    'physical_possible': score > 0.2,
                    'common_sense': score > 0.5,
                    'safe': score > 0.1,
                    'reasoning': f'Rule for {verb} + {obj}',
                    'requirements': 'Standard conditions',
                    'source': 'cache'
                }
        
        return rules



class BaseAgent:
    """所有Agent的基类，确保方法签名一致"""
    
    def analyze(self, image_path: str, proposal: HOIInstance, 
                debate_history: List = None) -> AgentArgument:
        """统一的分析方法"""
        raise NotImplementedError



class ProposalAgent:
    """提议Agent - 生成初始HOI假设"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print("初始化 Proposal Agent...")
        
        
        try:
            self.blip_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_caption_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(device)
            self.blip_caption_model.eval()
            self.use_blip_caption = True
        except Exception as e:
            print(f"BLIP描述模型加载失败: {e}")
            self.use_blip_caption = False

        
        try:
            self.blip_itm_processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
            self.blip_itm_model = BlipForImageTextRetrieval.from_pretrained(
                "Salesforce/blip-itm-base-coco"
            ).to(device)
            self.blip_itm_model.eval()
            self.use_blip_itm = True
        except Exception as e:
            print(f"BLIP-ITM模型加载失败: {e}")
            self.use_blip_itm = False
    
    def propose(self, image_path: str, detections: Dict) -> List[HOIInstance]:
        """生成HOI提议"""
        image = Image.open(image_path).convert('RGB')
        proposals = []
        
        
        scene_description = ""
        if self.use_blip_caption:
            scene_description = self._generate_scene_description(image)
        
        
        humans = detections['humans']
        objects = detections['objects']
        
        for human in humans:  
            for obj in objects:
                
                spatial_relation = self._compute_spatial_relation(human['bbox'], obj['bbox'])
                
                
                possible_verbs = self._predict_interactions_with_blip(
                    image, human['bbox'], obj['bbox'],
                    obj.get('class', 'object')
                )
                
                
                for verb, score in possible_verbs:
                    if score >= 1.0/(len(possible_verbs)+3):
                        proposal = HOIInstance(
                            human_bbox=human['bbox'],
                            object_bbox=obj['bbox'],
                            object_class=obj.get('class', 'object'),
                            verb=verb,
                            confidence=score,
                            reasoning={
                                'scene_context': scene_description,
                                'spatial_relation': spatial_relation,
                                'initial_score': score
                            }
                        )
                        proposals.append(proposal)
        
        return proposals
    
    def _generate_scene_description(self, image):
        """使用BLIP生成场景描述"""
        prompt = "a photo of"
        inputs = self.blip_caption_processor(image, prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.blip_caption_model.generate(**inputs, max_length=50)

        description = self.blip_caption_processor.decode(out[0], skip_special_tokens=True)
        return description
    
    def _compute_spatial_relation(self, human_bbox, object_bbox):
        """计算空间关系特征"""
        h_cx = (human_bbox[0] + human_bbox[2]) / 2
        h_cy = (human_bbox[1] + human_bbox[3]) / 2
        o_cx = (object_bbox[0] + object_bbox[2]) / 2
        o_cy = (object_bbox[1] + object_bbox[3]) / 2
        
        distance = np.sqrt((h_cx - o_cx)**2 + (h_cy - o_cy)**2)
        
        
        x1 = max(human_bbox[0], object_bbox[0])
        y1 = max(human_bbox[1], object_bbox[1])
        x2 = min(human_bbox[2], object_bbox[2])
        y2 = min(human_bbox[3], object_bbox[3])
        
        if x2 > x1 and y2 > y1:
            intersection = (x2 - x1) * (y2 - y1)
            h_area = (human_bbox[2] - human_bbox[0]) * (human_bbox[3] - human_bbox[1])
            o_area = (object_bbox[2] - object_bbox[0]) * (object_bbox[3] - object_bbox[1])
            union = h_area + o_area - intersection
            iou = intersection / (union + 1e-6)
        else:
            iou = 0
        
        return {
            'distance': distance,
            'iou': iou,
            'relative_position': 'left' if h_cx < o_cx else 'right',
            'vertical_relation': 'above' if h_cy < o_cy else 'below'
        }
    
    def _predict_interactions_with_blip(self, image, human_bbox, object_bbox, object_class):
        """使用BLIP图文匹配模型预测可能的交互"""
        
        x1 = int(max(0, min(human_bbox[0], object_bbox[0])))
        y1 = int(max(0, min(human_bbox[1], object_bbox[1])))
        x2 = int(min(image.size[0], max(human_bbox[2], object_bbox[2])))
        y2 = int(min(image.size[1], max(human_bbox[3], object_bbox[3])))
        
        if x2 <= x1 or y2 <= y1:
            interaction_region = image
        else:
            image_array = np.array(image)
            interaction_region = Image.fromarray(image_array[y1:y2, x1:x2])
        
        
        relevant_verbs = self._select_relevant_verbs(object_class)
        
        if not self.use_blip_itm:
            return [('hold', 0.3), ('watch', 0.2)]

        
        text_prompts = [f"a person {verb} a {object_class}" for verb in relevant_verbs[:20]]

        
        try:
            with torch.no_grad():
                inputs = self.blip_itm_processor(
                    text=text_prompts,
                    images=interaction_region,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                outputs = self.blip_itm_model(**inputs)
                scores = outputs.itm_score[:, 1]
                probs = torch.softmax(scores, dim=0).cpu().numpy()

            
            top_k = min(5, len(relevant_verbs))
            top_indices = np.argsort(probs)[-top_k:][::-1]
            results = [(relevant_verbs[i], float(probs[i])) for i in top_indices]

            return results

        except Exception as e:
            print(f"BLIP推理错误: {e}")
            return [('hold', 0.3), ('no_interaction', 0.2)]
    
    def _select_relevant_verbs(self, object_class):
        """根据物体类型选择相关动词"""
        
        verb_groups = {
            'vehicle': ['ride', 'drive', 'board', 'exit', 'push', 'wash', 'park', 'repair', 'load'],
            'personal_item': ['carry', 'hold', 'pack', 'wear', 'inspect', 'open'],
            'food': ['eat', 'cook', 'cut', 'serve', 'smell', 'hold', 'make', 'cut_with', 'buy'],
            'kitchenware': ['wash', 'hold', 'fill', 'pour', 'dry', 'clean'],
            'furniture': ['sit_on', 'lie_on', 'stand_on', 'move', 'clean', 'push'],
            'electronic': ['operate', 'control', 'type_on', 'text_on', 'watch', 'check'],
            'appliance': ['open', 'operate', 'clean', 'repair', 'install', 'wash'],
            'bathroom': ['clean', 'flush', 'wash', 'repair', 'inspect'],
            'animal': ['pet', 'feed', 'ride', 'hug', 'walk', 'train', 'groom'],
            'plant': ['pick', 'cut', 'inspect', 'move'],
            'sports': ['throw', 'catch', 'kick', 'hit', 'hold', 'swing', 'dribble', 'race'],
            'tool': ['hold', 'cut_with', 'repair', 'wield', 'assemble', 'make'],
            'structure': ['paint', 'clean', 'inspect', 'repair', 'stand_on'],
            'toy': ['hug', 'hold', 'carry', 'pet'],
            'human': ['hug', 'push', 'pull', 'talk_on', 'teach', 'train', 'watch'],
            'container': ['hold', 'move', 'clean', 'fill'],
            'accessory': ['wear', 'hold', 'open', 'pack', 'zip']
        }

        
        category_map = {
            'airplane': 'vehicle', 'apple': 'food', 'backpack': 'personal_item', 'banana': 'food',
            'baseball_bat': 'sports', 'baseball_glove': 'sports', 'bear': 'animal', 'bed': 'furniture',
            'bench': 'furniture', 'bicycle': 'vehicle', 'bird': 'animal', 'boat': 'vehicle',
            'book': 'personal_item', 'bottle': 'kitchenware', 'bowl': 'kitchenware', 'broccoli': 'food',
            'bus': 'vehicle', 'cake': 'food', 'car': 'vehicle', 'carrot': 'food',
            'cat': 'animal', 'cell_phone': 'electronic', 'chair': 'furniture', 'clock': 'structure',
            'couch': 'furniture', 'cow': 'animal', 'cup': 'kitchenware', 'dining_table': 'furniture',
            'dog': 'animal', 'donut': 'food', 'elephant': 'animal', 'fire_hydrant': 'structure',
            'fork': 'kitchenware', 'frisbee': 'sports', 'giraffe': 'animal', 'hair_drier': 'appliance',
            'handbag': 'personal_item', 'horse': 'animal', 'hot_dog': 'food', 'keyboard': 'electronic',
            'kite': 'sports', 'knife': 'tool', 'laptop': 'electronic', 'microwave': 'appliance',
            'motorcycle': 'vehicle', 'mouse': 'electronic', 'orange': 'food', 'oven': 'appliance',
            'parking_meter': 'structure', 'person': 'human', 'pizza': 'food', 'potted_plant': 'plant',
            'refrigerator': 'appliance', 'remote': 'electronic', 'sandwich': 'food', 'scissors': 'tool',
            'sheep': 'animal', 'sink': 'bathroom', 'skateboard': 'sports', 'skis': 'sports',
            'snowboard': 'sports', 'spoon': 'kitchenware', 'sports_ball': 'sports', 'stop_sign': 'structure',
            'suitcase': 'personal_item', 'surfboard': 'sports', 'teddy_bear': 'toy', 'tennis_racket': 'sports',
            'tie': 'accessory', 'toaster': 'appliance', 'toilet': 'bathroom', 'toothbrush': 'bathroom',
            'traffic_light': 'structure', 'train': 'vehicle', 'truck': 'vehicle', 'tv': 'electronic',
            'umbrella': 'accessory', 'vase': 'container', 'wine_glass': 'kitchenware', 'zebra': 'animal'
        }
        
        
        category = category_map.get(object_class, None)
        if category:
            specific_verbs = verb_groups.get(category, [])
        else:
            specific_verbs = []


        category_map = {
            "airplane": [
                "board", "direct", "exit", "fly", "inspect", "load",
                "ride", "sit_on", "wash", "no_interaction"
            ],
            "bicycle": [
                "carry", "hold", "inspect", "jump", "hop_on", "park",
                "push", "repair", "ride", "sit_on", "straddle",
                "walk", "wash", "no_interaction"
            ],
            "bird": [
                "chase", "feed", "hold", "pet", "release",
                "watch", "no_interaction"
            ],
            "boat": [
                "board", "drive", "exit", "inspect", "jump", "launch",
                "repair", "ride", "row", "sail", "sit_on", "stand_on",
                "tie", "wash", "no_interaction"
            ],
            "bottle": [
                "carry", "drink_with", "hold", "inspect",
                "lick", "open", "pour", "no_interaction"
            ],
            "bus": [
                "board", "direct", "drive", "exit", "inspect",
                "load", "ride", "sit_on", "wash", "wave",
                "no_interaction"
            ],
            "car": [
                "board", "direct", "drive", "hose", "inspect",
                "jump", "load", "park", "ride", "wash",
                "no_interaction"
            ],
            "cat": [
                "dry", "feed", "hold", "hug", "kiss",
                "pet", "scratch", "wash", "chase", "no_interaction"
            ],
            "chair": [
                "carry", "hold", "lie_on", "sit_on",
                "stand_on", "no_interaction"
            ],
            "couch": [
                "carry", "lie_on", "sit_on", "no_interaction"
            ],
            "cow": [
                "feed", "herd", "hold", "hug", "kiss",
                "lasso", "milk", "pet", "ride", "walk",
                "no_interaction"
            ],
            "dining_table": [
                "clean", "eat_at", "sit_at", "no_interaction"
            ],
            "dog": [
                "carry", "dry", "feed", "groom", "hold", "hose",
                "hug", "inspect", "kiss", "pet", "run", "scratch",
                "straddle", "train", "walk", "wash", "chase",
                "no_interaction"
            ],
            "horse": [
                "feed", "groom", "hold", "hug", "jump",
                "kiss", "load", "hop_on", "pet", "race",
                "ride", "run", "straddle", "train", "walk",
                "wash", "no_interaction"
            ],
            "motorcycle": [
                "hold", "inspect", "jump", "hop_on", "park",
                "push", "race", "ride", "sit_on", "straddle",
                "turn", "walk", "wash", "no_interaction"
            ],
            "person": [
                "carry", "greet", "hold", "hug", "kiss",
                "stab", "tag", "teach", "lick", "no_interaction"
            ],
            "potted_plant": [
                "carry", "hold", "hose", "no_interaction"
            ],
            "sheep": [
                "carry", "feed", "herd", "hold", "hug",
                "kiss", "pet", "ride", "shear", "walk",
                "wash", "no_interaction"
            ],
            "train": [
                "board", "drive", "exit", "load", "ride",
                "sit_on", "wash", "no_interaction"
            ],
            "tv": [
                "control", "repair", "watch", "no_interaction"
            ],
            "apple": [
                "buy", "cut", "eat", "hold", "inspect",
                "peel", "pick", "smell", "wash", "no_interaction"
            ],
            "backpack": [
                "carry", "hold", "inspect", "open",
                "wear", "no_interaction"
            ],
            "banana": [
                "buy", "carry", "cut", "eat", "hold",
                "inspect", "peel", "pick", "smell",
                "no_interaction"
            ],
            "baseball_bat": [
                "break", "carry", "hold", "sign",
                "swing", "throw", "wield", "no_interaction"
            ],
            "baseball_glove": [
                "hold", "wear", "no_interaction"
            ],
            "bear": [
                "feed", "hunt", "watch", "no_interaction"
            ],
            "bed": [
                "clean", "lie_on", "sit_on", "no_interaction"
            ],
            "bench": [
                "inspect", "lie_on", "sit_on", "no_interaction"
            ],
            "book": [
                "carry", "hold", "open", "read", "no_interaction"
            ],
            "bowl": [
                "hold", "stir", "wash", "lick", "no_interaction"
            ],
            "broccoli": [
                "cut", "eat", "hold", "smell",
                "stir", "wash", "no_interaction"
            ],
            "cake": [
                "blow", "carry", "cut", "eat", "hold",
                "light", "make", "pick_up", "no_interaction"
            ],
            "carrot": [
                "carry", "cook", "cut", "eat", "hold",
                "peel", "smell", "stir", "wash",
                "no_interaction"
            ],
            "cell_phone": [
                "carry", "hold", "read", "repair",
                "talk_on", "text_on", "no_interaction"
            ],
            "clock": [
                "check", "hold", "repair", "set",
                "no_interaction"
            ],
            "cup": [
                "carry", "drink_with", "hold", "inspect",
                "pour", "sip", "smell", "fill", "wash",
                "no_interaction"
            ],
            "donut": [
                "buy", "carry", "eat", "hold",
                "make", "pick_up", "smell", "no_interaction"
            ],
            "elephant": [
                "feed", "hold", "hose", "hug", "kiss",
                "hop_on", "pet", "ride", "walk",
                "wash", "watch", "no_interaction"
            ],
            "fire_hydrant": [
                "hug", "inspect", "open", "paint",
                "no_interaction"
            ],
            "fork": [
                "hold", "lift", "stick", "lick",
                "wash", "no_interaction"
            ],
            "frisbee": [
                "block", "catch", "hold", "spin",
                "throw", "no_interaction"
            ],
            "giraffe": [
                "feed", "kiss", "pet", "ride",
                "watch", "no_interaction"
            ],
            "hair_drier": [
                "hold", "operate", "repair", "no_interaction"
            ],
            "handbag": [
                "carry", "hold", "inspect", "no_interaction"
            ],
            "hot_dog": [
                "carry", "cook", "cut", "eat",
                "hold", "make", "no_interaction"
            ],
            "keyboard": [
                "carry", "clean", "hold", "type_on",
                "no_interaction"
            ],
            "kite": [
                "assemble", "carry", "fly", "hold",
                "inspect", "launch", "pull", "no_interaction"
            ],
            "knife": [
                "cut_with", "hold", "stick", "wash",
                "wield", "lick", "no_interaction"
            ],
            "laptop": [
                "hold", "open", "read", "repair",
                "type_on", "no_interaction"
            ],
            "microwave": [
                "clean", "open", "operate", "no_interaction"
            ],
            "mouse": [
                "control", "hold", "repair", "no_interaction"
            ],
            "orange": [
                "buy", "cut", "eat", "hold", "inspect",
                "peel", "pick", "squeeze", "wash",
                "no_interaction"
            ],
            "oven": [
                "clean", "hold", "inspect", "open",
                "repair", "operate", "no_interaction"
            ],
            "parking_meter": [
                "check", "pay", "repair", "no_interaction"
            ],
            "pizza": [
                "buy", "carry", "cook", "cut", "eat",
                "hold", "make", "pick_up", "slide",
                "smell", "no_interaction"
            ],
            "refrigerator": [
                "clean", "hold", "move", "open",
                "no_interaction"
            ],
            "remote": [
                "hold", "point", "swing", "no_interaction"
            ],
            "sandwich": [
                "carry", "cook", "cut", "eat",
                "hold", "make", "no_interaction"
            ],
            "scissors": [
                "cut_with", "hold", "open", "no_interaction"
            ],
            "sink": [
                "clean", "repair", "wash", "no_interaction"
            ],
            "skateboard": [
                "carry", "flip", "grind", "hold",
                "jump", "pick_up", "ride", "sit_on",
                "stand_on", "no_interaction"
            ],
            "skis": [
                "adjust", "carry", "hold", "inspect",
                "jump", "pick_up", "repair", "ride",
                "stand_on", "wear", "no_interaction"
            ],
            "snowboard": [
                "adjust", "carry", "grind", "hold",
                "jump", "ride", "stand_on", "wear",
                "no_interaction"
            ],
            "spoon": [
                "hold", "lick", "wash", "sip",
                "no_interaction"
            ],
            "sports_ball": [
                "block", "carry", "catch", "dribble",
                "hit", "hold", "inspect", "kick",
                "pick_up", "serve", "sign", "spin",
                "throw", "no_interaction"
            ],
            "stop_sign": [
                "hold", "stand_under", "stop_at",
                "no_interaction"
            ],
            "suitcase": [
                "carry", "drag", "hold", "hug",
                "load", "open", "pack", "pick_up",
                "zip", "no_interaction"
            ],
            "surfboard": [
                "carry", "drag", "hold", "inspect",
                "jump", "lie_on", "load", "ride",
                "stand_on", "sit_on", "wash",
                "no_interaction"
            ],
            "teddy_bear": [
                "carry", "hold", "hug", "kiss",
                "no_interaction"
            ],
            "tennis_racket": [
                "carry", "hold", "inspect", "swing",
                "no_interaction"
            ],
            "tie": [
                "adjust", "cut", "hold", "inspect",
                "pull", "tie", "wear", "no_interaction"
            ],
            "toaster": [
                "hold", "operate", "repair", "no_interaction"
            ],
            "toilet": [
                "clean", "flush", "open", "repair",
                "sit_on", "stand_on", "wash",
                "no_interaction"
            ],
            "toothbrush": [
                "brush_with", "hold", "wash",
                "no_interaction"
            ],
            "traffic_light": [
                "install", "repair", "stand_under",
                "stop_at", "no_interaction"
            ],
            "truck": [
                "direct", "drive", "inspect", "load",
                "repair", "ride", "sit_on", "wash",
                "no_interaction"
            ],
            "umbrella": [
                "carry", "hold", "lose", "open",
                "repair", "set", "stand_under",
                "no_interaction"
            ],
            "vase": [
                "hold", "make", "paint", "no_interaction"
            ],
            "wine_glass": [
                "fill", "hold", "sip", "toast",
                "lick", "wash", "no_interaction"
            ],
            "zebra": [
                "feed", "hold", "pet", "watch",
                "no_interaction"
            ]
        }

        
        specific_verbs = category_map.get(object_class, [])
        
        
        
        

        if specific_verbs == []:
            
            general_verbs = ['hold', 'carry', 'point', 'inspect', 'move', 'watch']
            
            
            
            all_verbs = list(set(specific_verbs + general_verbs))
        else:
            all_verbs = specific_verbs
        
        return all_verbs[:20]  



class VisualAnalystAgent(BaseAgent):
    """视觉分析Agent - 验证视觉证据"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.rule_generator = LLMRuleGenerator()
        self.action_categories = get_action_categories()
        print("初始化 Visual Analyst Agent...")
    
    def analyze(self, image_path: str, proposal: HOIInstance, 
                debate_history: List = None) -> AgentArgument:
        """分析视觉证据"""
        image = cv2.imread(image_path) if isinstance(image_path, str) else image_path
        
        
        contact_analysis = self._analyze_contact(image, proposal.human_bbox, proposal.object_bbox)
        spatial_score = self._analyze_spatial(proposal.human_bbox, proposal.object_bbox)
        color_consistency = self._analyze_color_consistency(image, proposal.human_bbox, proposal.object_bbox)
        motion_blur = self._detect_motion_blur(image, proposal.human_bbox)
        
        
        action_specific_score = self._action_specific_analysis(
            proposal.verb, proposal.object_class, contact_analysis['score'], spatial_score
        )
        
        
        visual_score = (action_specific_score * 0.4 + 
                       contact_analysis['score'] * 0.3 + 
                       spatial_score * 0.2 + 
                       color_consistency * 0.1)
        
        
        if debate_history and len(debate_history) > 0:
            history_factor = self._consider_history(debate_history, proposal)
            visual_score *= history_factor
        
        
        if visual_score > 0.6:
            stance = 'support'
        elif visual_score < 0.4:
            stance = 'oppose'
        else:
            stance = 'neutral'
        
        reasoning = self._generate_reasoning(
            proposal, contact_analysis, spatial_score, visual_score, debate_history
        )
        
        return AgentArgument(
            agent_name='visual_analyst',
            proposal_id=id(proposal),
            stance=stance,
            evidence={
                'contact': contact_analysis,
                'spatial_score': spatial_score,
                'color_consistency': color_consistency,
                'motion_blur': motion_blur,
                'visual_score': visual_score
            },
            confidence=visual_score,
            reasoning=reasoning,
            response_to=self._get_response_target(debate_history)
        )
    
    def _action_specific_analysis(self, verb, object_class, contact_score, spatial_score):
        """基于完整动作分类的分析"""
        rules = self.rule_generator.generate_interaction_rules(verb, object_class)
        base_score = rules['plausibility']
        
        
        if verb in self.action_categories['contact_required']:
            
            final_score = base_score * 0.3 + contact_score * 0.7
        elif verb in self.action_categories['distance_required']:
            
            final_score = base_score * 0.3 + (1 - contact_score) * 0.3 + spatial_score * 0.4
        else:
            
            final_score = base_score * 0.4 + spatial_score * 0.6
        
        return final_score
    
    def _analyze_contact(self, image, human_bbox, object_bbox):
        """分析接触关系"""
        x1 = max(human_bbox[0], object_bbox[0])
        y1 = max(human_bbox[1], object_bbox[1])
        x2 = min(human_bbox[2], object_bbox[2])
        y2 = min(human_bbox[3], object_bbox[3])
        
        if x2 > x1 and y2 > y1:
            overlap_area = (x2 - x1) * (y2 - y1)
            human_area = (human_bbox[2] - human_bbox[0]) * (human_bbox[3] - human_bbox[1])
            overlap_ratio = overlap_area / human_area
            return {'contact': True, 'score': min(1.0, overlap_ratio * 2)}
        else:
            
            h_cx = (human_bbox[0] + human_bbox[2]) / 2
            h_cy = (human_bbox[1] + human_bbox[3]) / 2
            o_cx = (object_bbox[0] + object_bbox[2]) / 2
            o_cy = (object_bbox[1] + object_bbox[3]) / 2
            
            distance = np.sqrt((h_cx - o_cx)**2 + (h_cy - o_cy)**2)
            if image is not None and hasattr(image, 'shape'):
                max_distance = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
                proximity_score = 1.0 - (distance / max_distance)
            else:
                proximity_score = 0.5
            
            return {'contact': False, 'score': proximity_score * 0.5}
    
    def _analyze_spatial(self, human_bbox, object_bbox):
        """分析空间关系"""
        h_cx = (human_bbox[0] + human_bbox[2]) / 2
        h_cy = (human_bbox[1] + human_bbox[3]) / 2
        o_cx = (object_bbox[0] + object_bbox[2]) / 2
        o_cy = (object_bbox[1] + object_bbox[3]) / 2
        
        distance = np.sqrt((h_cx - o_cx)**2 + (h_cy - o_cy)**2)
        return np.exp(-distance / 200)
    
    def _analyze_color_consistency(self, image, human_bbox, object_bbox):
        """分析颜色一致性"""
        if image is None or not hasattr(image, 'shape'):
            return 0.5
        
        h, w = image.shape[:2]
        
        
        h_x1 = int(max(0, min(human_bbox[0], w-1)))
        h_y1 = int(max(0, min(human_bbox[1], h-1)))
        h_x2 = int(max(h_x1+1, min(human_bbox[2], w)))
        h_y2 = int(max(h_y1+1, min(human_bbox[3], h)))
        
        o_x1 = int(max(0, min(object_bbox[0], w-1)))
        o_y1 = int(max(0, min(object_bbox[1], h-1)))
        o_x2 = int(max(o_x1+1, min(object_bbox[2], w)))
        o_y2 = int(max(o_y1+1, min(object_bbox[3], h)))
        
        try:
            h_region = image[h_y1:h_y2, h_x1:h_x2]
            o_region = image[o_y1:o_y2, o_x1:o_x2]
            
            if h_region.size == 0 or o_region.size == 0:
                return 0.5
            
            
            hist_h = cv2.calcHist([h_region], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            hist_o = cv2.calcHist([o_region], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_o = cv2.normalize(hist_o, hist_o).flatten()
            
            
            similarity = cv2.compareHist(hist_h, hist_o, cv2.HISTCMP_CORREL)
            
            return (similarity + 1) / 2
        except:
            return 0.5
    
    def _detect_motion_blur(self, image, bbox):
        """检测运动模糊"""
        if image is None or not hasattr(image, 'shape'):
            return 0
        
        h, w = image.shape[:2]
        
        x1 = int(max(0, min(bbox[0], w-1)))
        y1 = int(max(0, min(bbox[1], h-1)))
        x2 = int(max(x1+1, min(bbox[2], w)))
        y2 = int(max(y1+1, min(bbox[3], h)))
        
        try:
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return 0
            
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            
            blur_score = 1.0 / (1.0 + fm / 100)
            
            return blur_score
        except:
            return 0
    
    def _consider_history(self, debate_history, proposal):
        """考虑辩论历史"""
        support_count = 0
        oppose_count = 0
        
        for round_args in debate_history:
            for arg in round_args:
                if arg.proposal_id == id(proposal):
                    if arg.stance == 'support':
                        support_count += 1
                    elif arg.stance == 'oppose':
                        oppose_count += 1
        
        if support_count > oppose_count * 2:
            return 1.1
        elif oppose_count > support_count * 2:
            return 0.9
        return 1.0
    
    def _generate_reasoning(self, proposal, contact_analysis, spatial_score, 
                           final_score, debate_history):
        """生成推理说明"""
        reasoning = f"Visual evidence for '{proposal.verb} + {proposal.object_class}': "
        
        if proposal.verb in self.action_categories['contact_required']:
            reasoning += f"Contact-required action, "
        elif proposal.verb in self.action_categories['distance_required']:
            reasoning += f"Distance-required action, "
        else:
            reasoning += f"Flexible-distance action, "
        
        if contact_analysis['contact']:
            reasoning += "strong contact detected. "
        elif contact_analysis['score'] > 0.3:
            reasoning += "partial proximity detected. "
        else:
            reasoning += "no direct contact. "
        
        if debate_history and len(debate_history) > 0:
            reasoning += f"After {len(debate_history)} rounds of debate, "
        
        reasoning += f"final visual confidence: {final_score:.2f}"
        
        return reasoning
    
    def _get_response_target(self, debate_history):
        """确定回应目标"""
        if not debate_history or len(debate_history) == 0:
            return None
        
        last_round = debate_history[-1]
        for arg in last_round:
            if arg.stance == 'oppose':
                return arg.agent_name
        return None



class SemanticExpertAgent(BaseAgent):
    """语义专家Agent - 常识推理"""
    
    def __init__(self):
        print("初始化 Semantic Expert Agent...")
        self.rule_generator = LLMRuleGenerator()
    
    def analyze(self, image_path: str, proposal: HOIInstance, 
                debate_history: List = None) -> AgentArgument:
        """语义分析"""
        
        rules = self.rule_generator.generate_interaction_rules(
            proposal.verb, proposal.object_class
        )
        
        
        if debate_history:
            consensus_factor = self._analyze_consensus(debate_history, proposal)
            rules['plausibility'] = rules['plausibility'] * 0.7 + consensus_factor * 0.3
        
        
        if rules['plausibility'] > 0.6:
            stance = 'support'
        elif rules['plausibility'] < 0.3:
            stance = 'oppose'
        else:
            stance = 'neutral'
        
        reasoning = self._generate_reasoning(proposal, rules, debate_history)
        
        return AgentArgument(
            agent_name='semantic_expert',
            proposal_id=id(proposal),
            stance=stance,
            evidence=rules,
            confidence=rules['plausibility'],
            reasoning=reasoning,
            response_to=self._get_response_target(debate_history)
        )
    
    def _analyze_consensus(self, debate_history, proposal):
        """分析共识程度"""
        scores = []
        for round_args in debate_history:
            for arg in round_args:
                if arg.proposal_id == id(proposal):
                    scores.append(arg.confidence)
        
        if scores:
            return np.mean(scores)
        return 0.5
    
    def _generate_reasoning(self, proposal, rules, debate_history):
        """生成推理"""
        reasoning = f"Semantic analysis for '{proposal.verb} + {proposal.object_class}': "
        
        if rules['source'] == 'llm':
            reasoning += f"LLM analysis suggests plausibility={rules['plausibility']:.2f}. "
        else:
            reasoning += f"Rule-based analysis: "
        
        if rules['common_sense']:
            reasoning += "Common sense supports this interaction. "
        else:
            reasoning += "Uncommon/unusual interaction. "
        
        if rules['physical_possible']:
            reasoning += "Physically possible. "
        else:
            reasoning += "Physical constraints exist. "
        
        if debate_history:
            consensus = self._analyze_consensus(debate_history, proposal)
            reasoning += f"Current consensus level: {consensus:.2f}"
        
        return reasoning
    
    def _get_response_target(self, debate_history):
        if not debate_history:
            return None
        
        last_round = debate_history[-1]
        for arg in last_round:
            if arg.stance != 'neutral':
                return arg.agent_name
        return None



class PhysicsAuditorAgent(BaseAgent):
    """物理审计Agent - 检查物理合理性"""
    
    def __init__(self):
        print("初始化 Physics Auditor Agent...")
        self.rule_generator = LLMRuleGenerator()
    
    def analyze(self, image_path: str, proposal: HOIInstance,
                debate_history: List = None) -> AgentArgument:
        """物理分析"""
        
        physics_score = analyze_physics_constraints_complete(proposal)
        
        
        rules = self.rule_generator.generate_interaction_rules(
            proposal.verb, proposal.object_class
        )
        
        
        final_score = rules['physical_possible'] * 0.4 + physics_score * 0.6
        
        
        if debate_history:
            history_influence = self._consider_debate(debate_history, proposal)
            final_score = final_score * 0.8 + history_influence * 0.2
        
        
        if final_score > 0.6:
            stance = 'support'
        elif final_score < 0.4:
            stance = 'oppose'
        else:
            stance = 'neutral'
        
        reasoning = self._generate_reasoning(proposal, physics_score, final_score, debate_history)
        
        return AgentArgument(
            agent_name='physics_auditor',
            proposal_id=id(proposal),
            stance=stance,
            evidence={
                'physics_score': physics_score,
                'physical_possible': rules['physical_possible'],
                'final_score': final_score
            },
            confidence=final_score,
            reasoning=reasoning,
            response_to=self._get_response_target(debate_history)
        )
    
    def _consider_debate(self, debate_history, proposal):
        """考虑辩论历史"""
        visual_scores = []
        for round_args in debate_history:
            for arg in round_args:
                if arg.proposal_id == id(proposal) and arg.agent_name == 'visual_analyst':
                    visual_scores.append(arg.confidence)
        
        if visual_scores:
            return np.mean(visual_scores)
        return 0.5
    
    def _generate_reasoning(self, proposal, physics_score, final_score, debate_history):
        """生成推理"""
        reasoning = f"Physics check for '{proposal.verb} + {proposal.object_class}': "
        reasoning += f"Detailed physics score: {physics_score:.2f}. "
        
        if physics_score > 0.6:
            reasoning += "Physically plausible based on spatial constraints. "
        else:
            reasoning += "Physical constraints detected (size/position/reachability). "
        
        if debate_history:
            reasoning += f"Considering {len(debate_history)} rounds of discussion. "
        
        reasoning += f"Final physics confidence: {final_score:.2f}"
        
        return reasoning
    
    def _get_response_target(self, debate_history):
        if not debate_history:
            return None
        return 'visual_analyst'



class JudgeAgent:
    """裁判Agent - 综合决策"""
    
    def __init__(self):
        print("初始化 Judge Agent...")
        self.agent_weights = {
            'visual_analyst': 0.35,
            'semantic_expert': 0.35,
            'physics_auditor': 0.30
        }
    
    def evaluate(self, arguments: List[AgentArgument]) -> Dict:
        """评估所有论据"""
        grouped = defaultdict(list)
        for arg in arguments:
            grouped[arg.proposal_id].append(arg)
        
        decisions = {}
        for proposal_id, args in grouped.items():
            decision = self._make_decision(args)
            decisions[proposal_id] = decision
        
        return decisions
    
    def _make_decision(self, arguments: List[AgentArgument]) -> Dict:
        """做出决策"""
        weighted_score = 0
        total_weight = 0
        votes = {'support': 0, 'oppose': 0, 'neutral': 0}
        
        for arg in arguments:
            weight = self.agent_weights.get(arg.agent_name, 0.25)
            votes[arg.stance] += 1
            
            if arg.stance == 'support':
                weighted_score += arg.confidence * weight
            elif arg.stance == 'oppose':
                weighted_score += (1 - arg.confidence) * weight * 0.5
            else:
                weighted_score += 0.5 * weight
            
            total_weight += weight
        
        final_score = weighted_score / (total_weight + 1e-6)
        
        
        if final_score > 0.5 and votes['support'] >= votes['oppose']:
            verdict = 'accept'
        elif final_score < 0.3 or votes['oppose'] > votes['support'] * 1.5:
            verdict = 'reject'
        else:
            verdict = 'uncertain'
        
        return {
            'verdict': verdict,
            'confidence': final_score,
            'votes': votes,
            'num_rounds': len(arguments) // 3
        }



class DebateCoordinator:
    """辩论协调器"""
    
    def __init__(self):
        self.max_rounds = 3
        self.consensus_threshold = 0.7
    
    def conduct_debate(self, agents: Dict, image_path: str, 
                       proposals: List[HOIInstance]) -> List[HOIInstance]:
        """组织辩论"""
        final_proposals = []
        
        for proposal in proposals[:20]:  
            debate_history = []
            
            for round_num in range(self.max_rounds):
                round_arguments = []
                
                
                for agent_name, agent in agents.items():
                    if agent_name == 'judge':
                        continue
                    
                    
                    argument = agent.analyze(image_path, proposal, debate_history)
                    round_arguments.append(argument)
                
                debate_history.append(round_arguments)
                
                
                if self._check_consensus(round_arguments):
                    break
            
            
            all_arguments = [arg for round_args in debate_history for arg in round_args]
            decision = agents['judge'].evaluate(all_arguments)
            
            if decision[id(proposal)]['verdict'] == 'accept':
                proposal.confidence = decision[id(proposal)]['confidence']
                proposal.debate_history = debate_history
                final_proposals.append(proposal)
        
        return final_proposals
    
    def _check_consensus(self, arguments):
        """检查是否达成共识"""
        if not arguments:
            return False
        
        stances = [arg.stance for arg in arguments]
        most_common = max(set(stances), key=stances.count)
        return stances.count(most_common) / len(stances) >= self.consensus_threshold



class HOIEvaluator:
    """HOI检测的mAP评估器"""
    
    def __init__(self, verb_classes=None, object_classes=None, iou_threshold=0.5):
        self.verb_classes = verb_classes or VERB_CLASSES
        self.object_classes = object_classes or OBJECT_CLASSES
        self.iou_threshold = iou_threshold
        self.hoi_categories = self._build_hoi_categories()
    
    def _build_hoi_categories(self):
        """构建所有可能的HOI类别组合"""
        categories = []
        for verb in self.verb_classes:
            for obj in self.object_classes:
                categories.append(f"{verb}_{obj}")
        return categories
    
    def compute_map(self, predictions, ground_truths):
        """计算mAP"""
        
        ap_scores = {}
        
        for verb in self.verb_classes:
            for obj in self.object_classes:
                hoi_class = f"{verb}_{obj}"
                
                
                class_predictions = []
                class_ground_truths = []
                
                
                for img_id, preds in predictions.items():
                    for pred in preds:
                        if hasattr(pred, 'verb'):  
                            pred_verb = pred.verb
                            pred_obj = pred.object_class
                            if pred_verb == verb and pred_obj == obj:
                                class_predictions.append({
                                    'image_id': img_id,
                                    'human_bbox': pred.human_bbox,
                                    'object_bbox': pred.object_bbox,
                                    'score': pred.confidence
                                })
                        else:  
                            if pred.get('verb') == verb and pred.get('object_class') == obj:
                                class_predictions.append({
                                    'image_id': img_id,
                                    'human_bbox': pred['human_bbox'],
                                    'object_bbox': pred['object_bbox'],
                                    'score': pred.get('confidence', 0.5)
                                })
                
                
                for img_id, gts in ground_truths.items():
                    for gt in gts:
                        gt_verb = gt.get('verb', '')
                        gt_obj = gt.get('object_class', '')
                        
                        if gt_verb == verb and gt_obj == obj:
                            class_ground_truths.append({
                                'image_id': img_id,
                                'human_bbox': gt['human_bbox'],
                                'object_bbox': gt['object_bbox']
                            })
                
                
                if len(class_ground_truths) > 0:
                    ap = self._compute_ap(class_predictions, class_ground_truths)
                    if ap > 0:
                        ap_scores[hoi_class] = ap
        
        
        if ap_scores:
            mAP = np.mean(list(ap_scores.values()))
        else:
            mAP = 0.0
        
        
        verb_ap = defaultdict(list)
        object_ap = defaultdict(list)
        
        for hoi_class, ap in ap_scores.items():
            parts = hoi_class.split('_', 1)
            if len(parts) == 2:
                verb, obj = parts
                verb_ap[verb].append(ap)
                object_ap[obj].append(ap)
        
        verb_map = {v: np.mean(aps) for v, aps in verb_ap.items() if aps}
        object_map = {o: np.mean(aps) for o, aps in object_ap.items() if aps}
        
        return {
            'mAP': mAP,
            'num_classes_evaluated': len(ap_scores),
            'verb_mAP': verb_map,
            'object_mAP': object_map,
            'per_class_ap': ap_scores
        }
    
    def _compute_ap(self, predictions, ground_truths):
        """计算单个HOI类别的Average Precision"""
        if not predictions:
            return 0.0
        
        
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        
        tp = np.zeros(len(predictions))
        fp = np.zeros(len(predictions))
        
        
        gt_matched = defaultdict(set)
        
        for pred_idx, pred in enumerate(predictions):
            pred_img = pred['image_id']
            
            
            img_gts = [gt for gt in ground_truths if gt['image_id'] == pred_img]
            
            if not img_gts:
                fp[pred_idx] = 1
                continue
            
            
            best_match_idx = -1
            best_match_score = 0
            
            for gt_idx, gt in enumerate(img_gts):
                
                if gt_idx in gt_matched[pred_img]:
                    continue
                
                
                human_iou = self._compute_iou(pred['human_bbox'], gt['human_bbox'])
                object_iou = self._compute_iou(pred['object_bbox'], gt['object_bbox'])
                hoi_iou = min(human_iou, object_iou)
                
                if hoi_iou > best_match_score:
                    best_match_score = hoi_iou
                    best_match_idx = gt_idx
            
            
            if best_match_score >= self.iou_threshold:
                tp[pred_idx] = 1
                gt_matched[pred_img].add(best_match_idx)
            else:
                fp[pred_idx] = 1
        
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(ground_truths)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        
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



class MultiAgentHOISystem:
    """完整的多Agent HOI检测系统"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print("=" * 60)
        print("初始化多Agent HOI检测系统")
        print("=" * 60)
        
        self.device = device
        
        
        self.detector = ObjectDetector(device)
        
        
        self.proposal_agent = ProposalAgent(device)
        
        
        self.agents = {
            'visual': VisualAnalystAgent(device),
            'semantic': SemanticExpertAgent(),
            'physics': PhysicsAuditorAgent(),
            'judge': JudgeAgent()
        }
        
        
        self.coordinator = DebateCoordinator()
        
        
        self.evaluator = HOIEvaluator()
        
        
        self.action_categories = get_action_categories()
        
        print("系统初始化完成！")
        print(f"- 支持{len(VERB_CLASSES)}个动作")
        print(f"- 支持{len(OBJECT_CLASSES)}个物体类别")
        print(f"- 设备: {device}")
        print("=" * 60)
    
    def detect_hoi(self, image_path: str) -> List[HOIInstance]:
        """检测HOI - 完整流程"""
        print(f"\n处理图像: {image_path}")
        
        
        print("步骤1: 检测人和物体...")
        detections = self.detector.detect(image_path)
        print(f"  检测到 {len(detections['humans'])} 个人, {len(detections['objects'])} 个物体")
        
        if not detections['humans'] or not detections['objects']:
            print("  未检测到人或物体，跳过")
            return []
        
        
        print("步骤2: 生成HOI提议...")
        proposals = self.proposal_agent.propose(image_path, detections)
        print(f"  生成了 {len(proposals)} 个初始提议")
        
        if not proposals:
            return []
        
        
        print("步骤3: 多Agent辩论...")
        final_hois = self.coordinator.conduct_debate(
            self.agents, image_path, proposals
        )
        
        print(f"\n最终保留 {len(final_hois)} 个HOI")
        
        return final_hois
    
    def evaluate_map(self, predictions: Dict, ground_truths: Dict) -> Dict:
        """计算mAP"""
        return self.evaluator.compute_map(predictions, ground_truths)


def test_system(image_path: str, system):
    """测试系统"""
    print("\n" + "=" * 60)
    print("HOI检测系统测试")
    print("=" * 60)

    results = system.detect_hoi(image_path)
    
    print("\n检测结果:")
    print("-" * 40)
    
    if results:
        for i, hoi in enumerate(results[:200], 1):
            print(f"\n[{i}] {hoi.verb} + {hoi.object_class}")
            print(f"    置信度: {hoi.confidence:.3f}")
            print(f"    人框: {[f'{x:.1f}' for x in hoi.human_bbox]}")
            print(f"    物框: {[f'{x:.1f}' for x in hoi.object_bbox]}")
            
            if hoi.debate_history:
                print(f"    辩论轮数: {len(hoi.debate_history)}")
                
                last_round = hoi.debate_history[-1]
                stances = {arg.agent_name: arg.stance for arg in last_round}
                print(f"    最终立场: {stances}")
    else:
        print("未检测到HOI")
    
    results_new = {}
    results_new[os.path.basename(image_path)] = results

    return results_new


def test_on_hico_det(test_json_path: str, image_dir: str, output_path: str = None):
    """在HICO-DET测试集上测试"""
    print("加载测试数据...")
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)
    
    system = MultiAgentHOISystem()
    from hoi_complete_module import HOIEvaluator

    
    evaluator = HOIEvaluator()
       
    all_predictions = {}
    all_ground_truth = {}
    
    
    for idx, sample in enumerate(test_data):  
        image_name = sample['file_name']
        image_path = f"{image_dir}/{image_name}"
        
        print(f"\n[{idx+1}/{len(test_data)}] 处理图像: {image_name}")

        pre_result = test_system(image_path, system=system)
        for k, v in pre_result.items():
            all_predictions[k] = v
         
        gt_hois = []
        for hoi in sample.get('hoi_annotation', []):
            subject_bbox = sample['annotations'][hoi['subject_id']]['bbox']
            object_bbox = sample['annotations'][hoi['object_id']]['bbox']
            object_category = sample['annotations'][hoi['object_id']]['category_id']
            
            
            hoi_category_id = hoi['category_id']
            
            
            verb_idx = hoi_category_id % len(VERB_CLASSES) - 1
            
            gt_hois.append({
                'human_bbox': subject_bbox,
                'object_bbox': object_bbox,
                'object_class': COCO_TO_OUR_CLASSES.get(object_category, 'object'),
                'verb': VERB_CLASSES[verb_idx]
            })
        
        all_ground_truth[image_name] = gt_hois
    
    
    print("\n计算mAP...")
    
    
    results = evaluator.compute_map(all_predictions, all_ground_truth)
    print(f"mAP: {results['mAP']:.4f}")


    print(f"\n评估结果:")
    print(f"mAP: {results['mAP']:.4f}")
    
    
    
    if output_path:
        output = {
            'mAP': results['mAP'],
            'predictions': {
                img_name: [
                    {
                        'human_bbox': pred.human_bbox,
                        'object_bbox': pred.object_bbox,
                        'object_class': pred.object_class,
                        'verb': pred.verb,
                        'confidence': pred.confidence
                    }
                    for pred in preds
                ]
                for img_name, preds in all_predictions.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"结果已保存到: {output_path}")
    
    return results




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Agent HOI Detection System')
    parser.add_argument('--test_json', type=str, required=True,
                       help='Path to test json file')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Path to save results')
    
    args = parser.parse_args()
    
    
    results = test_on_hico_det(
        test_json_path=args.test_json,
        image_dir=args.image_dir,
        output_path=args.output
    )
    
    print("\n测试完成！")