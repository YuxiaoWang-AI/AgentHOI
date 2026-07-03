"""离线生成LLM规则缓存的脚本"""
import argparse
import json
import os
import sys
from typing import Dict

from hio_multi_agent_system import (
    VERB_CLASSES,
    OBJECT_CLASSES,
    LLMRuleGenerator,
    OFFLINE_RULE_CACHE_PATH,
)


def build_cache(output_path: str, resume: bool = True) -> None:
    """构建完整的<verb, object>规则缓存"""
    generator = LLMRuleGenerator()

    if not generator.use_llm:
        raise RuntimeError(
            "LLM模型未成功加载，无法生成离线缓存。"
        )

    cache: Dict[str, Dict[str, Dict]] = {}
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"检测到已有缓存，继续生成缺失项：{output_path}")

    total = len(VERB_CLASSES) * len(OBJECT_CLASSES)
    processed = 0

    for verb in VERB_CLASSES:
        if verb not in cache:
            cache[verb] = {}

        for object_class in OBJECT_CLASSES:
            processed += 1
            if object_class in cache[verb] and cache[verb][object_class]:
                continue

            try:
                rule = generator._generate_with_llm(verb, object_class)
                cache[verb][object_class] = rule
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                print(f"生成 {verb} + {object_class} 失败: {exc}")
                cache[verb][object_class] = generator._use_cached_rules(verb, object_class)
                cache[verb][object_class]["source"] = "fallback"
                status = "fallback"

            if processed % 50 == 0 or processed == total:
                print(f"进度: {processed}/{total} - 当前: {verb}+{object_class} ({status})")
                # break
        # break

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"已完成所有规则的离线生成，共 {total} 项。缓存保存至: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预生成HOI LLM规则缓存")
    parser.add_argument(
        "--output",
        default=OFFLINE_RULE_CACHE_PATH,
        help="缓存文件保存路径 (默认: offline_llm_rule_cache.json)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="若设置，则忽略已有缓存并重新生成",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        build_cache(args.output, resume=not args.no_resume)
    except Exception as exc:  # noqa: BLE001
        print(f"构建缓存失败: {exc}")
        sys.exit(1)