"""
prompt_layer_test.py — 对比不同 system prompt 层注入对 LLM 回答风格的影响。

用法:
    python scripts/prompt_layer_test.py

可扩展: 在 LAYER_CONFIGS 列表末尾追加新的 LayerConfig 即可测试更多层
(如: 学习者画像层、记忆摘要层、课程模式层等)。
"""

import os
import sys
import pathlib
import textwrap
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 依赖: openai (pip install openai python-dotenv)
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai not installed. Run: pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("WARNING: python-dotenv not installed; will rely on shell env vars only.")
    load_dotenv = None  # type: ignore

# ---------------------------------------------------------------------------
# 加载环境变量 — 优先读取 edu-platform/.env, 其次读取项目根 .env
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).parent.parent

def _load_env() -> None:
    if load_dotenv is None:
        return
    # edu-platform/.env 包含 LLM_CHAT_MODEL 等 Next.js 平台使用的配置
    for env_path in [
        REPO_ROOT / "edu-platform" / ".env",
        REPO_ROOT / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            print(f"[env] loaded {env_path.relative_to(REPO_ROOT)}")

_load_env()

def _e(name: str, fallback: str = "") -> str:
    return (os.getenv(name) or "").strip() or fallback

# 复现 llm-registry.ts 里 "chat" role 的 fallback 逻辑
API_KEY  = _e("LLM_CHAT_API_KEY") or _e("LLM_API_KEY")
BASE_URL = _e("LLM_CHAT_BASE_URL") or _e("LLM_BASE_URL") or None
MODEL    = _e("LLM_CHAT_MODEL") or _e("LLM_MODEL") or "deepseek-flash"

print(f"[llm]  base_url={BASE_URL or '(openai default)'}")
print(f"[llm]  model={MODEL}")
print()

if not API_KEY:
    print("ERROR: No API key found. Set LLM_API_KEY or LLM_CHAT_API_KEY.")
    sys.exit(1)

CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL if BASE_URL else None)

# ---------------------------------------------------------------------------
# 素材: 从 skills/ 目录读取真实的苏格拉底技能内容
# ---------------------------------------------------------------------------
def _read_skill_body(path: pathlib.Path) -> str:
    """解析 frontmatter 并返回 body 内容。"""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()

SKILLS_DIR = REPO_ROOT / "skills"
SOCRATIC_BODY = _read_skill_body(SKILLS_DIR / "socratic.md")

# ---------------------------------------------------------------------------
# 复现 prompt-builder.ts 里的核心 prompt 片段
# ---------------------------------------------------------------------------
DEFAULT_PERSONA = """\
# 角色：智能教学助手

你是一位耐心、专业、富有启发性的 AI 教学助手。你的目标是帮助学习者深入理解知识，培养独立思考能力，而不仅仅是提供答案。

## 教学原则
- **以学习者为中心**：根据学习者的水平调整语言难度和解释深度。
- **启发引导**：尽量通过提问引导学习者自己得出结论，而非直接给出答案。
- **及时反馈**：对学习者的回答给予具体、积极的反馈，指出不足时保持鼓励性语气。
- **学科准确性**：确保所有知识性内容准确；不确定时如实说明并提示查阅权威来源。\
"""

SKILLS_INDEX_BLOCK = """\
<available_skills>
- **socratic**: 苏格拉底式提问引导学习者自主推理，而非直接给出答案
</available_skills>\
"""

SOCRATIC_INJECT_BLOCK = f"## 教学策略：socratic\n{SOCRATIC_BODY}"

# ---------------------------------------------------------------------------
# 层级配置
# ---------------------------------------------------------------------------
@dataclass
class LayerConfig:
    name: str
    description: str
    # 每个 block 按顺序拼接到 system prompt
    blocks: list[str] = field(default_factory=list)

    def build_system_prompt(self) -> str:
        return "\n\n".join(b.strip() for b in self.blocks if b.strip())


LAYER_CONFIGS: list[LayerConfig] = [
    LayerConfig(
        name="no_persona",
        description="无 persona，裸模型 (无任何 system prompt 内容)",
        blocks=["你是一个助手。"],
    ),
    LayerConfig(
        name="base_only",
        description="仅基础教学 persona，无任何技能注入",
        blocks=[DEFAULT_PERSONA],
    ),
    LayerConfig(
        name="with_skills_index",
        description="Persona + Socratic 技能仅作为索引列表 (alwaysInject=false, 默认行为)",
        blocks=[DEFAULT_PERSONA, SKILLS_INDEX_BLOCK],
    ),
    LayerConfig(
        name="with_socratic_injected",
        description="Persona + Socratic 技能**直接注入** (模拟: alwaysInject=true)",
        blocks=[DEFAULT_PERSONA, SOCRATIC_INJECT_BLOCK],
    ),
    # ---- 在此处添加更多层 ------------------------------------------------
    # LayerConfig(
    #     name="with_learner_profile",
    #     description="Persona + Socratic 注入 + 学习者画像",
    #     blocks=[DEFAULT_PERSONA, SOCRATIC_INJECT_BLOCK, LEARNER_PROFILE_BLOCK],
    # ),
]

# ---------------------------------------------------------------------------
# 测试问题集 — 代表性问题: 概念性/原理性问题，最能体现苏格拉底风格差异
# ---------------------------------------------------------------------------
TEST_QUESTIONS: list[str] = [
    "TCP 为什么需要三次握手？两次不够吗？",
]

# ---------------------------------------------------------------------------
# 核心: 调用 LLM 并返回回答
# ---------------------------------------------------------------------------
def call_llm(system_prompt: str, user_message: str) -> str:
    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=600,
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# 辅助: 打印输出
# ---------------------------------------------------------------------------
DIVIDER = "=" * 72

def print_block(title: str, content: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)
    print(textwrap.fill(content, width=70, subsequent_indent="  ") if len(content) < 500
          else content)


def run_tests() -> None:
    print(f"\n{'#' * 72}")
    print("  Prompt Layer Test — 苏格拉底式技能注入风格对比")
    print(f"{'#' * 72}")

    for question in TEST_QUESTIONS:
        print(f"\n\n{'*' * 72}")
        print(f"  提问: {question}")
        print(f"{'*' * 72}")

        results: dict[str, str] = {}

        for layer in LAYER_CONFIGS:
            print(f"\n[层级 {layer.name}] {layer.description}")
            sys_prompt = layer.build_system_prompt()
            answer = call_llm(sys_prompt, question)
            results[layer.name] = answer
            print_block(f"回答 [{layer.name}]", answer)

        # ---- 差异摘要 ----
        print(f"\n{DIVIDER}")
        print("  差异摘要 (字数)")
        print(DIVIDER)
        for name, ans in results.items():
            q_count = ans.count("？") + ans.count("?")
            print(f"  {name:30s}  {len(ans):4d} 字  问句数: {q_count}")

    print(f"\n\n{DIVIDER}")
    print("  测试完成")
    print(DIVIDER)


if __name__ == "__main__":
    run_tests()
