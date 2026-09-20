import { describe, it, expect } from "vitest";
import { PromptBuilder } from "@/lib/agent/prompt-builder";
import type { SkillEntry } from "@/lib/agent/skills-loader";

const basePrompt = "你是课堂助教";
const skills: SkillEntry[] = [
  {
    name: "Socratic",
    description: "通过提问启发",
    version: "1.0.0",
    body: "先问后答",
    alwaysInject: true,
    subFiles: {},
    source: "built-in",
  },
  {
    name: "Assignment",
    description: "作业拆解",
    version: "1.0.0",
    body: "分解任务",
    alwaysInject: false,
    subFiles: {},
    source: "built-in",
  },
];

describe("PromptBuilder", () => {
  it("业务规则：课程模式下应包含课程检索约束与安全准则", () => {
    // given
    const builder = new PromptBuilder();

    // when
    const prompt = builder.buildSystemPrompt(
      basePrompt,
      skills,
      "- TCP（掌握度 0.70）",
      { userId: "u-1", profile: { name: "小明", learning_style: "图示" } },
      {
        userId: "u-1",
        sessionId: "s-1",
        accessibleCourseIds: ["c-1"],
        courseId: "c-1",
      },
    );

    // then
    expect(prompt).toContain("Current Session: Course Knowledge Base Mode");
    expect(prompt).toContain("教学策略：Socratic");
    expect(prompt).toContain("<available_skills>");
    expect(prompt).toContain("Assignment");
    expect(prompt).toContain("学习者画像");
    expect(prompt).toContain("姓名：小明");
    expect(prompt).toContain("已知掌握情况");
    expect(prompt).toContain("Safety Guidelines");
    expect(prompt).toContain("Tool Usage Guidelines");
    expect(prompt.indexOf("Tool Usage Guidelines")).toBeLessThan(
      prompt.indexOf("学习者画像"),
    );
    expect(prompt.indexOf("Tool Usage Guidelines")).toBeLessThan(
      prompt.indexOf("已知掌握情况"),
    );
  });

  it("业务规则：问答中心模式下应提示跨课程检索策略", () => {
    // given
    const builder = new PromptBuilder();

    // when
    const prompt = builder.buildSystemPrompt(
      basePrompt,
      skills,
      "",
      null,
      {
        userId: "u-1",
        sessionId: "s-1",
        accessibleCourseIds: ["c-1", "c-2"],
        courseId: null,
      },
    );

    // then
    expect(prompt).toContain("Current Session: Q&A Center");
    expect(prompt).not.toContain("已知掌握情况（近期记忆）");
    expect(prompt).not.toContain("姓名：");
  });
});
