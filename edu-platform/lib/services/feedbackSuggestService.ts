import { getLLMClient, getRoleConfig } from "@/lib/agent/llm-registry";
import { createStandaloneTrace, flushLangfuse, recordGeneration } from "@/lib/agent/tracing/langfuse-tracer";
import type { SuggestFeedbackBody } from "@/lib/dto/submission.dto";

/**
 * Build the FIM prompt: system context + teacher prefix concatenated into a
 * single string so the model continues naturally after the last character.
 */
function buildFimPrompt(body: SuggestFeedbackBody): string {
  return (
    `[批改上下文]\n` +
    `题目：${body.questionText}\n` +
    `学生答案：${body.studentAnswer}\n` +
    `得分：${body.score}/${body.maxScore}\n\n` +
    `[教师评语]\n` +
    (body.prefix ?? "")
  );
}

/**
 * Generate an AI continuation suggestion for teacher feedback text.
 * Uses the 'completion' role (deepseek-flash + beta FIM endpoint by default)
 * so the model directly continues the teacher's partial comment without
 * rephrasing or wrapping in extra tags.
 */
export async function suggestFeedback(
  body: SuggestFeedbackBody,
): Promise<{ suggestion: string }> {
  const config = getRoleConfig("completion");
  const client = getLLMClient("completion");

  const prompt = buildFimPrompt(body);

  const trace = createStandaloneTrace({
    name: "teaching.suggest_feedback",
    metadata: { model: config.model, mode: "fim" },
  });

  // Use the OpenAI-compatible /completions endpoint (FIM / text-completion).
  // deepseek-flash on https://api.deepseek.com/beta supports this natively.
  const resp = await client.completions.create({
    model: config.model,
    prompt,
    max_tokens: 80,
    temperature: 0.3,
    stop: ["\n\n", "["],
  });

  const suggestion = resp.choices[0]?.text?.trim() ?? "";

  recordGeneration(trace, {
    name: "suggest_feedback_fim",
    model: config.model,
    input: [{ role: "user", content: prompt }],
    output: suggestion,
    usage: {
      promptTokens: resp.usage?.prompt_tokens,
      completionTokens: resp.usage?.completion_tokens,
      totalTokens: resp.usage?.total_tokens,
    },
  });
  void flushLangfuse();

  return { suggestion };
}
