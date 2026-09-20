/** Standalone process that dispatches timezone-aware daily memory reviews. */

const baseUrl = process.env.NEXTJS_INTERNAL_URL ?? "http://nextjs:3000";
const dispatchUrl = `${baseUrl}/api/v1/internal/memory-review/dispatch`;
const internalKey = process.env.INTERNAL_API_KEY ?? "";
const tickIntervalMs = 60_000;

if (internalKey.length < 16) {
  throw new Error("INTERNAL_API_KEY must be set and at least 16 characters");
}

async function tick() {
  try {
    const response = await fetch(dispatchUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-internal-key": internalKey,
      },
      body: "{}",
      signal: AbortSignal.timeout(30_000),
    });
    if (!response.ok) {
      throw new Error(`dispatch returned HTTP ${response.status}`);
    }
    const data = await response.json();
    if (data.dispatched > 0) {
      console.log(
        `[review-scheduler] ${new Date().toISOString()} ` +
          `dispatched=${data.dispatched} skipped=${data.skipped}`,
      );
    }
  } catch (error) {
    console.error(
      "[review-scheduler] tick error:",
      error instanceof Error ? error.message : error,
    );
  }
}

console.log(`[review-scheduler] starting, dispatch URL: ${dispatchUrl}`);
await tick();
setInterval(() => void tick(), tickIntervalMs);
