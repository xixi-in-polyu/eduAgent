"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { PDFDocumentLoadingTask } from "pdfjs-dist";
import { FileText, Hash, Loader2, AlertCircle, Download, ChevronLeft, ChevronRight } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { isOfficeMaterialFileType } from "@/lib/material-office";
import { captureScrollViewportToPngFile, EDU_CHAT_ADD_ATTACHMENT_EVENT } from "@/lib/captureElementToPngFile";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import {
  markdownRehypePlugins,
  markdownRemarkPlugins,
  normalizeMathDelimiters,
} from "@/lib/markdownMath";
import { markdownComponents } from "@/lib/markdownComponents";

type MaterialDetail = {
  id: string;
  filename: string;
  file_type: string;
  status: string;
  preview_pdf_status: "NA" | "PENDING" | "READY" | "FAILED";
  indexed_chunk_count: number;
  created_at: string;
  status_message: string | null;
  transcript: string | null;
  video_summary: string | null;
};

type Props = {
  /** Reserved for future scoped URLs; optional. */
  courseId?: string;
  materialId: string | null;
  chunkId?: string;
  sourceLabel?: string;
  /** Base path for material API calls, e.g. "/api/v1/me/materials". Defaults to "/api/v1/materials". */
  apiBase?: string;
  /**
   * Called whenever the capture function changes. Pass a function to capture the current
   * page as a File, or null when no capturable content is available (material unloaded, audio).
   * Used by the chat component to silently attach a page snapshot on message send.
   */
  onCaptureFnChange?: (fn: (() => Promise<File | null>) | null) => void;
};

const VIDEO_EXTENSIONS = new Set(["mp4", "webm", "mov", "mkv", "avi", "m4v", "wmv"]);
const AUDIO_EXTENSIONS = new Set(["mp3", "wav", "m4a", "flac", "ogg", "opus"]);

function isVideoType(ft: string) { return VIDEO_EXTENSIONS.has(ft.toLowerCase()); }
function isAudioType(ft: string) { return AUDIO_EXTENSIONS.has(ft.toLowerCase()); }

export default function CourseMaterialViewer({
  materialId,
  chunkId,
  sourceLabel,
  apiBase = "/api/v1/materials",
  onCaptureFnChange,
}: Props) {
  const [material, setMaterial] = useState<MaterialDetail | null>(null);
  const [chunkError, setChunkError] = useState<string | null>(null);
  const [textBody, setTextBody] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [captureBusy, setCaptureBusy] = useState(false);
  const [videoDataLoaded, setVideoDataLoaded] = useState(false);
  const [pdfViewportReady, setPdfViewportReady] = useState(false);
  const [pdfLoadError, setPdfLoadError] = useState<string | null>(null);

  // PDF canvas renderer state
  const [pdfPageNum, setPdfPageNum] = useState(1);
  const [pdfTotalPages, setPdfTotalPages] = useState(0);
  const [pdfRenderBusy, setPdfRenderBusy] = useState(false);
  const [pdfLoadProgress, setPdfLoadProgress] = useState<number | null>(null);
  const pdfCanvasRef = useRef<HTMLCanvasElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pdfDocRef = useRef<any>(null);

  const textScrollRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const loadMaterial = useCallback(async () => {
    if (!materialId) {
      setMaterial(null);
      setTextBody(null);
      setChunkError(null);
      return;
    }
    setLoading(true);
    try {
      const mRes = await fetch(`${apiBase}/${materialId}`, {
        credentials: "include",
      });
      if (mRes.ok) {
        setMaterial((await mRes.json()) as MaterialDetail);
      } else {
        setMaterial(null);
      }
    } finally {
      setLoading(false);
    }
  }, [materialId, apiBase]);

  useEffect(() => {
    setVideoDataLoaded(false);
  }, [materialId]);

  useEffect(() => {
    void loadMaterial();
  }, [loadMaterial]);

  const office = material ? isOfficeMaterialFileType(material.file_type) : false;
  const pollPreview =
    !!material &&
    office &&
    material.preview_pdf_status === "PENDING";

  // Poll while a video/audio file is still being transcribed (no transcript yet)
  const pollTranscribe =
    !!material &&
    (isVideoType(material.file_type) || isAudioType(material.file_type)) &&
    !["FAILED"].includes(material.status) &&
    !(material.transcript || material.video_summary);

  useEffect(() => {
    if (!pollPreview && !pollTranscribe) return;
    const t = setInterval(() => void loadMaterial(), 2500);
    return () => clearInterval(t);
  }, [pollPreview, pollTranscribe, loadMaterial]);

  useEffect(() => {
    if (!materialId || !material) {
      setTextBody(null);
      return;
    }
    const ft = material.file_type.toLowerCase();
    if (ft !== "md" && ft !== "txt") {
      setTextBody(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      const res = await fetch(`${apiBase}/${materialId}/content`, {
        credentials: "include",
      });
      if (!res.ok || cancelled) return;
      const t = await res.text();
      if (!cancelled) setTextBody(t);
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [materialId, material?.file_type, material?.id, apiBase]);

  useEffect(() => {
    setPdfViewportReady(false);
    setPdfPageNum(1);
    setPdfTotalPages(0);
    setPdfLoadProgress(null);
    setPdfLoadError(null);
    pdfDocRef.current = null;
  }, [materialId]);

  // Render PDF page onto canvas using pdfjs-dist
  const renderPdfPage = useCallback(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    async (doc: any, pageNum: number) => {
      const canvas = pdfCanvasRef.current;
      if (!canvas) return;
      setPdfRenderBusy(true);
      try {
        const page = await doc.getPage(pageNum);
        const viewport = page.getViewport({ scale: 1.5 });
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        await page.render({ canvasContext: ctx, viewport }).promise;
      } finally {
        setPdfRenderBusy(false);
      }
    },
    [],
  );

  // Load PDF document when material is a native PDF or an Office file with a ready preview
  useEffect(() => {
    if (!materialId || !material) return;
    const ft = material.file_type.toLowerCase();
    const isOfficeReady =
      isOfficeMaterialFileType(material.file_type) &&
      material.preview_pdf_status === "READY";
    if (ft !== "pdf" && !isOfficeReady) return;
    let cancelled = false;
    let loadingTask: PDFDocumentLoadingTask | null = null;
    void (async () => {
      try {
        setPdfLoadError(null);
        // Dynamic import to avoid SSR issues
        const pdfjsLib = await import("pdfjs-dist/legacy/build/pdf.mjs");
        pdfjsLib.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs?v=legacy-6.3.289";
        setPdfLoadProgress(0);
        loadingTask = pdfjsLib.getDocument({
          url: `${apiBase}/${materialId}/content`,
          withCredentials: true,
          rangeChunkSize: 65536,
          disableStream: false,
          disableRange: false,
        });
        loadingTask.onProgress = (data: { loaded: number; total: number }) => {
          if (data.total > 0 && !cancelled) {
            setPdfLoadProgress(Math.round((data.loaded / data.total) * 100));
          }
        };
        const doc = await loadingTask.promise;
        if (cancelled) return;
        pdfDocRef.current = doc;
        setPdfTotalPages(doc.numPages);
        setPdfPageNum(1);
        await renderPdfPage(doc, 1);
        if (!cancelled) {
          setPdfLoadProgress(null);
          setPdfViewportReady(true);
        }
      } catch (e) {
        if (!cancelled) {
          setPdfLoadProgress(null);
          setPdfViewportReady(false);
          const msg = e instanceof Error ? e.message : "未知错误";
          setPdfLoadError(`PDF 预览加载失败: ${msg}`);
        }
      }
    })();
    return () => {
      cancelled = true;
      void loadingTask?.destroy();
      pdfDocRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [materialId, material?.file_type, material?.preview_pdf_status, material?.id, renderPdfPage, apiBase]);

  // Re-render when page number changes
  useEffect(() => {
    if (!pdfDocRef.current || pdfPageNum < 1) return;
    void renderPdfPage(pdfDocRef.current, pdfPageNum);
  }, [pdfPageNum, renderPdfPage]);

  useEffect(() => {
    if (!materialId || !chunkId) {
      setChunkError(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      const cRes = await fetch(
        `/api/v1/materials/${materialId}/chunks/${chunkId}`,
        { credentials: "include" },
      );
      if (cancelled) return;
      if (cRes.status === 501) {
        setChunkError("引用片段暂不支持从服务端拉取全文块，请查看下方资料预览。");
        return;
      }
      if (!cRes.ok) {
        setChunkError("无法加载引用片段");
        return;
      }
      setChunkError(null);
    })();
    return () => {
      cancelled = true;
    };
  }, [materialId, chunkId]);

  // Derived variables — computed before early returns so that Hooks below are
  // always called unconditionally.  When `material` is null the early returns
  // below will fire first, so the default values here are never used in the JSX.
  const ft = material?.file_type.toLowerCase() ?? "";
  const showPdfCanvas =
    ft === "pdf" || (!!office && material?.preview_pdf_status === "READY");
  const previewFailed =
    !!office && material?.preview_pdf_status === "FAILED";
  const previewPending =
    !!office && material?.preview_pdf_status === "PENDING";
  const previewPendingText =
    previewPending && material?.status === "READY"
      ? "索引已完成，正在同步 PDF 预览状态…"
      : "正在生成 PDF 预览…";

  const textPreviewReady =
    (ft === "md" || ft === "txt") && textBody !== null;

  const isVideo = isVideoType(ft);
  const isAudio = isAudioType(ft);

  const hasMediaContent =
    (isVideo || isAudio) &&
    material?.status === "READY" &&
    !!(material?.transcript || material?.video_summary);

  // Safe base name for screenshot filenames (strip chars invalid in filenames)
  const safeScreenshotBase = (material?.filename ?? "资料")
    .replace(/\.[^/.]+$/, "")           // remove extension
    .replace(/[\\/:*?"<>|]/g, "_")      // replace FS-unsafe chars
    .slice(0, 60);                       // cap length

  // Screenshot availability per type
  const canScreenshot =
    !captureBusy &&
    !loading &&
    !previewPending &&
    (
      textPreviewReady ||
      (showPdfCanvas && pdfViewportReady) ||
      (isVideo && videoDataLoaded) ||
      previewFailed
    ) &&
  !isAudio;

  /**
   * Capture the current page/frame as a File without side-effects.
   * Returns null if the material is not in a capturable state (audio, loading, etc.).
   */
  const captureCurrentPageFile = useCallback(async (): Promise<File | null> => {
    if (!canScreenshot) return null;
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    if (showPdfCanvas && pdfCanvasRef.current) {
      const canvas = pdfCanvasRef.current;
      const blob = await new Promise<Blob>((resolve, reject) => {
        canvas.toBlob((b) => b ? resolve(b) : reject(new Error("canvas.toBlob failed")), "image/png");
      });
      return new File([blob], `资料截图-${safeScreenshotBase}-${stamp}.png`, { type: "image/png" });
    } else if (isVideo && videoRef.current) {
      const video = videoRef.current;
      const tmpCanvas = document.createElement("canvas");
      tmpCanvas.width = video.videoWidth || 640;
      tmpCanvas.height = video.videoHeight || 360;
      const ctx = tmpCanvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);
      const blob = await new Promise<Blob>((resolve, reject) => {
        tmpCanvas.toBlob((b) => b ? resolve(b) : reject(new Error("canvas.toBlob failed")), "image/png");
      });
      return new File([blob], `资料截图-${safeScreenshotBase}-${stamp}.png`, { type: "image/png" });
    } else if (textScrollRef.current) {
      return captureScrollViewportToPngFile(
        textScrollRef.current,
        `资料截图-${safeScreenshotBase}-${stamp}.png`,
      );
    }
    return null;
  }, [canScreenshot, showPdfCanvas, isVideo, safeScreenshotBase]);

  // Notify parent whenever the capture function becomes available or unavailable.
  useEffect(() => {
    onCaptureFnChange?.(canScreenshot ? captureCurrentPageFile : null);
    return () => onCaptureFnChange?.(null);
  // captureCurrentPageFile is stable when canScreenshot deps are unchanged
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canScreenshot, captureCurrentPageFile]);

  if (!materialId) {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[120px] py-8 text-muted-foreground gap-2 px-3 text-center">
        <FileText size={26} />
        <p className="text-xs">选择资料即可预览</p>
      </div>
    );
  }

  if (loading && !material) {
    return (
      <div className="p-3 space-y-2">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/2" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  if (!material) {
    return (
      <div className="p-3 text-xs text-destructive">无法加载该资料</div>
    );
  }

  const handleScreenshotToChat = async () => {
    if (!canScreenshot) return;
    setCaptureBusy(true);
    try {
      const file = await captureCurrentPageFile();
      if (!file) return;
      window.dispatchEvent(
        new CustomEvent(EDU_CHAT_ADD_ATTACHMENT_EVENT, {
          detail: { file },
        }),
      );
    } catch (e) {
      alert(e instanceof Error ? e.message : "截屏失败");
    } finally {
      setCaptureBusy(false);
    }
  };

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      <div className="shrink-0 px-3 py-2 border-b border-border space-y-1">
        <div className="flex items-start gap-2">
          <FileText size={14} className="text-muted-foreground mt-0.5 shrink-0" />
          <div className="min-w-0">
            <p className="text-xs font-semibold leading-tight truncate">
              {material.filename}
            </p>
            <p className="text-[10px] text-muted-foreground uppercase font-mono">
              {material.file_type}
            </p>
          </div>
        </div>
        {sourceLabel && (
          <p className="text-[10px] text-muted-foreground line-clamp-2">
            <span className="font-medium text-foreground">引用：</span>
            {sourceLabel}
          </p>
        )}
        <div className="flex gap-2 items-stretch">
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="inline-flex shrink-0">
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="h-7 w-7"
                  disabled={!canScreenshot}
                  onClick={() => void handleScreenshotToChat()}
                  aria-label="上传资料页面作为消息附件"
                >
                  {captureBusy ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden
                    >
                      <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
                      <circle cx="12" cy="13" r="3" />
                    </svg>
                  )}
                </Button>
              </span>
            </TooltipTrigger>
            <TooltipContent>
              {isAudio
                ? "音频无法截屏"
                : isVideo && !videoDataLoaded
                  ? "请先播放或拖动进度条以加载视频帧"
                  : "上传资料页面作为消息附件"}
            </TooltipContent>
          </Tooltip>
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-[11px] flex-1 min-w-0"
            asChild
          >
            <a
              href={`${apiBase}/${materialId}/content?variant=original`}
              download={material.filename}
            >
              <Download size={12} className="mr-1 shrink-0" />
              下载原文件
            </a>
          </Button>
        </div>
      </div>

      <div
        ref={showPdfCanvas || isVideo || isAudio ? undefined : textScrollRef}
        className={
          showPdfCanvas || hasMediaContent
            ? "flex-1 min-h-0 flex flex-col overflow-hidden"
            : "flex-1 min-h-0 overflow-auto"
        }
      >
        {chunkId && chunkError && (
          <div className="mx-3 mt-2 rounded-lg border border-dashed border-border bg-muted/30 px-2 py-1.5 text-[10px] text-muted-foreground">
            {chunkError}
          </div>
        )}

        {previewPending && (
          <div className="flex items-center gap-2 m-3 text-xs text-muted-foreground">
            <Loader2 size={14} className="animate-spin" />
            {previewPendingText}
          </div>
        )}

        {previewFailed && (
          <div className="m-3 flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-2 text-[11px] text-destructive">
            <AlertCircle size={14} className="shrink-0 mt-0.5" />
            <span>预览转换失败，请下载原文件查看。</span>
          </div>
        )}

        {/* PDF (native or Office preview) → pdfjs-dist canvas */}
        {showPdfCanvas && (
          <div className="flex-1 min-h-0 flex flex-col overflow-hidden">
            {pdfLoadError && (
              <div className="m-3 flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-2 text-[11px] text-destructive">
                <AlertCircle size={14} className="shrink-0 mt-0.5" />
                <span>{pdfLoadError}</span>
              </div>
            )}
            {!pdfViewportReady && (
              <div className="flex flex-col gap-2 m-3">
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 size={14} className="animate-spin" />
                  {pdfLoadProgress !== null
                    ? `正在加载 PDF… ${pdfLoadProgress}%`
                    : "正在加载 PDF…"}
                </div>
                {pdfLoadProgress !== null && pdfLoadProgress < 100 && (
                  <div className="h-1 w-full rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full bg-primary rounded-full transition-all duration-200"
                      style={{ width: `${pdfLoadProgress}%` }}
                    />
                  </div>
                )}
              </div>
            )}
            {pdfViewportReady && pdfTotalPages > 1 && (
              <div className="shrink-0 flex items-center justify-center gap-2 px-3 py-1 border-b border-border">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  disabled={pdfPageNum <= 1 || pdfRenderBusy}
                  onClick={() => setPdfPageNum((p) => Math.max(1, p - 1))}
                  aria-label="上一页"
                >
                  <ChevronLeft size={14} />
                </Button>
                <span className="text-[11px] text-muted-foreground tabular-nums">
                  {pdfPageNum} / {pdfTotalPages}
                </span>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6"
                  disabled={pdfPageNum >= pdfTotalPages || pdfRenderBusy}
                  onClick={() => setPdfPageNum((p) => Math.min(pdfTotalPages, p + 1))}
                  aria-label="下一页"
                >
                  <ChevronRight size={14} />
                </Button>
              </div>
            )}
            <div className="flex-1 min-h-0 overflow-auto flex justify-center p-2">
              {pdfRenderBusy && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <Loader2 size={14} className="animate-spin text-muted-foreground" />
                </div>
              )}
              <canvas
                ref={pdfCanvasRef}
                className="max-w-full h-auto shadow-sm rounded"
                style={{ display: pdfViewportReady ? "block" : "none" }}
              />
            </div>
          </div>
        )}

        {/* Video player */}
        {isVideo && (
          <div className={hasMediaContent ? "shrink-0 flex justify-center p-3 bg-black/5 border-b border-border" : "flex-1 min-h-0 flex flex-col items-center justify-center p-3 gap-2"}>
            <video
              ref={videoRef}
              src={`${apiBase}/${materialId}/content`}
              controls
              crossOrigin="anonymous"
              className="max-w-full rounded shadow-sm"
              style={hasMediaContent ? { maxHeight: "min(50vh, 320px)" } : { maxHeight: "calc(100% - 8px)" }}
              preload="metadata"
              onLoadedData={() => setVideoDataLoaded(true)}
            />
          </div>
        )}

        {/* Audio player */}
        {isAudio && (
          <div className={hasMediaContent ? "shrink-0 px-4 py-3 border-b border-border" : "flex items-center justify-center p-4"}>
            <audio
              src={`${apiBase}/${materialId}/content`}
              controls
              className="w-full"
              preload="metadata"
            />
          </div>
        )}

        {/* Transcript / summary panel for video and audio */}
        {(isVideo || isAudio) && hasMediaContent && (
          <div className="flex-1 min-h-0 flex flex-col">
            <Tabs
              defaultValue={material.video_summary ? "summary" : "transcript"}
              className="flex flex-col flex-1 min-h-0"
            >
              <div className="shrink-0 px-3 pt-2 pb-1">
                <TabsList className="h-7">
                  {material.video_summary && (
                    <TabsTrigger value="summary" className="h-5 px-2.5 text-[11px]">摘要</TabsTrigger>
                  )}
                  {material.transcript && (
                    <TabsTrigger value="transcript" className="h-5 px-2.5 text-[11px]">转录文本</TabsTrigger>
                  )}
                </TabsList>
              </div>
              {material.video_summary && (
                <TabsContent value="summary" className="flex-1 min-h-0 overflow-auto mt-0 px-3 pb-3">
                  <div className="prose prose-sm dark:prose-invert max-w-none pt-1">
                    <ReactMarkdown
                      remarkPlugins={markdownRemarkPlugins}
                      rehypePlugins={markdownRehypePlugins}
                      components={markdownComponents}
                    >
                      {normalizeMathDelimiters(material.video_summary)}
                    </ReactMarkdown>
                  </div>
                </TabsContent>
              )}
              {material.transcript && (
                <TabsContent value="transcript" className="flex-1 min-h-0 overflow-auto mt-0 px-3 pb-3 pt-1">
                  <pre className="whitespace-pre-wrap text-[11px] font-mono text-muted-foreground leading-relaxed">
                    {material.transcript}
                  </pre>
                </TabsContent>
              )}
            </Tabs>
          </div>
        )}

        {(ft === "md" || ft === "txt") && textBody !== null && (
          <div className="p-3 prose prose-sm dark:prose-invert max-w-none">
            {ft === "md" ? (
              <ReactMarkdown
                remarkPlugins={markdownRemarkPlugins}
                rehypePlugins={markdownRehypePlugins}
                components={markdownComponents}
              >
                {normalizeMathDelimiters(textBody)}
              </ReactMarkdown>
            ) : (
              <pre className="whitespace-pre-wrap text-xs font-mono bg-muted/30 p-3 rounded-lg">
                {textBody}
              </pre>
            )}
          </div>
        )}

        {(ft === "md" || ft === "txt") && textBody === null && !loading && (
          <div className="p-3 flex items-center gap-2 text-xs text-muted-foreground">
            <Loader2 size={14} className="animate-spin" />
            加载文本…
          </div>
        )}

        {chunkId && !chunkError && (
          <div className="px-3 py-2 text-[10px] text-muted-foreground flex items-center gap-1">
            <Hash size={10} />
            <span className="font-mono truncate">{chunkId}</span>
          </div>
        )}
      </div>
    </div>
  );
}
