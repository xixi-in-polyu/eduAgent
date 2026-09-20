import type { NextRequest } from "next/server";
import { PutObjectCommand } from "@aws-sdk/client-s3";
import { requireAuthenticated } from "@/lib/admin";
import { getAuthFromRequest } from "@/lib/request-auth";
import { getMinioPresignedUrl, getS3Client } from "@/lib/minio";
import { getMinioConfig, isCosEnabled } from "@/lib/config";
import { getCosPresignedUrl, putCosObject } from "@/lib/cos";
import { ApiError } from "@/lib/http/api-error";
import { jsonError, jsonOk } from "@/lib/http/json-response";

export const dynamic = "force-dynamic";

const ALLOWED_MIME_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
  "application/pdf",
  "text/plain",
  "text/markdown",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/vnd.ms-powerpoint",
  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  "application/vnd.ms-excel",
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
]);

/** Normalize code-file MIME types that vary across browsers to text/plain. */
const MIME_NORMALIZE: Record<string, string> = {
  "text/x-python": "text/plain",
  "application/x-python": "text/plain",
  "application/x-python-code": "text/plain",
  "text/x-javascript": "text/plain",
  "application/javascript": "text/plain",
  "text/javascript": "text/plain",
  "text/typescript": "text/plain",
  "text/x-typescript": "text/plain",
};

const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB
const PRESIGN_TTL_SECONDS = 3600; // 1 hour

function sanitizeFilename(name: string): string {
  return name.replace(/[^a-zA-Z0-9._\-\u4e00-\u9fa5]/g, "_").slice(0, 200);
}

export async function POST(req: NextRequest) {
  try {
    const auth = requireAuthenticated(await getAuthFromRequest(req));

    let formData: FormData;
    try {
      formData = await req.formData();
    } catch {
      throw new ApiError(400, "VALIDATION_ERROR", "Expected multipart/form-data");
    }

    const file = formData.get("file");
    if (!(file instanceof File)) {
      throw new ApiError(400, "VALIDATION_ERROR", "Missing 'file' field");
    }

    if (file.size === 0) {
      throw new ApiError(400, "VALIDATION_ERROR", "File is empty");
    }
    if (file.size > MAX_FILE_SIZE) {
      throw new ApiError(413, "FILE_TOO_LARGE", `File exceeds 20 MB limit`);
    }

    const rawMime = file.type || "application/octet-stream";
    const mimeType = MIME_NORMALIZE[rawMime] ?? rawMime;
    if (!ALLOWED_MIME_TYPES.has(mimeType)) {
      throw new ApiError(415, "UNSUPPORTED_MEDIA_TYPE", `File type '${mimeType}' is not allowed`);
    }

    const id = crypto.randomUUID();
    const safeName = sanitizeFilename(file.name || "attachment");
    const objectKey = `tmp-attachments/${auth.sub}/${id}-${safeName}`;

    const buffer = Buffer.from(await file.arrayBuffer());

    // ── Image uploads: use COS when configured (generates public HTTPS presigned URLs
    //    that are accessible by external vision models such as Qwen-VL). ──
    if (mimeType.startsWith("image/") && isCosEnabled()) {
      await putCosObject({ objectKey, body: buffer, contentType: mimeType });
      const presignedUrl = await getCosPresignedUrl(objectKey);
      return jsonOk({
        id,
        key: objectKey,
        presigned_url: presignedUrl,
        mime_type: mimeType,
        name: file.name || safeName,
        size: file.size,
      });
    }

    // ── All other uploads (or image uploads when COS is not configured): use MinIO ──
    const c = getMinioConfig();
    const client = getS3Client();

    await client.send(
      new PutObjectCommand({
        Bucket: c.bucket,
        Key: objectKey,
        Body: buffer,
        ContentType: mimeType,
        ContentLength: buffer.byteLength,
      }),
    );

    const presignedUrl = await getMinioPresignedUrl(
      objectKey,
      PRESIGN_TTL_SECONDS,
    );

    return jsonOk({
      id,
      key: objectKey,
      presigned_url: presignedUrl,
      mime_type: mimeType,
      name: file.name || safeName,
      size: file.size,
    });
  } catch (e) {
    if (e instanceof ApiError) return jsonError(e);
    return jsonError(new ApiError(500, "INTERNAL_ERROR", "Internal server error"));
  }
}
