import {
  AbortMultipartUploadCommand,
  CompleteMultipartUploadCommand,
  CreateMultipartUploadCommand,
  DeleteObjectCommand,
  GetObjectCommand,
  HeadObjectCommand,
  PutObjectCommand,
  S3Client,
  UploadPartCommand,
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { NodeHttpHandler } from "@smithy/node-http-handler";
import { Buffer } from "node:buffer";
import type { Readable } from "node:stream";
import { getMinioConfig } from "@/lib/config";

const MAX_SINGLE_PUT_BYTES = 16 * 1024 * 1024;
const MULTIPART_PART_SIZE_BYTES = 8 * 1024 * 1024;

function buildClient(endpointOverride?: string): S3Client {
  const c = getMinioConfig();
  const endpoint = endpointOverride?.trim() || c.endpoint;
  const endpointUrl = endpoint.startsWith("http")
    ? endpoint
    : c.useSsl
      ? `https://${endpoint}`
      : `http://${endpoint}`;
  return new S3Client({
    region: c.region,
    endpoint: endpointUrl,
    forcePathStyle: true,
    credentials: {
      accessKeyId: c.accessKeyId,
      secretAccessKey: c.secretAccessKey,
    },
    // 大文件（视频/音频）上传耗时较长，设置宽松的 socket 超时避免 ECONNRESET
    requestHandler: new NodeHttpHandler({
      socketTimeout: 10 * 60 * 1000, // 10 分钟
      connectionTimeout: 10_000,      // 10 秒建连超时
    }),
  });
}

let _client: S3Client | null = null;
let _presignClient: S3Client | null = null;

export function getS3Client(): S3Client {
  if (!_client) _client = buildClient();
  return _client;
}

function getS3PresignClient(): S3Client {
  const publicEndpoint = process.env.MINIO_PUBLIC_ENDPOINT?.trim();
  if (!publicEndpoint) return getS3Client();
  if (!_presignClient) _presignClient = buildClient(publicEndpoint);
  return _presignClient;
}

export async function putObjectStream(params: {
  objectKey: string;
  body: Readable;
  contentLength?: number;
  contentType?: string;
}): Promise<void> {
  const c = getMinioConfig();
  const client = getS3Client();

  if (
    typeof params.contentLength === "number" &&
    params.contentLength <= MAX_SINGLE_PUT_BYTES
  ) {
    await client.send(
      new PutObjectCommand({
        Bucket: c.bucket,
        Key: params.objectKey,
        Body: params.body,
        ContentLength: params.contentLength,
        ContentType: params.contentType,
      }),
    );
    return;
  }

  const created = await client.send(
    new CreateMultipartUploadCommand({
      Bucket: c.bucket,
      Key: params.objectKey,
      ContentType: params.contentType,
    }),
  );
  const uploadId = created.UploadId;
  if (!uploadId) {
    throw new Error("S3 multipart upload initialization returned empty UploadId");
  }

  const completedParts: { ETag?: string; PartNumber?: number }[] = [];
  let partNumber = 1;

  try {
    for await (const part of splitReadableToParts(
      params.body,
      MULTIPART_PART_SIZE_BYTES,
    )) {
      const uploaded = await client.send(
        new UploadPartCommand({
          Bucket: c.bucket,
          Key: params.objectKey,
          UploadId: uploadId,
          PartNumber: partNumber,
          Body: part,
          ContentLength: part.length,
        }),
      );
      if (!uploaded.ETag) {
        throw new Error(`S3 multipart upload part ${partNumber} missing ETag`);
      }
      completedParts.push({ ETag: uploaded.ETag, PartNumber: partNumber });
      partNumber += 1;
    }

    if (completedParts.length === 0) {
      throw new Error("S3 multipart upload got empty body stream");
    }

    await client.send(
      new CompleteMultipartUploadCommand({
        Bucket: c.bucket,
        Key: params.objectKey,
        UploadId: uploadId,
        MultipartUpload: {
          Parts: completedParts,
        },
      }),
    );
  } catch (e) {
    await client
      .send(
        new AbortMultipartUploadCommand({
          Bucket: c.bucket,
          Key: params.objectKey,
          UploadId: uploadId,
        }),
      )
      .catch(() => {});
    throw e;
  }
}

async function* splitReadableToParts(
  stream: Readable,
  partSizeBytes: number,
): AsyncGenerator<Uint8Array<ArrayBufferLike>> {
  let pending: Uint8Array<ArrayBufferLike> = new Uint8Array(0);
  for await (const rawChunk of stream) {
    const chunk: Uint8Array<ArrayBufferLike> =
      typeof rawChunk === "string"
        ? Buffer.from(rawChunk)
        : Buffer.isBuffer(rawChunk)
          ? rawChunk
          : rawChunk instanceof Uint8Array
            ? rawChunk
            : Buffer.from(rawChunk as ArrayBufferLike);
    pending = concatBytes(pending, chunk);
    while (pending.length >= partSizeBytes) {
      yield pending.subarray(0, partSizeBytes);
      pending = pending.subarray(partSizeBytes);
    }
  }
  if (pending.length > 0) {
    yield pending;
  }
}

function concatBytes(
  a: Uint8Array<ArrayBufferLike>,
  b: Uint8Array<ArrayBufferLike>,
): Uint8Array<ArrayBufferLike> {
  if (a.length === 0) return b;
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

export async function deleteObject(objectKey: string): Promise<void> {
  const c = getMinioConfig();
  const client = getS3Client();
  await client.send(
    new DeleteObjectCommand({ Bucket: c.bucket, Key: objectKey }),
  );
}

export async function objectExists(objectKey: string): Promise<boolean> {
  const c = getMinioConfig();
  const client = getS3Client();
  try {
    await client.send(
      new HeadObjectCommand({ Bucket: c.bucket, Key: objectKey }),
    );
    return true;
  } catch (e) {
    const err = e as {
      name?: string;
      code?: string;
      Code?: string;
      $metadata?: { httpStatusCode?: number };
    };
    const status = Number(err.$metadata?.httpStatusCode ?? 0);
    const code = String(err.code ?? err.Code ?? err.name ?? "");
    if (status === 404 || code === "NotFound" || code === "NoSuchKey") {
      return false;
    }
    throw e;
  }
}

/**
 * Upload a Buffer directly (single PUT, max 5 GB but intended for small files).
 * Use putObjectStream for large streaming uploads.
 */
export async function putObjectBuffer(params: {
  objectKey: string;
  body: Buffer;
  contentType?: string;
}): Promise<void> {
  const c = getMinioConfig();
  const client = getS3Client();
  await client.send(
    new PutObjectCommand({
      Bucket: c.bucket,
      Key: params.objectKey,
      Body: params.body,
      ContentLength: params.body.byteLength,
      ContentType: params.contentType,
    }),
  );
}

/**
 * Generate a presigned GET URL for an object in the default MinIO bucket.
 * @param expiresIn TTL in seconds (default 900 = 15 min).
 */
export async function getMinioPresignedUrl(
  objectKey: string,
  expiresIn = 900,
): Promise<string> {
  const c = getMinioConfig();
  return getSignedUrl(
    getS3PresignClient(),
    new GetObjectCommand({ Bucket: c.bucket, Key: objectKey }),
    { expiresIn },
  );
}

export async function getObjectStream(params: {
  objectKey: string;
  /** HTTP Range header value, e.g. "bytes=0-1023". Enables 206 Partial Content. */
  range?: string;
}): Promise<{
  body: BodyInit;
  contentType: string | undefined;
  contentLength: number | undefined;
  contentRange: string | undefined;
  isPartial: boolean;
}> {
  const c = getMinioConfig();
  const client = getS3Client();
  const res = await client.send(
    new GetObjectCommand({
      Bucket: c.bucket,
      Key: params.objectKey,
      ...(params.range ? { Range: params.range } : {}),
    }),
  );
  if (!res.Body) {
    throw new Error("S3 GetObject returned empty body");
  }
  const isPartial = res.$metadata?.httpStatusCode === 206;
  return {
    body: res.Body.transformToWebStream() as unknown as BodyInit,
    contentType: res.ContentType,
    contentLength:
      typeof res.ContentLength === "number" ? res.ContentLength : undefined,
    contentRange: res.ContentRange,
    isPartial,
  };
}
