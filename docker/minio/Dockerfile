FROM golang:1.24-bookworm AS builder

ARG MINIO_VERSION=RELEASE.2025-10-15T17-29-55Z
WORKDIR /src
RUN git clone --depth 1 --branch "${MINIO_VERSION}" https://github.com/minio/minio.git .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    version="${MINIO_VERSION#RELEASE.}" \
 && ldflags="$(MINIO_RELEASE=RELEASE go run buildscripts/gen-ldflags.go "$version")" \
 && CGO_ENABLED=0 go build -tags kqueue -trimpath --ldflags "$ldflags" -o /go/bin/minio

FROM debian:bookworm-slim

RUN useradd --system --uid 10002 --create-home minio \
 && mkdir -p /data \
 && chown minio:minio /data
COPY --from=builder /go/bin/minio /usr/local/bin/minio

USER minio
EXPOSE 9000 9001
ENTRYPOINT ["minio"]
CMD ["server", "/data", "--console-address", ":9001"]
