# CI/CD deployment setup

The `CI/CD` workflow validates both applications on pull requests. Pushes to
`main` or `master` publish immutable SHA tags and `latest` tags to GHCR. The
production deployment job runs only when the repository variable
`DEPLOY_ENABLED` is set to `true`.

## Production environment configuration

Create a GitHub Environment named `production`, add protection rules if
desired, and configure these secrets:

- `DEPLOY_HOST`: production server hostname or IP.
- `DEPLOY_USER`: SSH user with Docker access.
- `DEPLOY_SSH_KEY`: private SSH key for that user.
- `DEPLOY_KNOWN_HOSTS`: pinned `known_hosts` entry for the server.
- `GHCR_USERNAME`: GitHub account used by the server to pull images.
- `GHCR_PULL_TOKEN`: fine-grained token with read access to all project container images.

Configure these repository or environment variables:

- `DEPLOY_ENABLED`: set to `true` only after the server is prepared.
- `DEPLOY_PORT`: optional SSH port, default `22`.
- `DEPLOY_PATH`: optional deployment directory, default `/opt/edu-platform`.

The server must already have Docker with Compose v2 and a production `.env` at
`DEPLOY_PATH/.env`. Never commit that file. The workflow copies `compose.yml`,
logs in to GHCR, and pulls the published images. One-shot Compose services create
the RAG database, apply Prisma migrations, seed the initial administrator, and
create the MinIO bucket before the long-running services become healthy.
Production runs immutable `sha-<commit>` image tags rather than relying on a
mutable `latest` deployment.

Next.js and the MinIO API are bound to `127.0.0.1:3000` and `127.0.0.1:9000`.
Put TLS reverse proxies in front of both endpoints (and set
`MINIO_PUBLIC_ENDPOINT`), or enable the optional `tunnel` profile. PostgreSQL,
Redis, the MinIO console, and the RAG API remain private to the Compose network.
