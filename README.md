# eduAgent

eduAgent 是一个面向教师、学生和管理员的智能教育平台。项目将 Next.js 教学应用与
Python RAG 服务组合在一起，支持课程资料处理、基于资料的问答、作业生成与批改、
个人知识库、学习记忆和复习提醒。

## 主要功能

- 课程、课时、成员与分享码管理
- PDF、文本、Office、图片和音视频资料上传、解析与向量化
- 带引用和工具调用的课程问答与个人知识库问答
- AI 辅助出题、教师编辑发布、学生提交和教师复核
- 学习记忆、每日复习、通知与课程分析
- 管理员用户、课程、Agent 技能和风格管理

## 技术架构

| 模块 | 主要技术 | 目录 |
| --- | --- | --- |
| Web 应用与 API | Next.js 15、React 19、TypeScript、Prisma | `edu-platform/` |
| RAG API 与任务 Worker | FastAPI、Python 3.11/3.12、pgvector | `src/rag_service/`、`src/rag_mvp/` |
| 基础设施 | PostgreSQL、Redis、MinIO、Docker Compose | `compose.yml` |
| 测试 | Pytest、Vitest | `tests/`、`edu-platform/__tests__/` |

## 环境要求

容器部署只需要 Docker Engine 和 Docker Compose v2。源码开发还需要 Python 3.11/3.12、
[uv](https://docs.astral.sh/uv/)、Node.js 22 和 npm。模型可使用兼容 OpenAI API 的服务，
Embedding 也可连接宿主机或远程 Ollama。

## 快速开始

1. 克隆仓库并创建本地配置：

   ```bash
   git clone https://github.com/xixi-in-polyu/eduAgent.git
   cd eduAgent
   cp .env.example .env
   ```

2. 修改 `.env`。至少应替换以下占位值：

   - `JWT_SECRET`、`INTERNAL_API_KEY`、`RAG_SERVICE_API_KEY`
   - `SEED_ADMIN_PASSWORD`、`LLM_CONFIG_ENCRYPTION_KEY`
   - `POSTGRES_PASSWORD`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY`
   - `LLM_API_KEY` 及所选模型配置
   - 使用 MinerU Cloud 时的 `MINERU_CLOUD_API_KEY`

3. 构建并启动完整容器栈：

   ```bash
   docker compose up -d --build
   ```

   Compose 会自动完成以下工作：

- 创建业务数据库和独立的 `edu_rag` 向量数据库
- 安装 pgvector 扩展并执行 Prisma migrations
- 创建初始管理员和 MinIO bucket
- 启动 Next.js、RAG API、资料 Worker 与复习调度器
- 持久化 PostgreSQL、Redis 和 MinIO 数据

4. 查看状态和日志：

   ```bash
   docker compose ps
   docker compose logs -f nextjs rag-service rag-worker
   ```

打开 <http://localhost:3000>，使用 `.env` 中的 `SEED_ADMIN_USERNAME` 和
`SEED_ADMIN_PASSWORD` 登录。默认 Next.js 的 `3000` 和 MinIO API 的 `9000` 端口只绑定
到宿主机回环地址；数据库、Redis、MinIO Console 和 RAG API 只在 Compose 网络内可见。

本地调试基础设施时，可额外暴露内部端口：

```bash
docker compose -f compose.yml -f compose.dev.yml up -d --build
```

## Linux 生产部署

在 Linux 主机安装 Docker Engine 与 Compose v2 后，将仓库和生产 `.env` 放到例如
`/opt/edu-agent`。生产环境建议用 Nginx/Caddy 将应用域名代理到 `127.0.0.1:3000`，
并将对象存储域名代理到 `127.0.0.1:9000`。将后者写入
`MINIO_PUBLIC_ENDPOINT=https://files.example.com`，保证浏览器和外部视觉模型能访问签名 URL：

```bash
cd /opt/edu-agent
docker compose up -d --build --wait
docker compose ps
```

以上源码构建方式不依赖容器仓库，适合 fork 首次部署。若已在 GitHub 仓库中启用 Actions，
并由 CI/CD workflow 成功发布 GHCR 镜像，则可改用不可变镜像快速部署：

```bash
GITHUB_OWNER=xixi-in-polyu IMAGE_TAG=sha-<commit> docker compose pull
GITHUB_OWNER=xixi-in-polyu IMAGE_TAG=sha-<commit> docker compose up -d --no-build --wait
```

若 Ollama 或其他模型服务运行在 Linux 宿主机上，容器内地址应使用
`http://host.docker.internal:<端口>`，不要使用 `localhost`。

使用 Cloudflare Tunnel 时运行：

```bash
docker compose --profile tunnel up -d
```

升级前备份三个 named volumes；升级使用 `docker compose pull && docker compose up -d`。
不要使用 `docker compose down -v`，该命令会删除数据库和对象存储数据。更完整的 CI/CD
配置见 [`.github/DEPLOYMENT.md`](.github/DEPLOYMENT.md)。

## 开发与验证

```bash
# Python
uv run python -m compileall -q src
uv run pytest -q tests/unit

# Web
cd edu-platform
npm run lint
npx tsc --noEmit
npm test
npm run build
```

部分集成、评估和性能测试依赖已启动的服务、测试数据与 `.env` 配置。评估说明位于
`tests/eval/README.md`，实现范围和验收视角的详细说明位于
[`docs/project-documentation.md`](docs/project-documentation.md)。

## 项目结构

```text
edu-platform/       Next.js 页面、API、服务、Prisma 与前端测试
src/rag_mvp/        文档处理、检索、生成与异步 Worker
src/rag_service/    FastAPI RAG 服务
scripts/            数据导入、迁移与维护工具
skills/             教育 Agent 技能说明
tests/unit/         Python 单元测试
tests/eval/         RAG 和生成效果评估
tests/perf/         性能测试
docs/               正式项目文档
compose.yml         完整生产容器编排
compose.dev.yml     本地端口覆盖配置
```

## 配置与安全

`.env`、数据库文件、运行输出、评估结果和本地开发笔记均由 `.gitignore` 排除。
不要把真实 API Key、令牌、密码或生产连接字符串提交到仓库；新增配置项时同步更新
`.env.example`，只保留无敏感信息的示例值。
