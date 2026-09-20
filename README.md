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
| 基础设施 | PostgreSQL、Redis、MinIO、Docker Compose | `edu-platform/docker-compose.yml` |
| 测试 | Pytest、Vitest | `tests/`、`edu-platform/__tests__/` |

## 环境要求

- Python 3.11 或 3.12
- [uv](https://docs.astral.sh/uv/)
- Node.js 22 和 npm
- Docker 与 Docker Compose
- 一个兼容 OpenAI API 的 LLM/Embedding 服务，或本地 Ollama Embedding 服务

## 快速开始

1. 克隆仓库并创建本地配置：

   ```bash
   git clone https://github.com/xixi-in-polyu/eduAgent.git
   cd eduAgent
   cp .env.example .env
   ln -s ../.env edu-platform/.env
   ```

   如果 `edu-platform/.env` 已存在，可跳过最后一条命令。

2. 修改 `.env`。至少应替换以下占位值：

   - `JWT_SECRET`、`INTERNAL_API_KEY`、`RAG_SERVICE_API_KEY`
   - `SEED_ADMIN_PASSWORD`、`LLM_CONFIG_ENCRYPTION_KEY`
   - `LLM_API_KEY` 及所选模型配置
   - 使用 MinerU Cloud 时的 `MINERU_CLOUD_API_KEY`

3. 安装依赖：

   ```bash
   uv sync --locked --dev
   cd edu-platform
   npm ci
   cd ..
   ```

4. 启动 PostgreSQL、Redis 和 MinIO：

   ```bash
   cd edu-platform
   docker compose up -d postgres redis minio
   ```

5. 初始化数据库并启动服务（分别在三个终端执行）：

   ```bash
   # 终端 1：数据库迁移、管理员账号和 Web 应用
   cd edu-platform
   npm run db:migrate
   npm run db:seed
   npm run dev
   ```

   ```bash
   # 终端 2：RAG API
   uv run rag-service
   ```

   ```bash
   # 终端 3：资料处理 Worker
   uv run edu-rag-worker
   ```

打开 <http://localhost:3000>，使用 `.env` 中的 `SEED_ADMIN_USERNAME` 和
`SEED_ADMIN_PASSWORD` 登录。RAG API 默认监听 <http://localhost:8001>。

也可以在配置好 `.env` 后运行完整容器栈：

```bash
cd edu-platform
docker compose up --build
```

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
```

## 配置与安全

`.env`、数据库文件、运行输出、评估结果和本地开发笔记均由 `.gitignore` 排除。
不要把真实 API Key、令牌、密码或生产连接字符串提交到仓库；新增配置项时同步更新
`.env.example`，只保留无敏感信息的示例值。
