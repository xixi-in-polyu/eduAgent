import { PrismaClient, UserRole } from "@prisma/client";
import argon2 from "argon2";

const prisma = new PrismaClient();

async function main() {
  const username = process.env.SEED_ADMIN_USERNAME ?? "admin";
  const password = process.env.SEED_ADMIN_PASSWORD;

  if (!password || password.length < 12) {
    throw new Error(
      "SEED_ADMIN_PASSWORD is required and must be at least 12 characters",
    );
  }

  const existing = await prisma.user.findUnique({ where: { username } });
  if (existing) {
    console.info("Seed skipped: admin username already exists");
    return;
  }

  const passwordHash = await argon2.hash(password, { type: argon2.argon2id });
  await prisma.user.create({
    data: {
      username,
      passwordHash,
      role: UserRole.ADMIN,
      realName: "System Admin",
      isActive: true,
    },
  });

  console.info("Seed completed: admin user created");
}

main()
  .catch((error) => {
    console.error(error);
    process.exitCode = 1;
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
