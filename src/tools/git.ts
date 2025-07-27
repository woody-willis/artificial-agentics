/**
 * @module tools/git
 * @file This module provides agent tools for Git operations including cloning repositories and checking out branches.
 */

import { DynamicStructuredTool } from "@langchain/core/tools";
import { SimpleGit } from "simple-git";

import z from "zod";
import pino from "pino";

const logger = pino({
    level: "info",
    transport:
        process.env.ENVIRONMENT === "production"
            ? undefined
            : { target: "pino-pretty", options: { colorize: true } },
});

/**
 * Clones a Git repository to a specified local path.
 * * @param git The SimpleGit instance to use for the operation.
 * @param repositoryUrl The URL of the repository to clone.
 * @param localPath The local path where the repository should be cloned.
 * @returns A promise that resolves to the SimpleGit instance for the cloned repository.
 */
export async function cloneRepository(
    git: SimpleGit,
    repositoryUrl: string,
    localPath: string,
    agentId?: string
): Promise<SimpleGit> {
    await git.clone(repositoryUrl, localPath);
    await git.cwd(localPath);

    if (agentId) {
        await git.branch(["-m", agentId]);
        await git.checkout(agentId);
    }

    return git;
}

/**
 * Checks out a specific branch in a Git repository.
 * @param git The SimpleGit instance for the repository.
 * @param branchName The name of the branch to check out.
 * @returns A promise that resolves when the checkout is complete.
 */
export async function commitChanges(
    git: SimpleGit,
    message: string,
    branch: string = "main"
): Promise<void> {
    await git.add("./*");
    await git.commit(message);
    await git.push("origin", branch);
}

export const CommitChanges = (git: SimpleGit, agentId: string) => {
    const schema = z.object({
        message: z.string().describe("The commit message for the changes."),
    });

    const functionLogger = logger.child({
        module: "tools/git",
        function: "CommitChanges",
        agentId: agentId,
    });

    return new DynamicStructuredTool({
        name: "commit_changes",
        description: "Commits changes in the repository with a message.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            functionLogger.info("Committing changes", {
                message: input.message,
                agentId: agentId,
            });

            try {
                const { message } = input;
                await commitChanges(git, message, agentId);
                return `Successfully committed changes with message: ${message}`;
            } catch (error) {
                return `Error committing changes: ${(error as Error).message}`;
            }
        },
    });
};
