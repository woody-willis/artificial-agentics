/**
 * @module tools/file-system
 * @file This module provides agent tools for file system operations including searching using an embeddings model.
 */

import { DynamicStructuredTool } from "@langchain/core/tools";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";

import * as fs from "fs";
import * as fsp from "fs/promises";
import * as path from "path";
import { OllamaEmbeddings } from "@langchain/ollama";
import z from "zod";

/**
 * Creates a temporary directory for agent files.
 * The directory is created under the 'temp' folder in the current directory.
 * The name of the directory is generated with a random suffix to avoid collisions.
 *
 * @returns {string} The path to the created temporary agent directory.
 */
export function createTempAgentDirectory(): string {
    const tempDir = path.join(process.cwd(), "temp");
    if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir, { recursive: true });
    }

    const randomSuffix = Math.random().toString(36).substring(2, 15);
    const tempAgentDir = path.join(tempDir, "agent_" + randomSuffix);
    fs.mkdirSync(tempAgentDir, { recursive: true });

    return tempAgentDir;
}

/**
 * Deletes the temporary agent directory.
 * @param {string} tempPath - The path to the temporary agent directory.
 * @returns {void} A promise that resolves when the directory is deleted.
 */
export function deleteTempAgentDirectory(tempPath: string): void {
    if (fs.existsSync(tempPath)) {
        fs.rmSync(tempPath, { recursive: true });
    }
}

export const ReadFile = (tempPath: string) => {
    const schema = z.object({
        filePath: z
            .string()
            .describe(
                "The path to the file to read, relative to the temporary agent directory."
            ),
    });

    return new DynamicStructuredTool({
        name: "read_file",
        description: "Reads the contents of a file.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            try {
                const { filePath } = input;
                const absolutePath = path.join(tempPath, filePath);
                return fsp.readFile(absolutePath, "utf-8");
            } catch (error) {
                return `Error reading file: ${(error as Error).message}`;
            }
        },
    });
};

export const WriteFile = (tempPath: string) => {
    const schema = z.object({
        filePath: z
            .string()
            .describe(
                "The path to the file to write, relative to the temporary agent directory."
            ),
        content: z.string().describe("The content to write to the file."),
    });

    return new DynamicStructuredTool({
        name: "write_file",
        description: "Writes content to a file.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            try {
                const { filePath, content } = input;
                const absolutePath = path.join(tempPath, filePath);

                let cleanedContent = content.trim();

                // Strip speech marks from beginning and end of content
                if (cleanedContent.startsWith('"')) {
                    cleanedContent = cleanedContent.slice(1);
                }
                if (cleanedContent.endsWith('"')) {
                    cleanedContent = cleanedContent.slice(0, -1);
                }

                // Replace newlines ascii with actual newlines
                cleanedContent = cleanedContent.replace(/\\n/g, "\n");

                // Remove weird backslashed characters
                cleanedContent = cleanedContent
                    .replace(/\\'/g, "'")
                    .replace(/\\"/g, '"')
                    .replace(/\\`/g, "`")
                    .replace(/\\\$/g, "$")
                    .replace(/\\\\/g, "\\");

                await fsp.writeFile(absolutePath, cleanedContent, "utf-8");

                return `Successfully wrote to ${filePath}`;
            } catch (error) {
                return `Error writing file: ${(error as Error).message}`;
            }
        },
    });
};

export const DeleteFile = (tempPath: string) => {
    const schema = z.object({
        filePath: z
            .string()
            .describe(
                "The path to the file to delete, relative to the temporary agent directory."
            ),
    });

    return new DynamicStructuredTool({
        name: "delete_file",
        description: "Deletes a file.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            try {
                const { filePath } = input;
                const absolutePath = path.join(tempPath, filePath);
                await fsp.unlink(absolutePath);
                return `Successfully deleted ${filePath}`;
            } catch (error) {
                return `Error deleting file: ${(error as Error).message}`;
            }
        },
    });
};

export const ListDirectory = (tempPath: string) => {
    const schema = z.object({
        dirPath: z
            .string()
            .describe(
                "The path to the directory to list, relative to the temporary agent directory."
            ),
    });

    return new DynamicStructuredTool({
        name: "list_directory",
        description: "Lists the contents of a directory.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            try {
                const { dirPath } = input;
                const absolutePath = path.join(tempPath, dirPath);
                const items = await fsp.readdir(absolutePath, {
                    withFileTypes: true,
                });
                const result = items.map((item) => {
                    return item.isDirectory() ? `${item.name}/` : item.name;
                });
                return JSON.stringify(result);
            } catch (error) {
                return `Error listing directory: ${(error as Error).message}`;
            }
        },
    });
};

export const CreateDirectory = (tempPath: string) => {
    const schema = z.object({
        dirPath: z
            .string()
            .describe(
                "The path to the directory to create, relative to the temporary agent directory."
            ),
    });

    return new DynamicStructuredTool({
        name: "create_directory",
        description: "Creates a new directory.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            try {
                const { dirPath } = input;
                const absolutePath = path.join(tempPath, dirPath);
                await fsp.mkdir(absolutePath, { recursive: true });
                return `Successfully created directory ${dirPath}`;
            } catch (error) {
                return `Error creating directory: ${(error as Error).message}`;
            }
        },
    });
};

export const RemoveDirectory = (tempPath: string) => {
    const schema = z.object({
        dirPath: z
            .string()
            .describe(
                "The path to the directory to remove, relative to the temporary agent directory."
            ),
    });

    return new DynamicStructuredTool({
        name: "remove_directory",
        description: "Removes a directory.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            try {
                const { dirPath } = input;
                const absolutePath = path.join(tempPath, dirPath);
                await fsp.rmdir(absolutePath, { recursive: true });
                return `Successfully removed directory ${dirPath}`;
            } catch (error) {
                return `Error removing directory: ${(error as Error).message}`;
            }
        },
    });
};

export const PathExists = (tempPath: string) => {
    const schema = z.object({
        pathToCheck: z
            .string()
            .describe(
                "The path to check, relative to the temporary agent directory."
            ),
    });

    return new DynamicStructuredTool({
        name: "path_exists",
        description: "Checks if a path exists.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            try {
                const { pathToCheck } = input;
                const absolutePath = path.join(tempPath, pathToCheck);
                try {
                    await fsp.access(absolutePath);
                    return JSON.stringify({ exists: true });
                } catch {
                    return JSON.stringify({ exists: false });
                }
            } catch (error) {
                return `Error checking path: ${(error as Error).message}`;
            }
        },
    });
};

const shouldProcessFile = (filePath: string): boolean => {
    const textExtensions = [
        ".txt",
        ".md",
        ".js",
        ".ts",
        ".py",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".css",
        ".html",
        ".xml",
        ".json",
        ".yaml",
        ".yml",
        ".sql",
        ".sh",
        ".bat",
    ];
    const ext = path.extname(filePath).toLowerCase();
    return textExtensions.includes(ext);
};

const getProcessableFiles = async (dirPath: string): Promise<string[]> => {
    const files: string[] = [];

    const processDir = async (currentPath: string) => {
        try {
            const items = await fsp.readdir(currentPath);

            for (const item of items) {
                const itemPath = path.join(currentPath, item);
                const stats = await fsp.stat(itemPath);

                if (stats.isDirectory()) {
                    // Skip common directories that shouldn't be indexed
                    if (
                        ![
                            "node_modules",
                            ".git",
                            ".vscode",
                            "dist",
                            "build",
                        ].includes(item)
                    ) {
                        await processDir(itemPath);
                    }
                } else if (stats.isFile() && shouldProcessFile(itemPath)) {
                    files.push(itemPath);
                }
            }
        } catch (error) {
            // Skip directories we can't read
            console.warn(
                `Skipping directory ${currentPath}: ${(error as Error).message}`
            );
        }
    };

    await processDir(dirPath);
    return files;
};

export const SearchFiles = (tempPath: string) => {
    const schema = z.object({
        dirPath: z
            .string()
            .describe(
                "The path to the directory to search, relative to the temporary agent directory."
            ),
        query: z.string().describe("The search query."),
        topK: z
            .number()
            .min(1)
            .optional()
            .describe("The number of top results to return."),
    });

    return new DynamicStructuredTool({
        name: "search_files",
        description: "Searches for files in a directory based on a query.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            try {
                const { dirPath, query, topK = 5 } = input;

                // Get all processable files
                const files = await getProcessableFiles(dirPath);

                if (files.length === 0) {
                    return JSON.stringify({
                        success: true,
                        results: [],
                        message: "No processable files found in the directory",
                    });
                }

                // Create documents from files
                const documents: Document[] = [];
                const textSplitter = new RecursiveCharacterTextSplitter({
                    chunkSize: 1000,
                    chunkOverlap: 200,
                });

                for (const filePath of files) {
                    try {
                        const content = await fsp.readFile(filePath, "utf-8");
                        const chunks = await textSplitter.splitText(content);

                        for (let i = 0; i < chunks.length; i++) {
                            documents.push(
                                new Document({
                                    pageContent: chunks[i],
                                    metadata: {
                                        filePath,
                                        chunkIndex: i,
                                        totalChunks: chunks.length,
                                    },
                                })
                            );
                        }
                    } catch (error) {
                        console.warn(
                            `Skipping file ${filePath}: ${(error as Error).message}`
                        );
                    }
                }

                if (documents.length === 0) {
                    return JSON.stringify({
                        success: true,
                        results: [],
                        message: "No content could be extracted from files",
                    });
                }

                // Create vector store and perform search
                const embeddings = new OllamaEmbeddings({
                    model: "nomic-embed-text",
                });
                const vectorStore = await MemoryVectorStore.fromDocuments(
                    documents,
                    embeddings
                );
                const searchResults =
                    await vectorStore.similaritySearchWithScore(query, topK);

                // Format results
                const results = await Promise.all(
                    searchResults.map(async ([doc, score]) => {
                        const result: any = {
                            filePath: doc.metadata.filePath,
                            score,
                            chunkIndex: doc.metadata.chunkIndex,
                            totalChunks: doc.metadata.totalChunks,
                        };

                        // Get file stats
                        try {
                            const stats = await fsp.stat(doc.metadata.filePath);
                            result.fileSize = stats.size;
                            result.lastModified = stats.mtime;
                        } catch (error) {
                            // File might have been deleted, skip stats
                        }

                        return result;
                    })
                );

                return JSON.stringify({
                    success: true,
                    results,
                    message: `Found ${results.length} relevant chunks across ${files.length} files`,
                });
            } catch (error) {
                return JSON.stringify({
                    success: false,
                    results: [],
                    message: `Error searching files: ${(error as Error).message}`,
                });
            }
        },
    });
};

export const fileSystemTools = (tempPath: string) => {
    return [
        ReadFile(tempPath),
        WriteFile(tempPath),
        DeleteFile(tempPath),
        ListDirectory(tempPath),
        CreateDirectory(tempPath),
        PathExists(tempPath),
        SearchFiles(tempPath),
    ];
};
