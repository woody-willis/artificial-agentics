/**
 * @module teams/development/code-reviewer
 * @file This agent is able to review code changes, suggest improvements, and ensure coding standards are met.
 */

import "dotenv/config";

import {
    Annotation,
    CompiledStateGraph,
    END,
    START,
    StateGraph,
} from "@langchain/langgraph";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import {
    AIMessage,
    HumanMessage,
    SystemMessage,
} from "@langchain/core/messages";

import { createModelInstance } from "../../model";
import {
    CreateDirectory,
    DeleteFile,
    deleteTempAgentDirectory,
    ListDirectory,
    PathExists,
    ReadFile,
    RemoveDirectory,
    SearchFiles,
    WriteFile,
} from "../../tools/file-system";
import { cloneRepository, CommitChanges } from "../../tools/git";
import { Tool } from "@langchain/core/tools";
import { SearxSearch } from "../../tools/search";
import simpleGit, { SimpleGit } from "simple-git";
import z from "zod";
import pino from "pino";

const logger = pino({
    level: "info",
    transport:
        process.env.ENVIRONMENT === "production"
            ? undefined
            : { target: "pino-pretty", options: { colorize: true } },
});

export interface DevelopmentCodeReviewerInvocationOptions {
    diff: string;
}

export interface DevelopmentCodeReviewerReviewCodeOptions {
    diff: string;
}

export interface DevelopmentCodeReviewerReviewCodeResponse {
    success: boolean;
    approved: boolean;
    suggestions?: string[];
}

const AgentState = Annotation.Root({
    messages: Annotation({ reducer: (x, y) => x.concat(y), default: () => [] }),
    success: Annotation({ reducer: (x, y) => y ?? x, default: () => false }),
    approved: Annotation({ reducer: (x, y) => y ?? x, default: () => false }),
    suggestions: Annotation({ reducer: (x, y) => y ?? x, default: () => [] }),
    toolCalls: Annotation({ reducer: (x, y) => y ?? x, default: () => [] }),
    iterations: Annotation({ reducer: (x, y) => y ?? x, default: () => 0 }),
    maxIterations: Annotation({ reducer: (x, y) => y ?? x, default: () => 25 }),
});

export class DevelopmentCodeReviewer {
    private threadId: string;
    private tempPath: string;
    private tools: Tool[];
    private responseSchema = z.object({
        success: z
            .boolean()
            .describe("Whether the code review was successful."),
        approved: z
            .boolean()
            .describe(
                "Whether the code changes meet all standards and are approved."
            ),
        suggestions: z
            .array(z.string())
            .optional()
            .describe("Suggestions for improvements."),
    });
    private modelInstance: BaseChatModel;
    private agentInstance: CompiledStateGraph<unknown, unknown>;
    private gitInstance: SimpleGit;
    private agentLogger: pino.Logger;

    /**
     * Initializes the DevelopmentCodeReviewer as a ReAct agent.
     */
    constructor(tempPath: string, verbose: boolean = false) {
        this.threadId = "development-code-reviewer-" + Date.now().toString();
        this.tempPath = tempPath;
        this.gitInstance = simpleGit();
        this.tools = [
            ReadFile(this.tempPath, this.threadId),
            ListDirectory(this.tempPath, this.threadId),
            // SearchFiles(this.tempPath, this.threadId),
        ];
        this.agentLogger = logger.child({
            module: "teams/development/code-reviewer",
            threadId: this.threadId,
        });

        this.modelInstance = createModelInstance({
            temperature: 0.7,
            maxRetries: 4,
            verbose: verbose,
        });
        this.modelInstance.bindTools(this.tools);

        this.agentInstance = this.createStateGraph();
    }

    private createStateGraph(): CompiledStateGraph<unknown, unknown> {
        const workflow = new StateGraph(AgentState);

        workflow.addNode("agent", this.agentNode.bind(this));
        workflow.addNode("tools", this.toolsNode.bind(this));
        workflow.addNode("format_response", this.formatResponseNode.bind(this));

        workflow.addEdge(START, "agent");

        workflow.addConditionalEdges("agent", this.shouldContinue.bind(this), {
            continue: "tools",
            format: "format_response",
            end: END,
        });

        workflow.addEdge("tools", "agent");
        workflow.addEdge("format_response", END);

        return workflow.compile();
    }

    private async agentNode(state) {
        const { messages, iterations, maxIterations } = state;

        if (iterations >= maxIterations) {
            return {
                ...state,
                messages: [
                    ...messages,
                    new AIMessage({
                        content: JSON.stringify({
                            plan: "Maximum iterations reached. Unable to complete task.",
                        }),
                    }),
                ],
            };
        }

        const systemMessage = new SystemMessage({
            content:
                "You are a professional code reviewer. You must use the tools provided to you to read files in the project repository and analyse the supplied git diff. You do not need to do any testing, as that will be done by another agent. Only approve code if it matches the style and standards of existing code in the repository. Output a JSON response with the following structure: { success: true/false, approved: true/false, suggestions: ['Suggestion 1', 'Suggestion 2'] }",
        });

        const TOKEN_LIMIT = 30000;
        const CHUNK_SIZE = 25000;

        const estimateTokens = (msgs) => {
            const totalCharacters = msgs.reduce(
                (acc, msg) => acc + (msg.content?.length || 0),
                0
            );
            return Math.ceil(totalCharacters / 4);
        };

        const chunkMessage = (message, maxTokens = 5000) => {
            const maxChars = maxTokens * 4;
            if (!message.content || message.content.length <= maxChars) {
                return [message];
            }

            const chunks = [];
            const content = message.content;
            let startIndex = 0;

            while (startIndex < content.length) {
                const endIndex = Math.min(
                    startIndex + maxChars,
                    content.length
                );
                let chunkEnd = endIndex;

                // Try to break at natural boundaries (newlines, periods, spaces)
                if (endIndex < content.length) {
                    const lastNewline = content.lastIndexOf("\n", endIndex);
                    const lastPeriod = content.lastIndexOf(".", endIndex);
                    const lastSpace = content.lastIndexOf(" ", endIndex);

                    // Use the best break point, but ensure we make progress
                    const breakPoint = Math.max(
                        lastNewline,
                        lastPeriod,
                        lastSpace
                    );
                    if (breakPoint > startIndex + maxChars * 0.5) {
                        chunkEnd = breakPoint + 1;
                    }
                }

                const chunkContent = content.substring(startIndex, chunkEnd);
                const chunkSuffix =
                    startIndex > 0
                        ? ` [Chunk ${Math.floor(startIndex / maxChars) + 1}]`
                        : chunkEnd < content.length
                          ? " [Chunk 1]"
                          : "";

                chunks.push(
                    new message.constructor({
                        content: chunkContent + chunkSuffix,
                        ...message,
                    })
                );

                startIndex = chunkEnd;
            }

            return chunks;
        };

        // Chunk large messages first
        let processedMessages = [];
        for (const message of messages) {
            const messageTokens = estimateTokens([message]);
            if (messageTokens > 5000) {
                const chunks = chunkMessage(message, 5000);
                processedMessages.push(...chunks);
            } else {
                processedMessages.push(message);
            }
        }

        let filteredMessages = [...processedMessages];
        let allMessages = [systemMessage, ...filteredMessages];
        let tokenEstimate = estimateTokens(allMessages);

        // Remove oldest messages (excluding system message) until under token limit
        while (tokenEstimate > CHUNK_SIZE && filteredMessages.length > 0) {
            filteredMessages = filteredMessages.slice(1); // Remove oldest message
            allMessages = [systemMessage, ...filteredMessages];
            tokenEstimate = estimateTokens(allMessages);
        }

        // If still over limit with just system message, chunk the system message
        if (tokenEstimate > CHUNK_SIZE) {
            const systemTokens = estimateTokens([systemMessage]);
            if (systemTokens > CHUNK_SIZE) {
                const chunkedSystemMessages = chunkMessage(
                    systemMessage,
                    CHUNK_SIZE - 1000
                );
                // Use only the first chunk of system message to stay within limits
                allMessages = [chunkedSystemMessages[0], ...filteredMessages];
                tokenEstimate = estimateTokens(allMessages);
            }
        }

        this.agentLogger.info(
            `Final token estimate: ${tokenEstimate}/${TOKEN_LIMIT} (${filteredMessages.length} messages retained)`
        );

        let response: AIMessage;

        for (let i = 0; i < 3; i++) {
            response = await this.modelInstance.invoke(allMessages, {
                recursionLimit: 100,
                configurable: { thread_id: this.threadId },
            });

            if (response.content.length > 0) {
                break;
            }
        }

        this.agentLogger.info(
            `Agent response: ${(response.content as string).substring(0, 100)} (Iteration: ${iterations + 1})`
        );

        return {
            ...state,
            messages: [...messages, response],
            iterations: iterations + 1,
        };
    }
    private async toolsNode(state) {
        const { messages } = state;
        const lastMessage = messages[messages.length - 1];

        const toolCalls = lastMessage.tool_calls || [];
        const toolMessages = [];

        for (const toolCall of toolCalls) {
            const tool = this.tools.find((t) => t.name === toolCall.name);
            if (tool) {
                try {
                    const result = await tool.invoke(toolCall.args);
                    this.agentLogger.info(
                        `Tool ${toolCall.name} executed successfully with result: ${JSON.stringify(result).substring(0, 100)}`
                    );
                    toolMessages.push({
                        role: "tool",
                        content: JSON.stringify(result),
                        tool_call_id: toolCall.id,
                    });
                } catch (error) {
                    this.agentLogger.error(
                        `Error executing tool ${toolCall.name}: ${error.message}`
                    );
                    toolMessages.push({
                        role: "tool",
                        content: `Error executing ${toolCall.name}: ${error.message}`,
                        tool_call_id: toolCall.id,
                    });
                }
            }
        }

        return { ...state, messages: [...messages, ...toolMessages] };
    }
    private async formatResponseNode(state) {
        const { messages } = state;

        const response = await createModelInstance({
            temperature: 0,
            maxRetries: 3,
            verbose: false,
        })
            .withStructuredOutput(this.responseSchema, {
                includeRaw: true,
                name: "format_response",
            })
            .invoke(
                [
                    ...messages,
                    new HumanMessage({
                        content: "Please format the response as valid JSON.",
                    }),
                ],
                {
                    recursionLimit: 25,
                    configurable: { thread_id: this.threadId },
                }
            );

        try {
            const parsedResponse = response.parsed;
            const validatedResponse = this.responseSchema.parse(parsedResponse);

            return {
                ...state,
                success: validatedResponse.success,
                approved: validatedResponse.approved,
                suggestions: validatedResponse.suggestions || [],
                messages: [
                    ...messages,
                    new AIMessage({
                        content: JSON.stringify(validatedResponse),
                    }),
                ],
            };
        } catch (error) {
            return {
                ...state,
                messages: [
                    ...messages,
                    new HumanMessage({
                        content: `The response format is invalid. Please provide a valid JSON response with the required structure: ${JSON.stringify(this.responseSchema.shape)}`,
                    }),
                ],
            };
        }
    }
    private shouldContinue(state) {
        const { messages, iterations, maxIterations } = state;
        const lastMessage = messages[messages.length - 1];

        if (iterations >= maxIterations) {
            return "end";
        }

        if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
            return "continue";
        }

        if (
            lastMessage.content &&
            (lastMessage.content.includes("{") ||
                lastMessage.content.includes("```"))
        ) {
            return "format";
        }

        return "end";
    }

    async init(): Promise<DevelopmentCodeReviewer> {
        // await cloneRepository(
        //     this.gitInstance,
        //     "https://github.com/woody-willis/artificial-agentics.git",
        //     this.tempPath,
        //     this.threadId
        // );

        return this;
    }

    async dispose(): Promise<void> {
        // if (this.tempPath) {
        //     deleteTempAgentDirectory(this.tempPath);
        // }
    }

    /**
     * Invokes the agent with the specified task and data.
     * @param {DevelopmentCodeReviewerInvocationOptions} options - The options for invoking the agent.
     * @returns {Promise<string>} The response from the agent.
     */
    async invoke(
        options: DevelopmentCodeReviewerInvocationOptions
    ): Promise<DevelopmentCodeReviewerReviewCodeResponse> {
        const { diff } = options;

        return this.reviewCode({ diff });
    }

    /**
     * Reviews code changes based on the git diff.
     * @param {DevelopmentCodeReviewerReviewCodeOptions} data - The data for reviewing code.
     * @returns {Promise<boolean>} True if the code changes were successfully reviewed, false otherwise.
     */
    private async reviewCode(
        data: DevelopmentCodeReviewerReviewCodeOptions
    ): Promise<DevelopmentCodeReviewerReviewCodeResponse> {
        // Get code reviewer agent to review the code changes
        const initialState = {
            ...AgentState,
            messages: [
                new HumanMessage({
                    content: `Here is the git diff of the code changes to review:\n\n${data.diff}`,
                }),
            ],
            iterations: 0,
        };

        const implementationResponse =
            await this.agentInstance.invoke(initialState);

        return {
            success: implementationResponse.success,
            approved: implementationResponse.approved,
            suggestions: implementationResponse.suggestions || [],
        };
    }
}
