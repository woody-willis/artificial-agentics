/**
 * @module teams/development/team-manager
 * @file This agent manages the roles within the development team to complete tasks efficiently and accurately.
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
    createTempAgentDirectory,
    deleteTempAgentDirectory,
    ListDirectory,
    PathExists,
    ReadFile,
    SearchFiles,
} from "../../tools/file-system";
import { cloneRepository } from "../../tools/git";
import { Tool } from "@langchain/core/tools";
import { SearxSearch } from "../../tools/search";
import { DevelopmentCodeWriter } from "./code-writer";
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

export enum DevelopmentTeamManagerInvocationTask {
    AddFeature = "AddFeature",
    FixBug = "FixBug",
}

export interface DevelopmentTeamManagerInvocationOptions {
    task: DevelopmentTeamManagerInvocationTask;
    data: Record<string, any>;
}

export interface DevelopmentTeamManagerAddFeatureOptions {
    description: string;
}

export interface DevelopmentTeamManagerFixBugOptions {
    location: string;
    description: string;
    severity: "low" | "medium" | "high" | "critical";
    stepsToReproduce?: string[];
    expectedBehavior?: string;
    actualBehavior?: string;
    additionalInfo?: string;
}

const AgentState = Annotation.Root({
    messages: Annotation({ reducer: (x, y) => x.concat(y), default: () => [] }),
    plan: Annotation({ reducer: (x, y) => y ?? x, default: () => null }),
    toolCalls: Annotation({ reducer: (x, y) => y ?? x, default: () => [] }),
    iterations: Annotation({ reducer: (x, y) => y ?? x, default: () => 0 }),
    maxIterations: Annotation({ reducer: (x, y) => y ?? x, default: () => 25 }),
});

export class DevelopmentTeamManager {
    private threadId: string;
    private tempPath: string;
    private tools: Tool[];
    private responseSchema = z.object({
        plan: z
            .string()
            .describe(
                "The detailed plan for the task, including references to files to modify, create, or delete."
            ),
    });
    private modelInstance: BaseChatModel;
    private agentInstance: CompiledStateGraph<unknown, unknown>;
    private gitInstance: SimpleGit;
    private agentLogger: pino.Logger;

    /**
     * Initializes the DevelopmentTeamManager as a ReAct agent.
     */
    constructor(verbose: boolean = false) {
        this.threadId = "development-team-manager-" + Date.now().toString();
        this.tempPath = createTempAgentDirectory();
        this.gitInstance = simpleGit();
        this.tools = [
            SearxSearch(),
            ReadFile(this.tempPath, this.threadId),
            ListDirectory(this.tempPath, this.threadId),
            PathExists(this.tempPath, this.threadId),
            // SearchFiles(this.tempPath, this.threadId)
        ];

        this.agentLogger = logger.child({
            module: "teams/development/team-manager",
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
                "You are the team manager of a development team. You are given a task for which you must create a detailed plan to achieve that task. You must use the tools provided to you to gather information. The plan must align with the style and language already used in the codebase. Use the read file and list directory tools to explore the codebase. Output a detailed plan in the format: { plan: 'Your detailed plan here' } only once you have gathered enough information and done enough enumeration.",
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
                recursionLimit: 25,
                configurable: { thread_id: this.threadId },
            });

            if (response.content.length > 0) {
                break;
            }
        }

        this.agentLogger.info(
            `Agent response length: ${response.content.length} (Iteration: ${iterations + 1})`
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
            if (!response.parsed) {
                const rawOutputParse = JSON.parse(
                    response.raw.content as string
                );
                response.parsed = rawOutputParse;
            }

            const parsedResponse = response.parsed;
            const validatedResponse = this.responseSchema.parse(parsedResponse);

            return {
                ...state,
                plan: validatedResponse.plan,
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

    async init(): Promise<DevelopmentTeamManager> {
        await cloneRepository(
            this.gitInstance,
            "https://github.com/woody-willis/artificial-agentics.git",
            this.tempPath
        );

        return this;
    }

    async dispose(): Promise<void> {
        if (this.tempPath) {
            deleteTempAgentDirectory(this.tempPath);
        }
    }

    /**
     * Invokes the agent with the specified task and data.
     * @param {DevelopmentTeamManagerInvocationOptions} options - The options for invoking the agent.
     * @returns {Promise<string>} The response from the agent.
     */
    async invoke(
        options: DevelopmentTeamManagerInvocationOptions
    ): Promise<boolean> {
        const { task, data } = options;

        switch (task) {
            case DevelopmentTeamManagerInvocationTask.AddFeature:
                return this.addFeature(
                    data as DevelopmentTeamManagerAddFeatureOptions
                );
            case DevelopmentTeamManagerInvocationTask.FixBug:
                return this.fixBug(data as DevelopmentTeamManagerFixBugOptions);
            default:
                throw new Error(`Unknown task: ${task}`);
        }
    }

    /**
     * Adds a feature based on the provided data.
     * @param {DevelopmentTeamManagerAddFeatureOptions} data - The data for the feature to be added.
     * @returns {Promise<boolean>} True if the feature was added successfully, false otherwise.
     */
    private async addFeature(
        data: DevelopmentTeamManagerAddFeatureOptions
    ): Promise<boolean> {
        // Get team manager to analyse codebase and generate a detailed plan for the feature
        const initialState = {
            ...AgentState,
            messages: [
                new HumanMessage({
                    content: `Add a feature with the following description: ${data.description}\n\nCreate a detailed plan to achieve this task and reference files to modify, create or delete etc. You must use tools and read files.`,
                }),
            ],
            iterations: 0,
        };

        const planResponse = await this.agentInstance.invoke(initialState);
        const plan = planResponse.plan;

        if (!plan) {
            throw new Error("Failed to generate a plan");
        }

        console.log(`Generated plan: ${plan}`);

        const codeWriterAgent = await new DevelopmentCodeWriter().init();

        const codeWriteResponse = await codeWriterAgent
            .invoke({ plan: plan })
            .catch((error) => {
                console.error("Error invoking code writer agent:", error);
                return false;
            });

        await codeWriterAgent.dispose();

        console.log(`Code write success: ${codeWriteResponse}`);

        return true;
    }

    /**
     * Fixes a bug based on the provided data.
     * @param {DevelopmentTeamManagerFixBugOptions} data - The data for the bug to be fixed.
     * @returns {Promise<boolean>} True if the bug was fixed successfully, false otherwise.
     */
    private async fixBug(
        data: DevelopmentTeamManagerFixBugOptions
    ): Promise<boolean> {
        // Implement the logic for fixing a bug
        return true;
    }
}
