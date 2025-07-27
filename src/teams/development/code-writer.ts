/**
 * @module teams/development/code-writer
 * @file This agent is able to write code based on a plan provided by the team manager agent.
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

import z from "zod";

import { createModelInstance } from "../../model";
import {
    CreateDirectory,
    createTempAgentDirectory,
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

export interface DevelopmentCodeWriterInvocationOptions {
    plan: string;
}

export interface DevelopmentCodeWriterCompletePlanOptions {
    plan: string;
}

const AgentState = Annotation.Root({
    messages: Annotation({ reducer: (x, y) => x.concat(y), default: () => [] }),
    success: Annotation({ reducer: (x, y) => y ?? x, default: () => false }),
    toolCalls: Annotation({ reducer: (x, y) => y ?? x, default: () => [] }),
    iterations: Annotation({ reducer: (x, y) => y ?? x, default: () => 0 }),
    maxIterations: Annotation({ reducer: (x, y) => y ?? x, default: () => 25 }),
});

export class DevelopmentCodeWriter {
    private threadId: string;
    private tempPath: string;
    private tools: Tool[];
    private responseSchema = z.object({
        success: z
            .boolean()
            .describe(
                "Indicates whether the code writing & commit were successful."
            ),
    });
    private modelInstance: BaseChatModel;
    private agentInstance: CompiledStateGraph<unknown, unknown>;
    private gitInstance: SimpleGit;

    /**
     * Initializes the DevelopmentCodeWriter as a ReAct agent.
     */
    constructor(verbose: boolean = false) {
        this.threadId = "development-code-writer-" + Date.now().toString();
        this.tempPath = createTempAgentDirectory();
        this.gitInstance = simpleGit();
        this.tools = [
            SearxSearch(),
            ReadFile(this.tempPath),
            WriteFile(this.tempPath),
            DeleteFile(this.tempPath),
            ListDirectory(this.tempPath),
            CreateDirectory(this.tempPath),
            RemoveDirectory(this.tempPath),
            // SearchFiles(this.tempPath),
            CommitChanges(this.gitInstance, this.threadId),
        ];

        this.modelInstance = createModelInstance({
            modelName: "gemini-2.5-flash",
            temperature: 0.7,
            maxRetries: 8,
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
                "You are a code writer agent. You are given a plan to implement and you must use the tools provided to you to read, write, and modify files in the project repository. You do not need to do any testing, as that will be done by another agent. You can also search for information online using the Searx search tool. Make sure to commit your changes to the repository after implementing the plan. Commit messages must follow the Conventional Commits specification.",
        });

        const allMessages = [systemMessage, ...messages];

        const response = await this.modelInstance.invoke(allMessages, {
            recursionLimit: 100,
            configurable: { thread_id: this.threadId },
        });

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
                    toolMessages.push({
                        role: "tool",
                        content: JSON.stringify(result),
                        tool_call_id: toolCall.id,
                    });
                } catch (error) {
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

    async init(): Promise<DevelopmentCodeWriter> {
        this.gitInstance = await cloneRepository(
            this.gitInstance,
            "https://github.com/woody-willis/artificial-agentics.git",
            this.tempPath,
            this.threadId
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
     * @param {DevelopmentCodeWriterInvocationOptions} options - The options for invoking the agent.
     * @returns {Promise<string>} The response from the agent.
     */
    async invoke(
        options: DevelopmentCodeWriterInvocationOptions
    ): Promise<boolean> {
        const { plan } = options;

        return this.completePlan({ plan });
    }

    /**
     * Completes the plan by implementing the code based on the provided plan.
     * @param {DevelopmentCodeWriterCompletePlanOptions} data - The data for the plan to be implemented.
     * @returns {Promise<boolean>} True if the plan was implemented successfully, false otherwise.
     */
    private async completePlan(
        data: DevelopmentCodeWriterCompletePlanOptions
    ): Promise<boolean> {
        // Get code writer agent instance to implement the plan
        const initialState = {
            ...AgentState,
            messages: [
                new HumanMessage({
                    content: `The following plan has been compiled for you to complete: ${data.plan}\n\nRead and edit files & directories to implement this plan.`,
                }),
            ],
            iterations: 0,
        };

        const implementationResponse =
            await this.agentInstance.invoke(initialState);
        const success = implementationResponse.success;

        if (!success) {
            throw new Error("Failed to implement the plan");
        }

        return true;
    }
}
