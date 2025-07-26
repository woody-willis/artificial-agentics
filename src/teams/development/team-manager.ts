/**
 * @module teams/development/team-manager
 * @file This agent manages the roles within the development team to complete tasks efficiently and accurately.
 */

import "dotenv/config";

import { Annotation, CompiledStateGraph, END, START, StateGraph } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";

import z from "zod";

import { createModelInstance } from "../../model";
import { createTempAgentDirectory, deleteTempAgentDirectory, ListDirectory, PathExists, ReadFile, SearchFiles } from "../../tools/file-system";
import { cloneRepository } from "../../tools/git";
import { DynamicTool } from "@langchain/core/tools";

export enum DevelopmentTeamManagerInvocationTask {
    AddFeature = "AddFeature",
    FixBug = "FixBug",
};

export interface DevelopmentTeamManagerInvocationOptions {
    task: DevelopmentTeamManagerInvocationTask;
    data: Record<string, any>;
};

export interface DevelopmentTeamManagerAddFeatureOptions {
    description: string;
};

export interface DevelopmentTeamManagerFixBugOptions {
    location: string;
    description: string;
    severity: "low" | "medium" | "high" | "critical";
    stepsToReproduce?: string[];
    expectedBehavior?: string;
    actualBehavior?: string;
    additionalInfo?: string;
};

const AgentState = Annotation.Root({
    messages: Annotation({
        reducer: (x, y) => x.concat(y),
        default: () => []
    }),
    plan: Annotation({
        reducer: (x, y) => y ?? x,
        default: () => null
    }),
    toolCalls: Annotation({
        reducer: (x, y) => y ?? x,
        default: () => []
    }),
    iterations: Annotation({
        reducer: (x, y) => y ?? x,
        default: () => 0
    }),
    maxIterations: Annotation({
        reducer: (x, y) => y ?? x,
        default: () => 25
    })
});

export class DevelopmentTeamManager {
    private threadId: string;
    private tempPath: string;
    private tools: DynamicTool[];
    private responseSchema = z.object({
        plan: z.string().describe("The detailed plan for the task, including references to files to modify, create, or delete.")
    });
    private modelInstance: BaseChatModel;
    private agentInstance: CompiledStateGraph<unknown, unknown>;

    /**
     * Initializes the DevelopmentTeamManager as a ReAct agent.
     */
    constructor(verbose: boolean = false) {
        this.threadId = "development-team-manager-" + Date.now().toString();
        this.tempPath = createTempAgentDirectory();
        this.tools = [
            ReadFile(this.tempPath),
            ListDirectory(this.tempPath),
            PathExists(this.tempPath),
            // SearchFiles(this.tempPath)
        ];

        this.modelInstance = createModelInstance({
            temperature: 0.3,
            maxRetries: 3,
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

        workflow.addConditionalEdges(
            "agent",
            this.shouldContinue.bind(this),
            {
                "continue": "tools",
                "format": "format_response",
                "end": END
            }
        );

        workflow.addEdge("tools", "agent");
        workflow.addEdge("format_response", END);

        return workflow.compile();
    }

    private async agentNode(state) {
        const { messages, iterations, maxIterations } = state;
        
        if (iterations >= maxIterations) {
            return {
                ...state,
                messages: [...messages, new AIMessage({
                    content: JSON.stringify({ 
                        plan: "Maximum iterations reached. Unable to complete task." 
                    })
                })]
            };
        }

        const systemMessage = new SystemMessage({
            content: 'You are the team manager of a development team. You are given a task to be completed and you must create a detailed plan to achieve the task. You must use the tools provided to you to gather information and create a plan. Through the tools, you are given access to the Git repository of the project that you are working on.'
        });

        const allMessages = [systemMessage, ...messages];

        const response = await this.modelInstance.invoke(allMessages, {
            recursionLimit: 25,
            configurable: {
                thread_id: this.threadId,
            },
        });

        return {
            ...state,
            messages: [...messages, response],
            iterations: iterations + 1
        };
    }
    private async toolsNode(state) {
        const { messages } = state;
        const lastMessage = messages[messages.length - 1];
        
        const toolCalls = lastMessage.tool_calls || [];
        const toolMessages = [];

        for (const toolCall of toolCalls) {
            const tool = this.tools.find(t => t.name === toolCall.name);
            if (tool) {
                try {
                    const result = await tool.invoke(toolCall.args);
                    toolMessages.push({
                        role: "tool",
                        content: JSON.stringify(result),
                        tool_call_id: toolCall.id
                    });
                } catch (error) {
                    toolMessages.push({
                        role: "tool",
                        content: `Error executing ${toolCall.name}: ${error.message}`,
                        tool_call_id: toolCall.id
                    });
                }
            }
        }

        return {
            ...state,
            messages: [...messages, ...toolMessages]
        };
    }
    private async formatResponseNode(state) {
        const { messages } = state;

        const response = await createModelInstance({
            temperature: 0,
            maxRetries: 3,
            verbose: false
        }).withStructuredOutput(this.responseSchema).invoke(
            [
                ...messages,
                new HumanMessage({
                    content: "Please format the response as valid JSON."
                }),
            ],
            {
                recursionLimit: 25,
                configurable: {
                    thread_id: this.threadId,
                },
            }
        );
        
        const lastMessage = response.text;

        try {
            const parsedResponse = JSON.parse(lastMessage);
            const validatedResponse = this.responseSchema.parse(parsedResponse);
            
            return {
                ...state,
                plan: validatedResponse.plan,
                messages: [...messages, new AIMessage({
                    content: JSON.stringify(validatedResponse)
                })]
            };
        } catch (error) {
            return {
                ...state,
                messages: [...messages, new HumanMessage({
                    content: `The response format is invalid. Please provide a valid JSON response with the required structure: ${JSON.stringify(this.responseSchema.shape)}`
                })]
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
        
        if (lastMessage.content && 
            (lastMessage.content.includes('{') || lastMessage.content.includes('```'))) {
            return "format";
        }
        
        return "end";
    }

    async init(): Promise<DevelopmentTeamManager> {
        await cloneRepository(
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
    async invoke(options: DevelopmentTeamManagerInvocationOptions): Promise<boolean> {
        const { task, data } = options;

        switch (task) {
            case DevelopmentTeamManagerInvocationTask.AddFeature:
                return this.addFeature(data as DevelopmentTeamManagerAddFeatureOptions);
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
    private async addFeature(data: DevelopmentTeamManagerAddFeatureOptions): Promise<boolean> {
        // Get team manager to analyse codebase and generate a detailed plan for the feature
        const initialState = {
            ...AgentState,
            messages: [new HumanMessage({ content: `Add a feature with the following description: ${data.description}\n\nCreate a detailed plan to achieve this task using the tools given to you including references to which files to modify, create or delete etc.` })],
            iterations: 0
        };

        const planResponse = await this.agentInstance.invoke(initialState);
        const plan = planResponse.plan;

        if (!plan) {
            throw new Error("Failed to generate a plan");
        }

        // console.log(`Generated plan: ${plan}`);

        return true;
    }

    /**
     * Fixes a bug based on the provided data.
     * @param {DevelopmentTeamManagerFixBugOptions} data - The data for the bug to be fixed.
     * @returns {Promise<boolean>} True if the bug was fixed successfully, false otherwise.
     */
    private async fixBug(data: DevelopmentTeamManagerFixBugOptions): Promise<boolean> {
        // Implement the logic for fixing a bug
        return true;
    }
};