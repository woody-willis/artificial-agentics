/**
 * @module agents/research/voyager
 * @file This agent is able to use a web browser to navigate and extract information from web pages.
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
    ToolMessage,
} from "@langchain/core/messages";
import { Tool } from "@langchain/core/tools";

import { createModelInstance, modelTokenBucket } from "../../model";
import z from "zod";
import pino from "pino";
import { promises as fsp } from "fs";
import {
    browserTools,
    getMarkedPageB64,
    initializeBrowser,
} from "../../tools/browser";
import { Page, Browser } from "puppeteer";

const logger = pino({
    level: "info",
    transport:
        process.env.ENVIRONMENT === "production"
            ? undefined
            : { target: "pino-pretty", options: { colorize: true } },
});

export interface ResearchVoyagerInvocationOptions {
    question: string;
}

export interface ResearchVoyagerAskQuestionOptions {
    question: string;
}

const AgentState = Annotation.Root({
    question: Annotation({ reducer: (x, y) => y ?? x, default: () => null }),
    answer: Annotation({ reducer: (x, y) => y ?? x, default: () => null }),
    thought: Annotation({ reducer: (x, y) => y ?? x, default: () => null }),
    bboxes: Annotation({ reducer: (x, y) => y ?? x, default: () => [] }),
    toolCalls: Annotation({ reducer: (x, y) => y ?? x, default: () => [] }),
    completed: Annotation({ reducer: (x, y) => y ?? x, default: () => false }),
    iterations: Annotation({ reducer: (x, y) => y ?? x, default: () => 0 }),
    maxIterations: Annotation({ reducer: (x, y) => y ?? x, default: () => 25 }),
    scratchpad: Annotation({ reducer: (x, y) => y ?? x, default: () => [] }),
});

export class ResearchVoyager {
    private threadId: string;
    private tools: Tool[];
    private responseSchema = z.object({
        answer: z.string().describe("The answer to the given question."),
    });
    private modelInstance: BaseChatModel;
    private agentInstance: CompiledStateGraph<unknown, unknown>;
    private agentLogger: pino.Logger;

    private browser: Browser;
    private page: Page;

    /**
     * Initializes the ResearchVoyager as a ReAct agent.
     */
    constructor(verbose: boolean = false) {
        this.threadId = "research-voyager-" + Date.now().toString();

        this.agentLogger = logger.child({
            module: "agents/research/voyager",
            threadId: this.threadId,
        });

        this.modelInstance = createModelInstance({
            temperature: 0,
            maxRetries: 3,
            verbose: verbose,
        });

        this.agentInstance = this.createStateGraph();
    }

    private createStateGraph(): CompiledStateGraph<unknown, unknown> {
        const workflow = new StateGraph(AgentState);

        workflow.addNode("agent", this.agentNode.bind(this));
        workflow.addNode("tools", this.toolsNode.bind(this));

        workflow.addEdge(START, "agent");

        workflow.addConditionalEdges("agent", this.shouldContinue.bind(this), {
            tools: "tools",
            agent: "agent",
            end: END,
        });

        workflow.addEdge("tools", "agent");

        return workflow.compile();
    }

    private async agentNode(state) {
        const { question, scratchpad, iterations, maxIterations } = state;

        if (iterations >= maxIterations) {
            return { ...state };
        }

        try {
            await this.page.waitForNetworkIdle({ timeout: 5_000 });
        } catch (error) {
            this.agentLogger.error("Error waiting for network idle:", error);
        }

        const { bboxes, base64Img } = await getMarkedPageB64(
            this.page,
            this.threadId
        );

        const systemMessage = new SystemMessage({
            content:
                "Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. This screenshot will feature Numerical Labels placed in the TOP LEFT corner of each Web Element. Carefully analyze the visual information to identify the Numerical Label corresponding to the Web Element that requires interaction, then follow the guidelines and choose one of the following actions:\n\n1. Click a Web Element.\n2. Delete existing content in a textbox and then type content.\n3. Scroll up or down.\n4. Wait \n5. Go back\n7. Return to google to start over.\n8. Respond with the final answer\n\nYou must answer in STRICTLY the following format:\n- ANSWER; [content]\n\nKey Guidelines You MUST follow:\n\n*Action guidelines*\n1) Execute ONLY one action.\n2) When clicking or typing, ensure to select the correct bounding box.\n3) Numeric labels lie in the top-left corner of their corresponding bounding boxes and are colored the same.\n4) Only perform the relevant action via calling a tool.\n5) Use the 'return to google' tool to make another search.\n\n*Web Browsing Guidelines*\n1) Don't interact with useless web elements like Login, Sign-in, donation, feedback that appear in Webpages\n2) Select strategically to minimize time wasted.\n3) If an 'Are you a robot?' CAPTCHA appears, you must wait for a human to complete it for you. Wait 15 seconds in this event.\n4) Always reject cookies if given the option.\n\nALWAYS make only 1 tool call UNLESS you are outputting an answer.\n\nYour reply should strictly follow the format:\n\nThought: {{Your brief thoughts (briefly summarize the info that will help ANSWER)}}\nAction: {{One Action format you choose}}\n\nThen the User will provide:\nObservation: {{A labeled screenshot Given by User}}",
        });

        let scratchpadText = "Your previous actions:\n";
        for (let i = 0; i < scratchpad.length; i++) {
            scratchpadText += `${i + 1}. ${scratchpad[i]}\n`;
        }
        this.agentLogger.info(
            `Scratchpad for iteration ${iterations}:\n${scratchpadText}`
        );

        const messages = [
            systemMessage,
            new HumanMessage({
                content: `Here is the question you need to answer: ${question}`,
            }),
            new HumanMessage({ content: scratchpadText }),
            new HumanMessage({
                content: [
                    { type: "text", content: `Observation:` },
                    {
                        type: "image",
                        source_type: "url",
                        url: `data:image/png;base64,${base64Img}`,
                    },
                ],
            }),
        ];

        const estimateTokens = (msgs) => {
            const totalCharacters = msgs.reduce(
                (acc, msg) => acc + (msg.content?.length || 0),
                0
            );
            return Math.ceil(totalCharacters / 4);
        };

        let estimatedTokens = estimateTokens(messages);

        if (estimatedTokens > 100_000) {
            estimatedTokens = 100_000;
        }

        while (modelTokenBucket.take(estimatedTokens) !== 0) {
            const timeoutTime = modelTokenBucket.take(estimatedTokens);
            this.agentLogger.info(
                `Waiting for token bucket to refill: ${estimatedTokens} tokens needed (${timeoutTime}ms)`
            );
            await new Promise((resolve) => setTimeout(resolve, timeoutTime));
        }

        this.agentLogger.info(
            `Invoking agent with ${messages.length} messages and estimated ${estimatedTokens} tokens`
        );

        let response: AIMessage;

        for (let i = 0; i < 3; i++) {
            response = await this.modelInstance.invoke(messages, {
                recursionLimit: 100,
                configurable: { thread_id: this.threadId },
            });

            if (response.content.length > 0) {
                break;
            }
        }

        this.agentLogger.info(
            `Agent response length: ${response.content.length} (Iteration: ${iterations + 1})`
        );

        const hasThought =
            typeof response.content === "string"
                ? response.content.includes("Thought:")
                : JSON.stringify(response.content).includes("Thought:");

        let thought: string | null = null;

        if (hasThought) {
            thought = response.content
                .split("Thought:")[1]
                .trim()
                .split("\n")[0];
        }

        const completed =
            typeof response.content === "string"
                ? response.content.includes("ANSWER;")
                : JSON.stringify(response.content).includes("ANSWER;");

        let answer: string | undefined;
        if (completed) {
            answer = response.content.split("ANSWER;")[1].trim();
        }

        return {
            ...state,
            bboxes: bboxes || [],
            iterations: iterations + 1,
            toolCalls: response.tool_calls || [],
            completed: completed,
            answer: answer,
            thought: thought,
        };
    }
    private async toolsNode(state) {
        const { scratchpad, thought, toolCalls, bboxes } = state;

        if (toolCalls.length === 0) {
            this.agentLogger.info("No tool calls to process, returning state.");
            return { ...state };
        }

        const toolMessages = [];
        for (const toolCall of toolCalls) {
            const tool = this.tools.find((t) => t.name === toolCall.name);
            if (tool) {
                try {
                    const result = await tool.invoke(toolCall.args, {
                        configurable: { bboxes: bboxes },
                    });
                    this.agentLogger.info(
                        `Tool ${toolCall.name} executed successfully with result: ${JSON.stringify(result).substring(0, 100)}`
                    );
                    toolMessages.push(result + ` - ${thought}`);
                } catch (error) {
                    this.agentLogger.error(
                        `Error executing tool ${toolCall.name}: ${error.message}`
                    );
                    toolMessages.push(error.message + ` - ${thought}`);
                }
            }

            break;
        }

        return {
            ...state,
            toolCalls: [],
            scratchpad: [...scratchpad, ...toolMessages],
        };
    }
    private shouldContinue(state) {
        const { completed, answer, toolCalls, iterations, maxIterations } =
            state;

        if (iterations >= maxIterations) {
            return "end";
        }

        if (toolCalls && toolCalls.length > 0) {
            return "tools";
        }

        if (!answer) {
            return "agent";
        }

        return "end";
    }

    async init(): Promise<ResearchVoyager> {
        const { browser, page } = await initializeBrowser();

        this.browser = browser;
        this.page = page;

        this.tools = [...browserTools(this.page)];

        this.modelInstance.bindTools(this.tools);

        return this;
    }

    async dispose(): Promise<void> {
        await this.browser.close();
    }

    /**
     * Invokes the agent with the specified task and data.
     * @param {ResearchVoyagerInvocationOptions} options - The options for invoking the agent.
     * @returns {Promise<string>} The response from the agent.
     */
    async invoke(options: ResearchVoyagerInvocationOptions): Promise<string> {
        const { question } = options;

        return this.answerQuestion({ question });
    }

    /**
     * Answers a question using a web browser.
     * @param {ResearchVoyagerAskQuestionOptions} data - The data for the task to be completed.
     * @returns {Promise<string>} The answer from the agent.
     */
    private async answerQuestion(
        data: ResearchVoyagerAskQuestionOptions
    ): Promise<string> {
        // Get voyager to answer a question
        const initialState = {
            ...AgentState,
            iterations: 0,
            question: data.question,
        };

        const answerResponse = await this.agentInstance.invoke(initialState, {
            recursionLimit: 100,
        });
        const answer = answerResponse.answer;

        if (!answer) {
            throw new Error("Failed to generate an answer");
        }

        return answer;
    }
}
