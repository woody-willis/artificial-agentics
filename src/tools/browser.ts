/**
 * @module tools/browser
 * @file This module provides agent tools for browser operations and navigation.
 */

import { DynamicStructuredTool, DynamicTool } from "@langchain/core/tools";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";

import * as fs from "fs";
import * as fsp from "fs/promises";
import * as path from "path";
import { OllamaEmbeddings } from "@langchain/ollama";
import z from "zod";
import pino from "pino";
import { createEmbeddingModelInstance } from "../model";

import { Browser, Page } from "puppeteer";
import puppeteer from "puppeteer-extra";
import puppeteerStealth from "puppeteer-extra-plugin-stealth";
import puppeteerAdblocker from "puppeteer-extra-plugin-adblocker";

puppeteer.use(puppeteerStealth());
puppeteer.use(puppeteerAdblocker({ blockTrackers: true }));

const logger = pino({
    level: "info",
    transport:
        process.env.ENVIRONMENT === "production"
            ? undefined
            : { target: "pino-pretty", options: { colorize: true } },
});

interface InitializeBrowserResponse {
    browser: Browser;
    page: Page;
}

/**
 * Initializes a Puppeteer browser and page.
 * @returns A promise that resolves to an object containing the browser and page.
 */
export async function initializeBrowser(): Promise<InitializeBrowserResponse> {
    const browser = await puppeteer.launch({
        headless: false,
        defaultViewport: { width: 1280, height: 800 },
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800 });

    browser.on("targetcreated", async (target) => {
        const newPage = await target.page();
        if (newPage) {
            await page.goto(newPage.url());
            await newPage.close();
        }
    });

    await page.goto("https://google.com");

    return { browser, page };
}

export async function getMarkedPageB64(page: Page, threadId: string) {
    const pageMarkPath = "./src/agents/research/assets/mark-page.js";
    const pageMarkInjection = (await fsp.readFile(pageMarkPath)).toString();

    const frames = page.frames();
    for (const frame of frames) {
        await frame
            .waitForSelector("body")
            .then(() => frame.evaluate(pageMarkInjection))
            .catch(() => {
                /* already injected */
            });
    }

    const bboxes = [];

    try {
        await fsp.stat("./temp");
    } catch {
        await fsp.mkdir("./temp");
    }

    for (const frame of frames) {
        // Get x and y offset of frame from page
        const { x: xOffset, y: yOffset } = await frame.evaluate(() => {
            const frameElement = window.frameElement;
            if (!frameElement) return { x: 0, y: 0 };
            const rect = frameElement.getBoundingClientRect();
            return {
                x: rect.left + window.scrollX,
                y: rect.top + window.scrollY,
            };
        });

        try {
            const frameBboxes = (await frame.evaluate(
                `markPage(${bboxes.length}, ${xOffset}, ${yOffset});`
            )) as Object[];
            bboxes.push(...frameBboxes);
        } catch (error) {
            // ignore
        }
    }

    const screenshotPath = `./temp/${threadId}-page.png`;
    await page.screenshot({ path: screenshotPath });
    const base64Img = await fsp.readFile(screenshotPath, {
        encoding: "base64",
    });
    await fsp.unlink(screenshotPath);

    for (const frame of frames) {
        try {
            await frame.evaluate("unmarkPage();");
        } catch (error) {
            // ignore
        }
    }

    return { bboxes, base64Img };
}

export const Click = (page: Page) => {
    const schema = z.object({
        bbox_id: z.string().describe("The ID of the bounding box to click."),
    });

    return new DynamicStructuredTool({
        name: "click",
        description:
            "Clicks on a web element identified by its bounding box ID.",
        schema: schema,
        func: async (
            input: z.infer<typeof schema>,
            _,
            config
        ): Promise<string> => {
            const { bbox_id } = input;
            const { configurable } = config;
            const { bboxes } = configurable;

            const bbox = bboxes[bbox_id];
            if (!bbox) {
                throw new Error(`Bounding box with ID ${bbox_id} not found.`);
            }

            const { x, y } = bbox;

            await page.mouse.click(x, y);

            return `Clicked '${bbox.text}' (${page.url().split("?")[0]})`;
        },
    });
};

export const TypeText = (page: Page) => {
    const schema = z.object({
        bbox_id: z
            .string()
            .describe("The ID of the bounding box to type text into."),
        text: z.string().describe("The text to type into the bounding box."),
    });

    return new DynamicStructuredTool({
        name: "type",
        description:
            "Types text into a web element identified by its bounding box ID.",
        schema: schema,
        func: async (
            input: z.infer<typeof schema>,
            _,
            config
        ): Promise<string> => {
            const { bbox_id, text } = input;
            const { configurable } = config;
            const { bboxes } = configurable;

            const bbox = bboxes[bbox_id];
            if (!bbox) {
                throw new Error(`Bounding box with ID ${bbox_id} not found.`);
            }

            const { x, y } = bbox;

            await page.mouse.click(x, y);

            const actionKey =
                process.platform == "darwin" ? "MetaLeft" : "ControlLeft";
            await page.keyboard.down(actionKey);
            await page.keyboard.press("q");
            await page.keyboard.up(actionKey);
            await page.keyboard.press("Backspace");

            // Type like a human
            for (const char of text) {
                const randomMs = Math.floor(Math.random() * 100) + 50;
                await new Promise((resolve) => setTimeout(resolve, randomMs));

                await page.keyboard.type(char);
            }

            await page.keyboard.press("Enter");

            return `Typed '${text}' into box and submitted (${page.url().split("?")[0]})`;
        },
    });
};

export const Scroll = (page: Page) => {
    const schema = z.object({
        direction: z.enum(["up", "down"]).describe("The direction to scroll."),
    });

    return new DynamicStructuredTool({
        name: "scroll",
        description: "Scrolls the webpage in the specified direction.",
        schema: schema,
        func: async (
            input: z.infer<typeof schema>,
            _,
            config
        ): Promise<string> => {
            const { direction } = input;

            const amount = 0.75 * page.viewport().height;

            await page.evaluate(
                (dir, amt) => {
                    window.scrollBy(0, dir === "up" ? -amt : amt);
                },
                direction,
                amount
            );

            return `Scrolled ${direction}`;
        },
    });
};

export const Wait = () => {
    const schema = z.object({
        duration: z
            .number()
            .min(0)
            .describe("The duration to wait in seconds."),
    });

    return new DynamicStructuredTool({
        name: "wait",
        description: "Waits for a specified duration.",
        schema: schema,
        func: async (input: z.infer<typeof schema>): Promise<string> => {
            const { duration } = input;

            await new Promise((resolve) =>
                setTimeout(resolve, duration * 1000)
            );

            return `Waited for ${duration} seconds`;
        },
    });
};

export const GoBack = (page: Page) => {
    return new DynamicTool({
        name: "go_back",
        description: "Navigates back to the previous page.",
        func: async () => {
            await page.goBack();
            return `Navigated back to the previous page (${page.url().split("?")[0]})`;
        },
    });
};

export const ToGoogle = (page: Page) => {
    return new DynamicTool({
        name: "to_google",
        description: "Navigates to Google.",
        func: async () => {
            await page.goto("https://www.google.com");
            return "Navigated back to Google.";
        },
    });
};

export const browserTools = (page: Page) => {
    return [
        Click(page),
        TypeText(page),
        Scroll(page),
        Wait(),
        GoBack(page),
        ToGoogle(page),
    ];
};
