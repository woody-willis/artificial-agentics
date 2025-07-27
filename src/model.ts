/**
 * @module model
 * @file This module defines utility functions for creating instances of LLMs for agents.
 */

import { ChatOllama } from "@langchain/ollama";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { Embeddings } from "@langchain/core/embeddings";

interface CreateModelInstanceOptions {
    modelName?: string;
    temperature?: number;
    maxRetries?: number;
    verbose?: boolean;
}

/**
 * Creates an instance of a language model (LLM) for an agent.
 * @param {CreateModelInstanceOptions} options - The options for creating the model instance.
 * @returns {BaseChatModel} The created model instance.
 */
// export function createModelInstance(options: CreateModelInstanceOptions): BaseChatModel {
//     const {
//         modelName = "qwen3",
//         temperature = 0.7,
//         maxRetries = 3,
//         verbose = false,
//     } = options;

//     return new ChatOllama({
//         model: modelName,
//         temperature: temperature,
//         maxRetries: maxRetries,
//         lowVram: true,
//         verbose: verbose,
//     });
// }
export function createModelInstance(
    options: CreateModelInstanceOptions
): BaseChatModel {
    const {
        // modelName = "Qwen/Qwen3-32B",
        // modelName = "gemini-2.0-flash-lite",
        modelName = "qwen3",
        temperature = 0.7,
        maxRetries = 3,
        verbose = false,
    } = options;

    return new ChatOpenAI({
        model: modelName,
        temperature: temperature,
        maxRetries: maxRetries,
        verbose: verbose,
        apiKey: process.env.OPENAI_API_KEY,
        configuration: { baseURL: process.env.OPENAI_API_URL },
    });
}

export function createEmbeddingModelInstance(
    options: CreateModelInstanceOptions
): Embeddings {
    const {
        modelName = "gemini-embedding-001",
        maxRetries = 3,
        verbose = false,
    } = options;

    return new OpenAIEmbeddings({
        model: modelName,
        maxRetries: maxRetries,
        verbose: verbose,
        apiKey: process.env.OPENAI_API_KEY,
        configuration: { baseURL: process.env.OPENAI_API_URL },
    });
}
