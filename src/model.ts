/**
 * @module model
 * @file This module defines utility functions for creating instances of LLMs for agents.
 */

import { ChatOllama } from "@langchain/ollama";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";

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
export function createModelInstance(options: CreateModelInstanceOptions): BaseChatModel {
    const {
        modelName = "qwen3",
        temperature = 0.7,
        maxRetries = 3,
        verbose = false,
    } = options;

    return new ChatOllama({
        model: modelName,
        temperature: temperature,
        maxRetries: maxRetries,
        lowVram: true,
        verbose: verbose,
    });
}