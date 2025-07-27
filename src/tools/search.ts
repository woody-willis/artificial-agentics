/**
 * @module tools/search
 * @file This module provides agent tools for searching across the internet.
 */

import { SearxngSearch } from "@langchain/community/tools/searxng_search";

export const SearxSearch = () => {
    return new SearxngSearch({
        apiBase: process.env.SEARX_SEARCH_API_URL,
        params: {
            format: "json",
            language: "en",
            numResults: 5,
            safesearch: 0,
        },
    });
};
