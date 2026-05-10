# PII-Privacy-Handler-for-LLMs

A privacy-focused middleware that detects and anonymizes Personally Identifiable Information (PII) before sending data to Large Language Models (LLMs), ensuring sensitive data never leaves your environment unprotected.

## Features
- 🔍 Detects PII entities (names, emails, phone numbers, addresses, etc.)
- 🔒 Anonymizes/masks sensitive data before LLM processing
- 🔄 Restores original data from LLM responses
- 🤖 Compatible with popular LLMs (OpenAI, Gemini, etc.)

## How It Works
1. Input text is scanned for PII
2. Detected PII is replaced with placeholders (e.g., `<NAME>`, `<EMAIL>`)
3. Sanitized text is sent to the LLM
4. LLM response is de-anonymized and returned to the user


