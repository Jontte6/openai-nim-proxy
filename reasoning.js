// reasoning.js
// Backend-specific "thinking" control: which chat_template_kwargs/top-level
// fields each model needs to toggle reasoning, and how to extract reasoning
// text from responses that embed it inline vs. as a structured field.

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');

// Everything returned by getReasoningPayload() is spread into the top-level
// JSON body sent to NIM. Do not wrap it in `extra_body` — that's an
// openai-SDK convention this proxy (raw axios) doesn't use.
//
// By default, reasoning is kept out of `content` and returned in a
// structured `reasoning`/`reasoning_content` field. Clients that want
// legacy inline <thinking> tags can opt in via `x-reasoning-format: inline`.

// Models that embed reasoning inline in `content` via delimiter tags instead
// of a structured field.
const CONTENT_DELIMITER_TAGS = {
  'minimaxai/minimax-m3': ['<mm:think>', '</mm:think>']
};

// Stateful parser for extracting reasoning blocks across streamed chunks.
class DelimiterParser {
  constructor(openTag, closeTag) {
    this.openTag = openTag;
    this.closeTag = closeTag;
    this.inThinking = false;
    this.buffer = '';
  }

  processChunk(chunk) {
    this.buffer += chunk;
    let content = '';
    let reasoning = '';

    while (true) {
      const targetTag = this.inThinking ? this.closeTag : this.openTag;
      const tagIndex = this.buffer.indexOf(targetTag);

      if (tagIndex !== -1) {
        const textBefore = this.buffer.substring(0, tagIndex);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.inThinking = !this.inThinking;
        this.buffer = this.buffer.substring(tagIndex + targetTag.length);
      } else {
        let partialLen = 0;
        const maxLen = Math.min(this.buffer.length, targetTag.length - 1);
        for (let i = maxLen; i > 0; i--) {
          if (targetTag.startsWith(this.buffer.substring(this.buffer.length - i))) {
            partialLen = i;
            break;
          }
        }
        const textBefore = this.buffer.substring(0, this.buffer.length - partialLen);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.buffer = this.buffer.substring(this.buffer.length - partialLen);
        break;
      }
    }

    return { content, reasoning };
  }

  flush() {
    let content = '';
    let reasoning = '';
    if (this.buffer) {
      if (this.inThinking) {
        reasoning += this.buffer;
      } else {
        content += this.buffer;
      }
      this.buffer = '';
    }
    return { content, reasoning };
  }
}

class StreamNormalizer {
  constructor(model) {
    this.model = model;
    this.parser = null;
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) {
      this.parser = new DelimiterParser(tags[0], tags[1]);
    }
  }

  processDelta(delta) {
    const normalizedDelta = { ...delta };
    let reasoning = normalizedDelta.reasoning || normalizedDelta.reasoning_content || '';
    let content = normalizedDelta.content || '';

    if (!reasoning && content && this.parser) {
      const parsed = this.parser.processChunk(content);
      reasoning = parsed.reasoning;
      content = parsed.content;
    }

    if (content) normalizedDelta.content = content;
    else delete normalizedDelta.content;

    if (reasoning) normalizedDelta.reasoning = reasoning;
    else delete normalizedDelta.reasoning;

    delete normalizedDelta.reasoning_content;
    return normalizedDelta;
  }

  flush() {
    if (!this.parser) return { content: '', reasoning: '' };
    return this.parser.flush();
  }
}

function normalizeNonStreamChoice(choice, model) {
  if (!choice) return choice;
  const message = choice.message || {};
  let reasoning = message.reasoning || message.reasoning_content || '';
  let content = message.content || '';

  if (!reasoning && content) {
    let parser = null;
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) {
      parser = new DelimiterParser(tags[0], tags[1]);
    }
    if (parser) {
      const parsed = parser.processChunk(content);
      const flushed = parser.flush();
      content = (parsed.content || '') + (flushed.content || '');
      reasoning = (parsed.reasoning || '') + (flushed.reasoning || '');
    }
  }

  const newMessage = { ...message };
  if (content) newMessage.content = content;
  if (reasoning) newMessage.reasoning = reasoning;
  delete newMessage.reasoning_content;

  return { ...choice, message: newMessage };
}

// Valid reasoning_effort values per model, where NIM enforces an enum.
// Values outside the set are dropped with a warning rather than forwarded.
const REASONING_EFFORT_ENUMS = {
  'openai/gpt-oss-120b': ['low', 'medium', 'high'],
  'openai/gpt-oss-20b': ['low', 'medium', 'high'],
  'deepseek-ai/deepseek-v4-flash-0731': ['low', 'high', 'max'],
  'deepseek-ai/deepseek-v4-pro-0813': ['low', 'high', 'max'],
  'nvidia/nemotron-3-super-120b-a12b': ['low'],
  'nvidia/nemotron-3-ultra-550b-a55b': ['low'],
  'minimaxai/minimax-m3': ['adaptive'],
  'moonshotai/kimi-k3': ['low', 'high', 'max'],
  'meta/muse-glimmer-30b': ['none', 'minimal', 'low', 'medium', 'high', 'max']
};

function validReasoningEffort(model, effort) {
  const allowed = REASONING_EFFORT_ENUMS[model];
  if (!allowed) return effort;
  if (allowed.includes(effort)) return effort;
  if (effort) {
    console.warn(`[REASONING] Dropping invalid reasoning_effort "${effort}" for ${model} (allowed: ${allowed.join(', ')})`);
  }
  return undefined;
}

// Resolves the client "off"/"on" override into an effective boolean. Shared
// with callWithFallback() so both agree on whether reasoning is active.
function resolveEffectiveThinking(enableThinking, clientReasoningEffort) {
  if (clientReasoningEffort === 'off') return false;
  if (clientReasoningEffort === 'on') return true;
  return enableThinking;
}

// Nemotron 3.5 Lightning has no boolean flag — only a top-level integer
// reasoning_budget (max reasoning tokens, -1 to 32768, default 16384). This
// tier mapping is this proxy's own approximation, not an NVIDIA-defined enum.
const NEMOTRON_LIGHTNING_BUDGET_MAP = { low: 2048, medium: 8192, high: 16384, max: -1 };

// Returns model-specific reasoning request payloads, spread into the
// top-level request body. reasoning_effort "off"/"on" overrides
// ENABLE_THINKING_MODE per-request for every model below.
function getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools) {
  enableThinking = resolveEffectiveThinking(enableThinking, clientReasoningEffort);

  const rawEffort = (clientReasoningEffort === 'off' || clientReasoningEffort === 'on')
    ? undefined
    : clientReasoningEffort;
  const effort = validReasoningEffort(model, rawEffort);

  switch (model) {
    case 'nvidia/nemotron-3-super-120b-a12b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      if (effort === 'low') payload.chat_template_kwargs.low_effort = true;
      return payload;
    }

    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      if (effort === 'low') payload.chat_template_kwargs.low_effort = true;
      if (hasTools) payload.chat_template_kwargs.force_nonempty_content = true; // unverified
      return payload;
    }

    case 'nvidia/nemotron-3.5-lightning-30b-a3b': {
      if (!enableThinking) return { reasoning_budget: 0 };
      return { reasoning_budget: NEMOTRON_LIGHTNING_BUDGET_MAP[effort] ?? 16384 };
    }

    case 'deepseek-ai/deepseek-v4-flash-0731':
    case 'deepseek-ai/deepseek-v4-pro-0813': {
      if (!enableThinking) {
        return { chat_template_kwargs: { thinking: false } };
      }
      return {
        chat_template_kwargs: {
          thinking: true,
          reasoning_effort: effort || 'high'
        }
      };
    }

    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b': {
      if (effort) return { reasoning_effort: effort };
      if (enableThinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'google/gemma-4-31b-it': {
      if (!enableThinking) return {};
      return {
        chat_template_kwargs: { enable_thinking: true },
        include_reasoning: SHOW_REASONING
      };
    }

    case 'meta/muse-glimmer-30b': {
      if (effort) return { reasoning_effort: effort };
      return { reasoning_effort: enableThinking ? 'high' : 'none' };
    }

    // poolside/laguna-xs-2.1: no documented reasoning param on NIM's hosted
    // endpoint (model, messages, temperature, top_p, max_tokens, stream
    // only). Falls through to default.

    case 'minimaxai/minimax-m3': {
      const thinkingMode = effort === 'adaptive'
        ? 'adaptive'
        : (enableThinking ? 'enabled' : 'disabled');
      return { chat_template_kwargs: { thinking_mode: thinkingMode } };
    }

    case 'moonshotai/kimi-k3': {
      // No off-switch — omitting the field falls back to Kimi's own 'max'.
      if (effort) return { reasoning_effort: effort };
      return { reasoning_effort: enableThinking ? 'high' : 'low' };
    }

    default:
      return {};
  }
}

module.exports = {
  SHOW_REASONING,
  getReasoningPayload,
  resolveEffectiveThinking,
  StreamNormalizer,
  normalizeNonStreamChoice
};
