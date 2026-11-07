FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies and code first (changes less often than prompts/dataset).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# The eval inputs. Mount over these to evaluate a different checkout.
COPY prompts ./prompts
COPY golden ./golden
COPY config ./config

RUN useradd --create-home --uid 1000 runner \
 && mkdir -p .regress out \
 && chown -R runner:runner /app
USER runner

# Runtime configuration (all optional except the API key for real runs):
#   OPENAI_API_KEY        LLM provider key
#   SLACK_WEBHOOK_URL     Slack incoming webhook for alerts
#   LLMREG_PROVIDER       "mock" for an offline run
#   LLMREG_OVERALL_WARN / LLMREG_OVERALL_CRITICAL / LLMREG_<SECTION>_<KEY>   threshold overrides
#   LLMREG_REPORT_URL     link to the hosted report, used in Slack and PR comments
VOLUME ["/app/.regress", "/app/out"]

ENTRYPOINT ["llm-regress"]
CMD ["run", "--track", "local"]
