"""BFCL prompting-mode handler for a chat model served on our own vLLM.

Imported by ``__main__.py`` in this directory, under the ``.venv-bfcl``
interpreter. It is a separate module rather than a nested class because
``OSSHandler`` inherits ``overrides.EnforceOverrides``: every method that
shadows a base method *must* carry ``@override``, and ``@override`` resolves the
base class by inspecting the defining frame, which fails for a class defined
inside a function ("No super class method found"). Keeping the class at module
level in its own file satisfies both that and ``__main__``'s need to stay
importable without ``bfcl_eval`` present (see the fixture in
``tests/test_expert_failure.py``).

Why this handler exists at all: BFCL's own GLM entries either target Zhipu's
cloud API (``GLMAPIHandler``) or hard-code ``glm-4-9b-chat``'s prompt format in
Python, and neither describes a GLM-4.5 checkpoint on a local server.
``OSSHandler`` already provides the whole prompting pipeline -- BFCL's default
system prompt carrying the function docs, the ``/v1/completions`` call, and
``default_decode_ast_prompting`` to parse ``[func(a=1)]`` back out. The only
model-specific piece is turning messages into a prompt string, and the
checkpoint's own ``chat_template.jinja`` is the authoritative answer, so it is
applied through the tokenizer instead of transcribed.
"""

from __future__ import annotations

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override

# Set by __main__ before the handler is constructed. Module-level because BFCL
# instantiates the handler itself, from the registry entry, with a fixed
# argument list -- there is no hook for passing extra options through.
ENABLE_THINKING: bool | None = False
STOP_TOKEN_IDS: tuple[int, ...] = ()


class LocalChatTemplateHandler(OSSHandler):
    """Prompting-mode handler that formats with the checkpoint's chat template."""

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(
            model_name, temperature, registry_name, is_fc_model, dtype=dtype, **kwargs
        )
        self.enable_thinking = ENABLE_THINKING
        if STOP_TOKEN_IDS:
            # Forwarded to vLLM as `extra_body["stop_token_ids"]` by
            # OSSHandler._query_prompting. GLM-4.5 has three EOS ids and its
            # generation_config.json lists them, but being explicit costs
            # nothing and covers a server started without that config.
            self.stop_token_ids = list(STOP_TOKEN_IDS)

    @override
    def _format_prompt(self, messages, function):
        template_kwargs = {}
        if self.enable_thinking is not None:
            # GLM-4.5's template renders this as a `/nothink` suffix on the last
            # user message plus a pre-filled `<think></think>` after the
            # generation prompt, so the model answers directly. A template that
            # does not declare the variable ignores it.
            template_kwargs["enable_thinking"] = self.enable_thinking
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
