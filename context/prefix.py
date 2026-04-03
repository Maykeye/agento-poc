from context.context_handler import ContextHandler
from typing import override

from context.context_handler import ContextMode, ContextEntry, LlmProto

CONTEXTS = {}


class PrefixHandler(ContextHandler):
    @override
    def mode(self):
        return ContextMode.PREFIX

    @override
    def update(self, path: str, text: str, oper: str, edit_chunk=None):
        del edit_chunk
        ContextEntry.last_id += 1
        id = f"CTX({ContextEntry.last_id})"

        if path in CONTEXTS:
            del CONTEXTS[path]
        CONTEXTS[path] = ContextEntry(path, text, id, oper)
        return {"context-change": {"path": path, "operation": oper, "id": id}}

    def prepare_current_llm(self, llm: LlmProto):
        messages = llm.messages()

        # allow 1st message to be the system
        idx = -1
        if messages[0]["role"] == "user":
            idx = 0
        elif len(messages) > 1 and messages[1]["role"] == "user":
            idx = 1

        if idx >= 0:
            if not messages[idx]["content"].startswith(self.context_header()):
                messages.insert(idx, {})
        else:
            raise ValueError(
                f"Can't find index for content message#{idx} is: {messages[idx:idx+1]}"
            )
        messages[idx] = llm.msg_user(self.format_context())

    def context_header(self):
        return "THIS IS AUTOMATICALLY GENERATED MESSAGE RESERVED FOR FILES CONTEXTS."

    def format_context(self):
        res = [
            self.context_header(),
            "Result of file context operations will be logged here. Eg if you read/edit/write file, it will be placed in this message.",
            "This message will be edited automatically, so even if you reply goes after it, the message content is up to date.",
            "Each file operation will be identified by `id`entifier.",
            "When you perform operation with any file, tool will give you the identifier and the identifier will be in the context together with path",
            "so you can focus on the fresh full file content",
            "",
        ]

        for ctx in CONTEXTS.values():
            res += [
                f"=== Path: {ctx.path} (last id: {ctx.id}; last operation: {ctx.operation}) CONTEXT START ==="
            ]
            res += ctx.text.splitlines()
            res += [
                f"=== Path: {ctx.path} (last id: {ctx.id}; last operation: {ctx.operation}) CONTEXT END ===\n"
            ]
        return "\n".join(res).rstrip()
