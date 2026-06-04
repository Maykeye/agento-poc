from agento.context.context_handler import ContextMode, ContextEntry, ContextHandler
from agento.context.raw import RawHandler
from agento.context.suffix import SuffixHandler


def set_context_mode(value: ContextMode, reset_ctx_id=False):
    global _CONTEXT_HANDLER
    if reset_ctx_id:
        ContextEntry.last_id = -1
    match value:
        case ContextMode.RAW:
            _CONTEXT_HANDLER = RawHandler()
        case ContextMode.SUFFIX:
            _CONTEXT_HANDLER = SuffixHandler()


_CONTEXT_HANDLER: ContextHandler = RawHandler()


def context_handler():
    return _CONTEXT_HANDLER


def llm_instance():
    """Get the current LLM instance."""
    from agento.llm import LLM

    return LLM.INSTANCES[-1].llm if LLM.INSTANCES else None
