from context.context_handler import ContextMode, ContextEntry, ContextHandler
from context.raw import RawHandler
from context.prefix import PrefixHandler
from context.suffix import SuffixHandler


def set_context_mode(value: ContextMode, reset_ctx_id=False):
    global _CONTEXT_HANDLER
    if reset_ctx_id:
        ContextEntry.last_id = -1
    match value:
        case ContextMode.RAW:
            _CONTEXT_HANDLER = RawHandler()
        case ContextMode.PREFIX:
            _CONTEXT_HANDLER = PrefixHandler()
        case ContextMode.SUFFIX:
            _CONTEXT_HANDLER = SuffixHandler()


_CONTEXT_HANDLER: ContextHandler = RawHandler()


def context_handler():
    return _CONTEXT_HANDLER
