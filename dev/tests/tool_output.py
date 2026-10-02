"""Tool-call grammars and output as a request produces them, for tests."""

from server import tool_schema


def argument_grammar(schema):
    """The argument grammar of a tool whose parameters are `schema`, framed
    within a request's budget as ToolPolicy frames it."""
    policy = tool_schema.ToolPolicy({}, {"tool": schema}, False, True)
    return tool_schema._argument_grammar(policy.argument_schemas["tool"])
