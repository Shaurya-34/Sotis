
def format_result(operation_name, value):
    if not isinstance(value, (int, float)):
        raise ValueError("Invalid format input")
    return f"[{operation_name} Output]: {value}"
