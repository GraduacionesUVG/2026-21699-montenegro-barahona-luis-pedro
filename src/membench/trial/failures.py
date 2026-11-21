from membench.trial.models import ErrorCause


def exception_causes(
    error: BaseException,
    *,
    max_depth: int = 5,
) -> tuple[ErrorCause, ...]:
    causes: list[ErrorCause] = []
    current = error.__cause__
    seen: set[int] = set()

    while current is not None and len(causes) < max_depth:
        if id(current) in seen:
            break
        seen.add(id(current))

        causes.append(
            ErrorCause(
                depth=len(causes) + 1,
                error_type=type(current).__name__,
                error_message=str(current) or type(current).__name__,
                module=type(current).__module__,
            )
        )

        current = current.__cause__

    return tuple(causes)