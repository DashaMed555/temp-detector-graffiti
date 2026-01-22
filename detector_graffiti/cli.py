from typing import List


def extract_command(argv: List[str]) -> str:
    if len(argv) < 2:
        raise ValueError(
            "Usage: graffiti-detector [train|inference] [overrides...]"
        )

    command = argv[1]
    del argv[1]
    return command
