"""image-match: Match exposure and color balance across images."""

import image_match.cli


def main() -> None:
    """CLI entrypoint delegating to the cli module."""
    image_match.cli.run()
