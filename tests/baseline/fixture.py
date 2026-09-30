"""Small committed project for later conversation and repository tests."""

from pathlib import Path

from tests.pipeline.support import git


def create_small_project(root: Path) -> Path:
    repository = root / "small-project"
    (repository / "src").mkdir(parents=True)
    (repository / "tests").mkdir()
    git(repository, "init")
    git(repository, "config", "user.name", "AI Agents Test")
    git(repository, "config", "user.email", "ai-agents-test@example.invalid")
    (repository / "README.md").write_text("Small project for AI-Agents baseline tests.\n", encoding="utf-8")
    (repository / "src" / "total.py").write_text(
        "def total(values):\n    return sum(values)\n", encoding="utf-8"
    )
    (repository / "tests" / "test_total.py").write_text(
        "from src.total import total\n\ndef test_total():\n    assert total([2, 3]) == 5\n",
        encoding="utf-8",
    )
    git(repository, "add", "README.md", "src/total.py", "tests/test_total.py")
    git(repository, "commit", "-m", "baseline fixture")
    return repository.resolve()
