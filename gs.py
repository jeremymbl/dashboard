from pathlib import Path
import os
import sys

# Configuration --------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT_DIR / "all_sources.txt"
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "env", ".idea", ".mypy_cache",
    ".streamlit", "secrets", ".uv", "dist", "build",
}
MAX_BINARY_SCAN = 1024         # octets à scanner pour détecter un NUL
MAX_FILE_SIZE = 1_000_000      # 1 Mo : on ignore les fichiers plus gros

TEXT_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".toml", ".yml", ".yaml",
    ".html", ".css", ".md", ".txt", ".ini", ".cfg",
}

# ---------------------------------------------------------------------------

def is_binary(file_path: Path) -> bool:
    """Heuristic: binary if contains NUL within first bytes or too big."""
    try:
        if file_path.stat().st_size > MAX_FILE_SIZE:
            return True
        with file_path.open("rb") as f:
            chunk = f.read(MAX_BINARY_SCAN)
            return b"\x00" in chunk
    except OSError:
        return True

def should_skip(file_path: Path) -> bool:
    if any(part in SKIP_DIRS for part in file_path.parts):
        return True
    if file_path.is_dir():
        return True
    if file_path == OUTPUT_FILE:
        return True
    if file_path.suffix.lower() not in TEXT_EXTS:
        return True
    if is_binary(file_path):
        return True
    return False

def gather_sources() -> list[Path]:
    return [
        p for p in ROOT_DIR.rglob("*")
        if not should_skip(p)
    ]

def write_all_sources(paths: list[Path]) -> None:
    with OUTPUT_FILE.open("w", encoding="utf-8") as out:
        for p in sorted(paths):
            rel = p.relative_to(ROOT_DIR)
            out.write(f"\n=== {rel} ===\n")
            try:
                out.write(p.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:
                out.write(f"[Erreur de lecture : {exc}]\n")
    print(f"✅ Fichier généré : {OUTPUT_FILE}")

def main() -> None:
    sources = gather_sources()
    if not sources:
        print("Aucun fichier source trouvé ! Vérifie tes filtres.", file=sys.stderr)
        sys.exit(1)
    write_all_sources(sources)

if __name__ == "__main__":
    main()