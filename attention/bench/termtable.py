# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. mlops; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Plain-text tables for the benchmarks' terminal output, printed row by row as results arrive."""


class Table:
    """columns: (title, width, align) with align "<" (text) or ">" (numbers); a column is at least as wide as its
    title.  Prints the header on construction, one line per row()."""

    def __init__(self, columns):
        self.cols = [(t, max(w, len(t)), al) for t, w, al in columns]
        print("  ".join(f"{t:{al}{w}}" for t, w, al in self.cols))
        print("  ".join("-" * w for _, w, _ in self.cols))

    def row(self, *cells):
        print("  ".join(f"{str(c):{al}{w}}" for c, (_, w, al) in zip(cells, self.cols)), flush=True)
