"""Navigate the vocabulary with four gaze cells and compose a full message before speaking it.

Screens (each one fills the 2x2 board; cell order: top-left, top-right, bottom-left,
bottom-right):

* **Inicio:** SÍ, NO (spoken at once), HABLAR (or MENSAJE when a draft exists), URGENTE.
* **Menú:** up to three entries plus VOLVER. Choosing a phrase adds it to the draft.
* **Intensidad:** UN POCO / BASTANTE / MUCHO / SIGUIENTE, after phrases that ask for it.
* **Mensaje:** DECIR (speak the whole draft), AGREGAR (compose more), BORRAR ÚLTIMO, INICIO.
* **Urgente:** entering it speaks the urgent call and raises the alarm at once (one
  selection from Inicio); the three details are also spoken at once.

The navigator is pure and clock-free: the board runner feeds it confirmed cells and acts on the
returned :class:`Outcome`. Positional cells keep the dwell session's rearm-on-exit rule: after
a screen change, the cell just looked at cannot fire again until the gaze leaves it, so a
steady gaze can never cascade through several menus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from mekasa_gaze.core.vocabulary import Entry, Vocabulary

CELLS = 4
SEP = "\u203a"  # single right-pointing angle quotation mark
NAV_CELL = 3


class CellKind(StrEnum):
    """Visual role of a cell."""

    ANSWER = "ANSWER"  # SÍ / NO
    MENU = "MENU"  # opens a submenu
    PHRASE = "PHRASE"  # adds to the draft
    NAV = "NAV"  # VOLVER / INICIO / SIGUIENTE
    SPEAK = "SPEAK"  # DECIR
    DELETE = "DELETE"  # BORRAR ÚLTIMO
    URGENT = "URGENT"
    EMPTY = "EMPTY"


@dataclass(frozen=True, slots=True)
class Cell:
    label: str
    kind: CellKind


class Screen(StrEnum):
    HOME = "HOME"
    MENU = "MENU"
    MODIFIER = "MODIFIER"
    REVIEW = "REVIEW"
    URGENT = "URGENT"


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the runner must do after a selection."""

    speak: str | None = None
    alarm: bool = False
    screen_changed: bool = False


def sentence(text: str) -> str:
    """Capitalize and close a fragment as a sentence."""
    text = " ".join(text.split())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!¡¿" else text + "."


def _join(*fragments: str) -> str:
    result = ""
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if result and fragment[0] in ",.;:":
            result += fragment
        else:
            result = f"{result} {fragment}".strip()
    return result


def _title(label: str) -> str:
    return label[:1].upper() + label[1:].lower() if label else label


@dataclass(slots=True)
class Navigator:
    """State of the conversation board."""

    vocabulary: Vocabulary
    yes_phrase: str = "Sí"
    no_phrase: str = "No"
    yes_label: str = "SÍ"
    no_label: str = "NO"
    screen: Screen = Screen.HOME
    path: list[Entry] = field(default_factory=list)  # menus below the talk root
    draft: list[str] = field(default_factory=list)
    last_said: str | None = None
    _pending: str = ""
    _modifier: str | None = None

    # ------------------------------------------------------------------ view

    def cells(self) -> list[Cell]:
        """The four cells of the current screen."""
        if self.screen is Screen.HOME:
            talk = (
                Cell(f"MENSAJE ({len(self.draft)})", CellKind.SPEAK)
                if self.draft
                else Cell("HABLAR", CellKind.MENU)
            )
            return [
                Cell(self.yes_label, CellKind.ANSWER),
                Cell(self.no_label, CellKind.ANSWER),
                talk,
                Cell("URGENTE", CellKind.URGENT),
            ]
        if self.screen is Screen.REVIEW:
            return [
                Cell("DECIR", CellKind.SPEAK),
                Cell("AGREGAR", CellKind.MENU),
                Cell("BORRAR ÚLTIMO", CellKind.DELETE),
                Cell("INICIO", CellKind.NAV),
            ]
        if self.screen is Screen.MODIFIER:
            entries = self.vocabulary.modifiers[self._modifier or "intensidad"]
            return [*self._entry_cells(entries), Cell("SIGUIENTE", CellKind.NAV)]
        if self.screen is Screen.URGENT:
            entries = self.vocabulary.urgent.children
            cells = [Cell(entry.label, CellKind.URGENT) for entry in entries]
            cells += [Cell("", CellKind.EMPTY)] * (NAV_CELL - len(cells))
            return [*cells, Cell("VOLVER", CellKind.NAV)]
        return [*self._entry_cells(self._menu().children), Cell("VOLVER", CellKind.NAV)]

    def _entry_cells(self, entries: tuple[Entry, ...]) -> list[Cell]:
        cells = [
            Cell(
                f"{entry.label} {SEP}"
                if entry.is_menu and not entry.label.endswith("…")
                else entry.label,
                CellKind.MENU if entry.is_menu else CellKind.PHRASE,
            )
            for entry in entries
        ]
        return cells + [Cell("", CellKind.EMPTY)] * (NAV_CELL - len(cells))

    def _menu(self) -> Entry:
        return self.path[-1] if self.path else self.vocabulary.talk

    @property
    def draft_text(self) -> str:
        return " ".join(self.draft)

    def breadcrumb(self) -> str:
        """Where the person is, for the status line."""
        if self.screen is Screen.HOME:
            return "Inicio"
        if self.screen is Screen.REVIEW:
            return f"Inicio {SEP} Mensaje"
        if self.screen is Screen.URGENT:
            return f"Inicio {SEP} Urgente"
        names = ["Inicio", "Hablar", *(_title(entry.label.rstrip("…")) for entry in self.path)]
        if self.screen is Screen.MODIFIER:
            names.append("¿Cuánto?")
        return f" {SEP} ".join(names)

    def message_line(self) -> str | None:
        """Text for the message bar: the draft while composing, else the last thing said."""
        if self.draft and self.screen is not Screen.URGENT:
            preview = self.draft_text
            if self.screen is Screen.MODIFIER:
                preview = _join(preview, sentence(self._pending))
            return f"Mensaje: {preview}"
        if self.screen is Screen.MODIFIER:
            return f"Mensaje: {sentence(self._pending)}"
        return self.last_said

    # ------------------------------------------------------------------ actions

    def home(self) -> Outcome:
        changed = self.screen is not Screen.HOME
        self.screen, self.path, self._pending, self._modifier = Screen.HOME, [], "", None
        return Outcome(screen_changed=changed)

    def back(self) -> Outcome:
        """VOLVER: one level up (keyboard shortcut too)."""
        if self.screen is Screen.MENU and self.path:
            self.path.pop()
            return Outcome(screen_changed=True)
        if self.screen is Screen.MODIFIER:
            self._commit(self._pending)
            return Outcome(screen_changed=True)
        return self.home()

    def select(self, index: int) -> Outcome:
        """Act on the confirmed cell ``index`` (0..3)."""
        if not 0 <= index < CELLS:
            return Outcome()
        cell = self.cells()[index]
        if cell.kind is CellKind.EMPTY:
            return Outcome()
        if self.screen is Screen.HOME:
            return self._home(index)
        if self.screen is Screen.REVIEW:
            return self._review(index)
        if self.screen is Screen.MODIFIER:
            return self._modify(index)
        if self.screen is Screen.URGENT:
            if index == NAV_CELL:
                return self.home()
            text = sentence(self.vocabulary.urgent.children[index].say)
            self.last_said = text
            return Outcome(speak=text, alarm=True)
        if index == NAV_CELL:
            return self.back()
        entry = self._menu().children[index]
        if entry.is_menu:
            self.path.append(entry)
            return Outcome(screen_changed=True)
        fragment = _join(*(item.say for item in self.path), entry.say)
        if entry.modifier is not None and entry.modifier in self.vocabulary.modifiers:
            self._pending, self._modifier = fragment, entry.modifier
            self.screen = Screen.MODIFIER
            return Outcome(screen_changed=True)
        self._commit(fragment)
        return Outcome(screen_changed=True)

    def _home(self, index: int) -> Outcome:
        if index in (0, 1):
            text = self.yes_phrase if index == 0 else self.no_phrase
            self.last_said = text
            return Outcome(speak=text)
        if index == 2:
            self.screen = Screen.REVIEW if self.draft else Screen.MENU
            self.path = []
            return Outcome(screen_changed=True)
        self.screen = Screen.URGENT
        text = sentence(self.vocabulary.urgent.say)
        self.last_said = text
        return Outcome(speak=text, alarm=True, screen_changed=True)

    def _review(self, index: int) -> Outcome:
        if index == 0:
            text = self.draft_text
            self.last_said = text
            self.draft.clear()
            self.home()
            return Outcome(speak=text, screen_changed=True)
        if index == 1:
            self.screen, self.path = Screen.MENU, []
            return Outcome(screen_changed=True)
        if index == 2:
            if self.draft:
                self.draft.pop()
            if not self.draft:
                self.screen, self.path = Screen.MENU, []
            return Outcome(screen_changed=True)
        return self.home()

    def _modify(self, index: int) -> Outcome:
        if index == NAV_CELL:
            self._commit(self._pending)
        else:
            modifier = self.vocabulary.modifiers[self._modifier or "intensidad"][index]
            self._commit(_join(self._pending, modifier.say))
        return Outcome(screen_changed=True)

    def _commit(self, fragment: str) -> None:
        text = sentence(fragment)
        if text:
            self.draft.append(text)
        self._pending, self._modifier = "", None
        self.screen, self.path = Screen.REVIEW, []


__all__ = ["CELLS", "Cell", "CellKind", "Navigator", "Outcome", "Screen", "sentence"]
