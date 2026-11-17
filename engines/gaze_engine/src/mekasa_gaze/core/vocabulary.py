"""Editable vocabulary tree for composing messages with the eyes (``config/vocabulario.txt``).

The format is plain indented text so a family member or clinician can edit it without code::

    @hablar
      NECESITO | Necesito
        COMER Y BEBER |
          AGUA | agua

Each menu holds at most three entries: the fourth cell of the 2x2 board is always reserved for
navigation (VOLVER). The 2x2 grid is kept because it is the layout whose accuracy Mekasa has
measured; deeper menus trade a few extra selections for that reliability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_CHOICES = 3
MAX_LABEL_LENGTH = 24
_INDENT = 2
_GENDER = re.compile(r"\{([^{}|]*)\|([^{}|]*)\}")
_SECTIONS = ("hablar", "urgente", "intensidad")
_OPTIONS = ("intensidad",)


@dataclass(frozen=True, slots=True)
class Entry:
    """One menu entry: a category (has children) or a phrase (leaf)."""

    label: str
    say: str = ""
    children: tuple[Entry, ...] = ()
    modifier: str | None = None

    @property
    def is_menu(self) -> bool:
        return bool(self.children)


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """Parsed vocabulary: the composition tree, urgent messages and modifier sets."""

    talk: Entry
    urgent: Entry
    modifiers: dict[str, tuple[Entry, ...]] = field(default_factory=dict)


def apply_gender(text: str, gender: str) -> str:
    """Resolve ``{masculine|feminine}`` alternatives (``gender`` is ``m`` or ``f``)."""
    index = 1 if gender == "f" else 0
    return _GENDER.sub(lambda match: match.group(index + 1), text)


@dataclass
class _Draft:
    label: str
    say: str
    modifier: str | None
    line: int
    children: list[_Draft] = field(default_factory=list)

    def freeze(self) -> Entry:
        return Entry(self.label, self.say, tuple(c.freeze() for c in self.children), self.modifier)


def parse_vocabulary(text: str, *, gender: str = "m") -> Vocabulary:
    """Parse the indented vocabulary format.

    Raises:
        ValueError: With the offending line number for any structural problem.
    """
    if gender not in ("m", "f"):
        raise ValueError("gender must be 'm' or 'f'")
    roots: dict[str, _Draft] = {}
    stack: list[tuple[int, _Draft]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise ValueError(f"línea {number}: use espacios, no tabulaciones")
        indent = len(raw) - len(raw.lstrip(" "))
        if indent % _INDENT:
            raise ValueError(f"línea {number}: la sangría debe ser de {_INDENT} espacios")
        level = indent // _INDENT
        fields = [part.strip() for part in apply_gender(raw.strip(), gender).split("|")]
        label = fields[0]
        say = fields[1] if len(fields) > 1 else ""
        option = fields[2] if len(fields) > 2 else ""
        if len(fields) > 3:
            raise ValueError(f"línea {number}: demasiados '|'")
        if label.startswith("@"):
            if level != 0:
                raise ValueError(f"línea {number}: las secciones @ van sin sangría")
            name = label[1:].strip().lower()
            if name not in _SECTIONS:
                raise ValueError(f"línea {number}: sección desconocida '{name}'")
            if name in roots:
                raise ValueError(f"línea {number}: sección '{name}' repetida")
            node = _Draft(name.upper(), say, None, number)
            roots[name] = node
            stack = [(0, node)]
            continue
        if not stack:
            raise ValueError(f"línea {number}: falta una sección (@hablar, @urgente, ...)")
        if option and option not in _OPTIONS:
            raise ValueError(f"línea {number}: opción desconocida '{option}'")
        if not label:
            raise ValueError(f"línea {number}: falta la etiqueta")
        if len(label) > MAX_LABEL_LENGTH:
            raise ValueError(f"línea {number}: etiqueta de más de {MAX_LABEL_LENGTH} letras")
        while stack and stack[-1][0] >= level:
            stack.pop()
        if not stack or stack[-1][0] != level - 1:
            raise ValueError(f"línea {number}: sangría inesperada")
        parent = stack[-1][1]
        node = _Draft(label, say, option or None, number)
        parent.children.append(node)
        stack.append((level, node))
    missing = [name for name in _SECTIONS if name not in roots]
    if missing:
        raise ValueError(f"faltan secciones: {', '.join('@' + name for name in missing)}")
    _check(roots["hablar"])
    _check(roots["urgente"])
    intensity = roots["intensidad"]
    if not 1 <= len(intensity.children) <= MAX_CHOICES or any(
        c.children for c in intensity.children
    ):
        raise ValueError("@intensidad necesita de 1 a 3 opciones simples")
    if not roots["urgente"].say:
        raise ValueError("@urgente necesita un texto: es lo que se dice al entrar")
    return Vocabulary(
        talk=roots["hablar"].freeze(),
        urgent=roots["urgente"].freeze(),
        modifiers={"intensidad": tuple(child.freeze() for child in intensity.children)},
    )


def _check(node: _Draft) -> None:
    if len(node.children) > MAX_CHOICES:
        raise ValueError(
            f"línea {node.line}: '{node.label}' tiene {len(node.children)} opciones; "
            f"el máximo es {MAX_CHOICES} (use un submenú MÁS…)"
        )
    if node.children and node.modifier:
        raise ValueError(f"línea {node.line}: 'intensidad' solo va en frases, no en menús")
    if not node.children and not node.say and node.line:
        raise ValueError(f"línea {node.line}: '{node.label}' no tiene texto ni submenú")
    for child in node.children:
        _check(child)


def load_vocabulary(path: Path, *, gender: str = "m") -> Vocabulary:
    return parse_vocabulary(path.read_text(encoding="utf-8-sig"), gender=gender)


__all__ = [
    "MAX_CHOICES",
    "Entry",
    "Vocabulary",
    "apply_gender",
    "load_vocabulary",
    "parse_vocabulary",
]
