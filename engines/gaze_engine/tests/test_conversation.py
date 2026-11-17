"""Vocabulary file and conversation navigator (compose a full message with four cells)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mekasa_gaze.core.vocabulary import load_vocabulary, parse_vocabulary
from mekasa_gaze.tracking.conversation import CellKind, Navigator, Screen, sentence

ROOT = Path(__file__).resolve().parents[3]

MINI = """
@urgente | ¡Ayuda!
  NO RESPIRO | No puedo respirar
@intensidad
  POCO | , un poco
  MUCHO | , mucho
@hablar
  NECESITO | Necesito
    AGUA | agua
    MÁS… |
      MANTA | una manta
  ME DUELE | Me duele
    CABEZA | la cabeza | intensidad
  ÁNIMO |
    CANSANCIO | Estoy {cansado|cansada}
"""


def _labels(navigator: Navigator) -> list[str]:
    return [cell.label for cell in navigator.cells()]


def _go(navigator: Navigator, *labels: str) -> None:
    for label in labels:
        navigator.select(_labels(navigator).index(label))


def test_repository_vocabulary_is_valid_for_both_genders() -> None:
    for gender in ("m", "f"):
        vocabulary = load_vocabulary(ROOT / "config" / "vocabulario.txt", gender=gender)
        assert len(vocabulary.talk.children) == 3
        assert vocabulary.urgent.say
        assert len(vocabulary.modifiers["intensidad"]) == 3

        def walk(entry, depth=0):  # type: ignore[no-untyped-def]
            assert len(entry.children) <= 3
            assert "{" not in entry.label and "{" not in entry.say
            for child in entry.children:
                walk(child, depth + 1)

        walk(vocabulary.talk)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("@hablar\n  A | a\n  B | b\n  C | c\n  D | d\n", "máximo"),
        ("@hablar\n   A | a\n", "sangría"),
        ("@hablar\n  A | a | volando\n", "opción desconocida"),
        ("@hablar\n  A |\n", "no tiene texto"),
        ("  A | a\n", "falta una sección"),
        ("@chau\n", "sección desconocida"),
        ("@hablar\n  A | a\n", "faltan secciones"),
    ],
)
def test_vocabulary_errors_name_the_line(text: str, error: str) -> None:
    if not error.startswith("falta"):
        text = "@urgente | ¡Ayuda!\n  X | x\n@intensidad\n  P | , poco\n" + text
    with pytest.raises(ValueError, match=error):
        parse_vocabulary(text)


def test_compose_with_intensity_and_speak_whole_message() -> None:
    navigator = Navigator(parse_vocabulary(MINI))
    assert _labels(navigator) == ["SÍ", "NO", "HABLAR", "URGENTE"]
    _go(navigator, "HABLAR", "ME DUELE \u203a", "CABEZA")
    assert navigator.screen is Screen.MODIFIER
    assert navigator.message_line() == "Mensaje: Me duele la cabeza."
    _go(navigator, "MUCHO")
    assert navigator.screen is Screen.REVIEW
    _go(navigator, "AGREGAR", "NECESITO \u203a", "MÁS…", "MANTA")
    assert navigator.draft == ["Me duele la cabeza, mucho.", "Necesito una manta."]
    navigator.home()
    assert _labels(navigator)[2] == "MENSAJE (2)"
    _go(navigator, "MENSAJE (2)")
    outcome = navigator.select(0)  # DECIR
    assert outcome.speak == "Me duele la cabeza, mucho. Necesito una manta."
    assert navigator.screen is Screen.HOME and not navigator.draft
    assert navigator.message_line() == outcome.speak  # stays visible after speaking


def test_skip_intensity_delete_last_and_back() -> None:
    navigator = Navigator(parse_vocabulary(MINI))
    _go(navigator, "HABLAR", "ME DUELE \u203a", "CABEZA", "SIGUIENTE")
    assert navigator.draft == ["Me duele la cabeza."]
    _go(navigator, "BORRAR ÚLTIMO")
    assert not navigator.draft and navigator.screen is Screen.MENU
    _go(navigator, "NECESITO \u203a", "MÁS…")
    assert navigator.breadcrumb() == "Inicio \u203a Hablar \u203a Necesito \u203a Más"
    _go(navigator, "VOLVER", "VOLVER", "VOLVER")
    assert navigator.screen is Screen.HOME


def test_yes_no_and_urgent_act_at_once_with_alarm() -> None:
    navigator = Navigator(parse_vocabulary(MINI))
    assert navigator.select(0).speak == "Sí"
    urgent = navigator.select(3)
    assert urgent.speak == "¡Ayuda!" and urgent.alarm
    assert _labels(navigator) == ["NO RESPIRO", "", "", "VOLVER"]
    assert navigator.cells()[1].kind is CellKind.EMPTY
    assert navigator.select(1).speak is None  # empty cells do nothing
    detail = navigator.select(0)
    assert detail.speak == "No puedo respirar." and detail.alarm


def test_gender_and_sentences() -> None:
    feminine = Navigator(parse_vocabulary(MINI, gender="f"))
    _go(feminine, "HABLAR", "ÁNIMO \u203a", "CANSANCIO")
    assert feminine.draft == ["Estoy cansada."]
    assert sentence("¿qué hora es?") == "¿qué hora es?"
    assert sentence("necesito agua") == "Necesito agua."
