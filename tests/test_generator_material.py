from __future__ import annotations

from newsroom.editorial.generator import GenerationContext, build_material
from newsroom.editorial.pipeline import _facts_from_base


# --- build_material (offline) -------------------------------------------------

def test_material_includes_facts_and_source():
    ctx = GenerationContext(
        title="НБУ знизив ставку",
        facts=["Облікова ставка знижена до 13%", "Рішення ухвалене 6 вересня"],
        source_excerpt="Нацбанк повідомив про зниження облікової ставки на пресконференції.",
    )
    material = build_material(ctx)
    assert "Подія: НБУ знизив ставку" in material
    assert "Облікова ставка знижена до 13%" in material
    assert "Факти" in material
    assert "Витяг із джерел" in material and "пресконференції" in material


def test_material_without_facts_falls_back_to_title_and_source():
    ctx = GenerationContext(title="Подія", source_excerpt="Деталі з джерела.")
    material = build_material(ctx)
    assert "Подія: Подія" in material and "Деталі з джерела." in material
    assert "Факти" not in material          # no facts section when there are none


def test_material_includes_rubric_for_register():
    ctx = GenerationContext(title="Шахтар зіграв внічию", rubrics=["sport"],
                            source_excerpt="Матч завершився 1:1.")
    material = build_material(ctx)
    assert "Рубрика: sport" in material          # register guidance so the model tunes the voice


def test_material_empty_context_is_empty():
    assert build_material(GenerationContext(title="")) == ""


# --- _facts_from_base (offline) -----------------------------------------------

def test_facts_sorted_by_confirmation_and_marks_divergence():
    fb = {"facts": [
        {"text": "Один-джерельний факт", "confirmed_by": 1, "divergent": False},
        {"text": "Підтверджений двома", "confirmed_by": 2, "divergent": False},
        {"text": "Розбіжні цифри", "confirmed_by": 2, "divergent": True},
        {"text": "", "confirmed_by": 5},          # dropped: empty text
    ]}
    facts = _facts_from_base(fb)
    assert facts[0] in ("Підтверджений двома", "Розбіжні цифри (джерела розходяться в цифрах)")
    assert "Один-джерельний факт" in facts
    assert any("розходяться в цифрах" in f for f in facts)
    assert all(f.strip() for f in facts)         # no empty facts


def test_facts_from_missing_base_is_empty():
    assert _facts_from_base(None) == []
    assert _facts_from_base({"facts": []}) == []
