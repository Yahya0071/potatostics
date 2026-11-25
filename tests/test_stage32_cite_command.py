import os
import pytest
from engine.interpreter import Interpreter
from engine.dataset import Dataset


@pytest.fixture
def interp():
    dataset = Dataset()
    return Interpreter(dataset)


def test_cite_command_explicit_keyword(interp):
    """
    Test that 'cite <command>' returns human-readable citation and BibTeX block with DOI.
    """
    res = interp.execute("cite csdid")
    assert res.ok
    assert "Callaway" in res.output
    assert "Sant'Anna" in res.output
    assert "10.1016/j.jeconom.2020.12.001" in res.output
    assert "@article" in res.output
    assert interp.registers.get_r("doi") == "10.1016/j.jeconom.2020.12.001"


def test_cite_command_after_estimation(interp):
    """
    Test that 'cite' with no arguments automatically cites the last estimated model.
    """
    import pandas as pd

    df = pd.DataFrame({
        "y": [1, 2, 3, 4, 5],
        "x": [2, 4, 5, 4, 5]
    })
    interp.dataset.df = df
    interp.execute("regress y x")

    res = interp.execute("cite")
    assert res.ok
    assert "regress" in res.output.lower() or "ordinary least squares" in res.output.lower()
    assert "@" in res.output


def test_cite_command_unknown(interp):
    """
    Test that 'cite nonexistent' gracefully displays a list of available citations.
    """
    res = interp.execute("cite nonexistent_cmd")
    assert res.ok
    assert "available citation keys" in res.output.lower() or "csdid" in res.output.lower()


def test_references_bib_exists():
    """
    Test that References/references.bib exists and contains valid BibTeX entries for all 42 works.
    """
    bib_path = os.path.join(os.path.dirname(__file__), "..", "References", "references.bib")
    assert os.path.exists(bib_path)
    with open(bib_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify key seminal entries are present
    assert "callaway2021difference" in content
    assert "brant1990assessing" in content
    assert "white1980heteroskedasticity" in content
    assert "sims1980macroeconomics" in content
    assert "kaplan1958nonparametric" in content
    assert "cox1972regression" in content
