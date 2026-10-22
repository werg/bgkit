from pathlib import Path

from bgkit2.agent.tools import ToolRegistry
from bgkit2.agent.web import Page, WebCorpus


def test_offline_web_tools_search_open_find(repo: Path):
    corpus = WebCorpus([
        Page("p1", "Eiffel Tower", "https://x/eiffel", "The Eiffel Tower is in Paris. It was completed in 1889."),
        Page("p2", "Tower Bridge", "https://x/bridge", "Tower Bridge crosses the Thames in London."),
    ])
    reg = ToolRegistry(str(repo))
    assert "web_search" not in reg.names
    reg.attach_web(corpus)
    r = reg.call("web_search", {"query": "when was the eiffel tower completed"})
    assert r.ok and r.compressible and r.text.startswith("[p1] Eiffel Tower")
    r = reg.call("web_find", {"pattern": "1889"})
    assert not r.ok  # nothing opened yet
    r = reg.call("web_open", {"id": "p1", "focus": "completion year"})
    assert r.ok and "completed in 1889" in r.text and "completion year" in r.prompt
    # web_find (default, encoded): the WHOLE page as a slot, prompted with the pattern + focus
    r = reg.call("web_find", {"pattern": "1889", "focus": "completion year"})
    assert r.ok and r.compressible and r.tool == "web_find"
    assert r.text == reg.call("web_open", {"id": "p1"}).text  # the page web_open returns
    assert r.prompt == "find 1889 in Eiffel Tower: completion year"
    assert reg.call("web_find", {"pattern": "1889"}).prompt == "find 1889 in Eiffel Tower"
    # another page by id (the page opened last stays p1)
    r = reg.call("web_find", {"pattern": "thames", "id": "p2"})
    assert r.compressible and r.prompt == "find thames in Tower Bridge" and "Thames" in r.text
    miss = reg.call("web_find", {"pattern": "1901"})
    assert miss.ok and not miss.compressible and "not found" in miss.text
    assert not reg.call("web_find", {"pattern": "x", "id": "nope"}).ok
    assert not reg.call("web_open", {"id": "nope"}).ok
    spec = {s["function"]["name"]: s["function"] for s in reg.specs()}["web_find"]
    assert set(spec["parameters"]["properties"]) == {"pattern", "id", "focus"}
    assert reg.spec("web_find").compressible


def test_web_find_verbatim_control_arm(repo: Path):
    corpus = WebCorpus([Page("p1", "Eiffel Tower", "u",
                             "The tower is in Paris. It was completed in 1889.")])
    reg = ToolRegistry(str(repo), web_find_mode="verbatim")
    reg.attach_web(corpus)
    assert not reg.call("web_find", {"pattern": "1889"}).ok  # nothing opened yet
    reg.call("web_open", {"id": "p1"})
    r = reg.call("web_find", {"pattern": "1889"})
    assert r.ok and not r.compressible and r.prompt == "" and r.text == "It was completed in 1889."
    spec = {s["function"]["name"]: s["function"] for s in reg.specs()}["web_find"]
    assert set(spec["parameters"]["properties"]) == {"pattern"}
    assert "verbatim" in spec["description"]
    assert not reg.spec("web_find").compressible
    try:
        ToolRegistry(str(repo), web_find_mode="exact")
    except ValueError:
        pass
    else:
        raise AssertionError("bad web_find_mode accepted")
