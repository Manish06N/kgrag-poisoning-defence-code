from kgrag import gcr_stage2


def beam(path, answer):
    return f"# Reasoning Path:\n{path}\n# Answer:\n{answer}"


def test_extracts_paths_in_beam_order_and_dedupes():
    beams = [beam("Jamaica -> languages -> Jamaican English", "Jamaican English"),
             beam("Jamaica -> languages -> Jamaican Creole", "Jamaican Creole"),
             beam("Jamaica -> languages -> Jamaican English", "Jamaican English")]     # same path, another beam
    assert gcr_stage2.beam_paths(beams) == ["Jamaica -> languages -> Jamaican English", "Jamaica -> languages -> Jamaican Creole"]


def test_multi_hop_paths_and_malformed_beams():
    beams = [beam("A -> r1 -> B -> r2 -> C", "C"), "no markers here", "# Reasoning Path:\n\n# Answer:\nX", ""]
    assert gcr_stage2.beam_paths(beams) == ["A -> r1 -> B -> r2 -> C"]                   # empty path and junk beams are skipped


def test_no_beams_gives_no_paths():
    assert gcr_stage2.beam_paths([]) == []
