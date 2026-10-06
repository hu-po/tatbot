import tatbot_ink


def test_program_format():
    assert (tatbot_ink.FORMAT, tatbot_ink.VERSION) == ("tatbot-program", 2)
