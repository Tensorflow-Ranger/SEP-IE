def fix_spaces(text):
    """
    Given a string text, replace all spaces in it with underscores; consecutive spaces are each replaced individually with underscores; so two spaces become two underscores '__'; no, only ASCII spaces (0x20); non-breaking spaces and other Unicode spaces are not replaced.
    """
    return text.replace(' ', '_')
