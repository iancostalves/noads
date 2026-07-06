# Copyright 2025 ISAE-SUPAERO, https://www.isae-supaero.fr/en/
# Copyright 2025 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
# International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
# Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Sort key used by sphinx-gallery (importable, hence picklable)."""


class RunScriptsFirstKey:
    """Order gallery entries: reproduction runs first, then comparison plots."""

    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __call__(self, fname):
        return (0 if fname.startswith("run_") else 1, fname)

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
