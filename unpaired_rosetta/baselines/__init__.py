"""The baselines of the paper, one module each. All follow the ``Aligner`` interface of ``unpaired_rosetta.evaluation``
(``fit(samples_x, samples_y, paired_x, paired_y, randomness)`` and ``similarity(queries_x, keys_y)``).

Unpaired: ``vec2vec`` (Jha et al.), ``mini_vec2vec`` (Dar). Few pairs: ``linear`` and ``orthogonal`` maps (Maiorca
et al.), ``asif`` (Norelli et al.), ``local_cka`` (Maniparambil et al.), ``sue`` (Yacobi et al.), ``structure``
(Groeger et al.), ``sotalign`` (Roschmann et al.). Single cell: ``scot_plus`` (Baker et al.). Controls: ``identity``.
``tests/<module>/test_equivalence.py`` compares each port with the official code.
"""
