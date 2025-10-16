if __name__ == "__main__":
    from randomgen.tests.data.compute_hashes import (
        final_configurations,
        hash_configuration,
    )

    computed_hashes = {}
    for key in final_configurations:
        computed_hashes[key] = hash_configuration(final_configurations[key])

    import black

    fm = black.FileMode(
        {
            black.TargetVersion.PY38,
            black.TargetVersion.PY39,
            black.TargetVersion.PY310,
            black.TargetVersion.PY311,
            black.TargetVersion.PY312,
            black.TargetVersion.PY313,
        }
    )
    out = black.format_file_contents(
        "known_hashes = " + str(dict(computed_hashes)),
        fast=True,
        mode=fm,
    )
    with open("stable_hashes.py", "w") as results:
        results.write(out)
