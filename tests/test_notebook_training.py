"""Exercise training-only selection and submission generation on synthetic data."""

import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class NotebookTrainingTests(unittest.TestCase):
    def test_selection_finishes_before_test_features_are_loaded(self):
        notebook = json.loads((ROOT / "run.ipynb").read_text())
        rng = np.random.default_rng(72)
        x = np.column_stack(
            [rng.integers(1, 3, 720), rng.uniform(18, 80, 720), rng.normal(size=720)]
        )
        y = np.where(rng.random(600) < 0.2 + 0.15 * (x[:600, 0] == 1), 1, -1)
        original_open, original_io_open, original_stat = (
            builtins.open,
            io.open,
            Path.stat,
        )
        namespace = {}
        prediction_started = False

        def check_access(filename):
            if isinstance(filename, (str, os.PathLike)):
                path = Path(filename)
                if path.name in {"x_test.csv", "x_test.npz"}:
                    self.assertTrue(prediction_started)
                self.assertNotIn("experiments", path.parts)

        def guarded_open(filename, *args, **kwargs):
            check_access(filename)
            return original_open(filename, *args, **kwargs)

        def guarded_io_open(filename, *args, **kwargs):
            check_access(filename)
            return original_io_open(filename, *args, **kwargs)

        def guarded_stat(path, *args, **kwargs):
            check_access(path)
            return original_stat(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data/dataset"
            data_dir.mkdir(parents=True)
            for name, rows, ids in (
                ("x_train.csv", x[:600], np.arange(600)),
                ("x_test.csv", x[600:], np.arange(600, 720)),
            ):
                np.savetxt(
                    data_dir / name,
                    np.column_stack([ids, rows]),
                    delimiter=",",
                    header="Id,SEX,_AGE80,measure",
                    comments="",
                )
            np.savetxt(
                data_dir / "y_train.csv",
                np.column_stack([np.arange(600), y]),
                delimiter=",",
                header="Id,Prediction",
                comments="",
                fmt="%d",
            )
            previous_directory = Path.cwd()
            os.chdir(directory)
            try:
                with patch("builtins.open", guarded_open), patch(
                    "io.open", guarded_io_open
                ), patch.object(Path, "stat", guarded_stat), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    for cell in notebook["cells"]:
                        if cell["cell_type"] != "code":
                            continue
                        stage = cell["metadata"]["tags"][0]
                        if stage == "load_test":
                            self.assertIn("best_params", namespace)
                            self.assertIn("best_blend", namespace)
                            prediction_started = True
                        exec("".join(cell["source"]), namespace)
                        if stage == "search_space":
                            namespace.update(
                                l1_values=(0.0, 1e-3),
                                l2_values=(1e-3,),
                                max_iters=30,
                                n_folds=2,
                            )
                    submission = np.loadtxt("submission.csv", delimiter=",", skiprows=1)
                    np.testing.assert_array_equal(submission[:, 0], np.arange(600, 720))
                    np.testing.assert_array_equal(
                        submission[:, 1], namespace["test_predictions"]
                    )
                    self.assertTrue(np.isin(submission[:, 1], [-1, 1]).all())
            finally:
                os.chdir(previous_directory)


if __name__ == "__main__":
    unittest.main()
