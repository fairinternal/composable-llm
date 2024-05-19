"""
Evaluation I/O class

License
-------
This source code is licensed under the MIT license found in the LICENSE file
in the root directory of this source tree.

@ 2024,
"""


class EvaluationIO:

    def __init__(self, file_path, overwrite=False, meaning=None):
        """
        I/O object to save evalutions.

        Parameters
        ----------
        file_path: str
            File path where to save the evluation.
        overwrite: bool
            Wether to overwrite potential existing file.
        meaning: dict
            Meaning of the different eval coordinates.
        """

        self.meaning = meaning
        self.file_path = file_path

        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as f:
                f.write("timestamp")
                for eval in self.meaning:
                    f.write(",")
                    f.write(eval)
                f.write("\n")

    def __call__(self, timestamp, evals):
        """
        Report eval at a given timestamp.

        Parameters
        ----------
        timestamp: int
            Timestamp of the evaluation.
        evals: np.ndarray
            Evaluation vector.
        """
        with open(self.file_path, "a") as f:
            f.write(str(timestamp))
            for eval in evals:
                f.write(",")
                f.write(str(eval.item()))
            f.write("\n")
