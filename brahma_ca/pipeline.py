from __future__ import annotations
from brahma_ca.state import CAState
from brahma_ca.engine import CAEngine
from brahma_ca.detector import Detector
from brahma_ca.labeler import Labeler
from brahma_ca.rules.base import CARule


class Pipeline:
    """
    Full identdynamics integration: fourier dict → post_run() arguments.

    Usage::

        from identdynamics import Client
        from ca import Pipeline, EdgeDetectionRule, AnomalyDetectionRule, GroupingRule

        client = Client(base_url, token)
        fourier = client.fourier_for_file(file_id)

        pipe = Pipeline(
            rule=EdgeDetectionRule() | AnomalyDetectionRule() | GroupingRule(),
            steps=1,
            threshold=0.45,
        )
        result = pipe.run(fourier)
        client.post_run(target={"kind": "file", "id": file_id}, **result)
    """

    def __init__(
        self,
        rule: CARule,
        model_name: str = "ca_pipeline",
        steps: int = 1,
        threshold: float = 0.5,
        score_mode: str = "max",
        label: bool = True,
        normalize: bool = True,
        binarize_input: bool = True,
        threshold_pct: float = 80.0,
        save_history: bool = False,
    ):
        self.rule = rule
        self.model_name = model_name
        self.steps = steps
        self.threshold = threshold
        self.score_mode = score_mode
        self.label = label
        self.normalize = normalize
        self.binarize_input = binarize_input
        self.threshold_pct = threshold_pct
        self.save_history = save_history

    def run(self, fourier: dict) -> dict:
        ca = CAState.from_fourier(fourier)

        engine = CAEngine(
            rule=self.rule,
            steps=self.steps,
            save_history=self.save_history,
            binarize_input=self.binarize_input,
            threshold_pct=self.threshold_pct,
        )
        ca = engine.evolve(ca)

        detector = Detector(
            score_mode=self.score_mode,
            threshold=self.threshold,
            normalize=self.normalize,
        )
        scores = detector.scores(ca)
        dets = detector.detections(ca)

        if self.label:
            labeler = Labeler()
            dets = labeler.label_detections(ca, dets)

        return {
            "model_name": self.model_name,
            "scores": scores.tolist(),
            "threshold": self.threshold,
            "detections": dets,
            "stft": fourier.get("stft", {}),
            "_ca": ca,      # available for inspection; stripped before post_run
        }

    def post(self, client, target: dict | str, fourier: dict) -> dict:
        """Convenience: run + post_run in one call. Returns API response."""
        result = self.run(fourier)
        ca = result.pop("_ca")
        client.post_run(target=target, **result)
        result["_ca"] = ca
        return result
