"""
Correction model for 340mm coarse localization to match 260mm fine accuracy.

Based on data from: two-stage-20260916_153939
Sample size: 29 holes with both coarse and fine results
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CoarseToFineCorrector:
    """
    Corrects 340mm coarse localization results to approximate 260mm fine accuracy.
    """

    def __init__(self, model_data_path: Optional[str] = None):
        """
        Initialize corrector with training data.

        Args:
            model_data_path: Path to correction model JSON data.
                           If None, uses default simple correction.
        """
        self.model_data = None
        self.simple_correction = np.array([-0.038, -0.509, -0.046])  # Mean dX, dY, dZ

        if model_data_path and Path(model_data_path).exists():
            with open(model_data_path, 'r', encoding='utf-8') as f:
                self.model_data = json.load(f)
            print(f"[CorrectionModel] Loaded model data from {model_data_path}")
            print(f"[CorrectionModel] Sample count: {self.model_data.get('sample_count')}")
        else:
            print(f"[CorrectionModel] Using simple statistical correction")

    def correct_simple(self, coarse_center_base_mm: List[float]) -> np.ndarray:
        """
        Apply simple statistical correction (mean offset).

        Args:
            coarse_center_base_mm: Coarse center from 340mm [x, y, z]

        Returns:
            Corrected center [x, y, z]
        """
        coarse = np.array(coarse_center_base_mm)
        corrected = coarse + self.simple_correction
        return corrected

    def correct_adaptive(
        self,
        coarse_center_base_mm: List[float],
        quality_metrics: Dict
    ) -> np.ndarray:
        """
        Apply adaptive correction based on quality metrics.

        Args:
            coarse_center_base_mm: Coarse center from 340mm [x, y, z]
            quality_metrics: Dictionary with keys:
                - ring_points_median: Number of ring points
                - tracking_distance_p95_px: Tracking stability
                - plane_rmse_mm: Plane fit quality
                - boundary_class: 'edge' or 'interior'

        Returns:
            Corrected center [x, y, z]
        """
        coarse = np.array(coarse_center_base_mm)

        # Base correction
        correction = self.simple_correction.copy()

        # Quality-based scaling
        # Lower quality -> larger uncertainty -> apply more conservative correction
        ring_points = quality_metrics.get('ring_points_median', 4000)
        tracking_p95 = quality_metrics.get('tracking_distance_p95_px', 1.0)
        plane_rmse = quality_metrics.get('plane_rmse_mm', 1.5)
        boundary = quality_metrics.get('boundary_class', 'interior')

        # Quality score (0-1, higher is better)
        quality_score = self._compute_quality_score(
            ring_points, tracking_p95, plane_rmse, boundary
        )

        # For high-quality measurements, apply full correction
        # For low-quality measurements, apply reduced correction (more conservative)
        correction_scale = 0.5 + 0.5 * quality_score

        corrected = coarse + correction * correction_scale

        return corrected

    def _compute_quality_score(
        self,
        ring_points: int,
        tracking_p95: float,
        plane_rmse: float,
        boundary: str
    ) -> float:
        """
        Compute quality score from 0 (worst) to 1 (best).
        """
        score = 1.0

        # Ring points contribution (worse if < 3000)
        if ring_points < 2500:
            score *= 0.5
        elif ring_points < 3000:
            score *= 0.7
        elif ring_points < 3500:
            score *= 0.85

        # Tracking stability contribution (worse if > 2.0)
        if tracking_p95 > 3.0:
            score *= 0.5
        elif tracking_p95 > 2.0:
            score *= 0.7
        elif tracking_p95 > 1.5:
            score *= 0.85

        # Plane RMSE contribution (worse if > 1.5)
        if plane_rmse > 2.0:
            score *= 0.7
        elif plane_rmse > 1.5:
            score *= 0.85

        # Boundary class (edge holes are less reliable)
        if boundary == 'edge':
            score *= 0.9

        return max(0.0, min(1.0, score))

    def correct_with_confidence(
        self,
        coarse_center_base_mm: List[float],
        quality_metrics: Dict
    ) -> Tuple[np.ndarray, float, str]:
        """
        Apply correction and return confidence level.

        Args:
            coarse_center_base_mm: Coarse center from 340mm [x, y, z]
            quality_metrics: Quality metrics dict

        Returns:
            Tuple of (corrected_center, confidence, recommendation)
            - corrected_center: Corrected [x, y, z]
            - confidence: 0-1, higher means more reliable correction
            - recommendation: String recommendation ('use_corrected', 'recapture', 'use_original')
        """
        corrected = self.correct_adaptive(coarse_center_base_mm, quality_metrics)

        quality_score = self._compute_quality_score(
            quality_metrics.get('ring_points_median', 4000),
            quality_metrics.get('tracking_distance_p95_px', 1.0),
            quality_metrics.get('plane_rmse_mm', 1.5),
            quality_metrics.get('boundary_class', 'interior')
        )

        # Confidence is based on quality and expected correction magnitude
        # High quality + small correction = high confidence
        # Low quality + large correction = low confidence
        confidence = quality_score

        # Recommendation
        if confidence > 0.8:
            recommendation = 'use_corrected'
        elif confidence > 0.5:
            recommendation = 'use_corrected_with_caution'
        else:
            recommendation = 'recapture_at_lower_height'

        return corrected, confidence, recommendation

    def evaluate_on_validation_set(self) -> Dict:
        """
        Evaluate correction model on the training data (for validation).

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.model_data:
            return {'error': 'No model data loaded'}

        corrections_data = self.model_data.get('corrections', [])

        # Test simple correction
        simple_errors = []
        adaptive_errors = []

        for item in corrections_data:
            coarse = item['coarse']
            fine = item['fine']

            # Simple correction
            corrected_simple = self.correct_simple(coarse)
            error_simple = np.linalg.norm(corrected_simple - np.array(fine))
            simple_errors.append(error_simple)

            # Adaptive correction
            quality_metrics = {
                'ring_points_median': item.get('coarse_ring_points'),
                'tracking_distance_p95_px': item.get('coarse_tracking_p95'),
                'plane_rmse_mm': item.get('coarse_plane_rmse'),
                'boundary_class': item.get('boundary'),
            }
            corrected_adaptive = self.correct_adaptive(coarse, quality_metrics)
            error_adaptive = np.linalg.norm(corrected_adaptive - np.array(fine))
            adaptive_errors.append(error_adaptive)

        # Original error (without correction)
        original_errors = [item['total'] for item in corrections_data]

        return {
            'sample_count': len(corrections_data),
            'original_mean_error_mm': float(np.mean(original_errors)),
            'simple_correction_mean_error_mm': float(np.mean(simple_errors)),
            'adaptive_correction_mean_error_mm': float(np.mean(adaptive_errors)),
            'simple_improvement_percent': float((1 - np.mean(simple_errors)/np.mean(original_errors)) * 100),
            'adaptive_improvement_percent': float((1 - np.mean(adaptive_errors)/np.mean(original_errors)) * 100),
        }


def main():
    """Test the correction model."""
    import sys

    # Load model
    model_path = Path(__file__).parent.parent / 'data' / 'coarse_to_fine_correction_model_data.json'
    corrector = CoarseToFineCorrector(str(model_path))

    # Evaluate
    eval_results = corrector.evaluate_on_validation_set()

    print("\n" + "=" * 80)
    print("CORRECTION MODEL EVALUATION")
    print("=" * 80)
    print(f"\nSample count: {eval_results.get('sample_count')}")
    print(f"\nOriginal error (340mm coarse): {eval_results.get('original_mean_error_mm'):.3f} mm")
    print(f"After simple correction:        {eval_results.get('simple_correction_mean_error_mm'):.3f} mm")
    print(f"After adaptive correction:      {eval_results.get('adaptive_correction_mean_error_mm'):.3f} mm")
    print(f"\nImprovement:")
    print(f"  Simple correction:  {eval_results.get('simple_improvement_percent'):.1f}%")
    print(f"  Adaptive correction: {eval_results.get('adaptive_improvement_percent'):.1f}%")

    # Example usage
    print("\n" + "=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)

    example_coarse = [700.0, 150.0, 35.0]
    example_quality = {
        'ring_points_median': 3500,
        'tracking_distance_p95_px': 1.5,
        'plane_rmse_mm': 1.2,
        'boundary_class': 'edge'
    }

    corrected, confidence, recommendation = corrector.correct_with_confidence(
        example_coarse, example_quality
    )

    print(f"\nInput coarse (340mm): {example_coarse}")
    print(f"Corrected estimate:   {corrected.tolist()}")
    print(f"Confidence:           {confidence:.2f}")
    print(f"Recommendation:       {recommendation}")


if __name__ == '__main__':
    main()
