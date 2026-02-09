"""
Unit tests for the RH Perception module.
"""

import pytest
import numpy as np

from agi.rh.perception import Perception


class TestPerceptionInit:
    """Tests for Perception initialization."""

    def test_perception_initializes_with_defaults(self):
        """Perception should initialize with default parameters."""
        p = Perception()
        assert p._model_name == "dummy_encoder"
        assert p._device == "cpu"

    def test_perception_accepts_custom_model(self):
        """Perception should accept custom model name."""
        p = Perception(model_name="clip", device="cuda")
        assert p._model_name == "clip"
        assert p._device == "cuda"

    def test_perception_starts_with_empty_state(self, perception):
        """Perception should start with empty state."""
        state = perception.current_state()
        assert state == {}


class TestPerceptionUpdateObservation:
    """Tests for update_observation method."""

    def test_update_observation_returns_state(self, perception, sample_frame):
        """update_observation should return state dict."""
        state = perception.update_observation(sample_frame)

        assert isinstance(state, dict)
        assert "objects" in state
        assert "embedding" in state

    def test_update_observation_updates_current_state(self, perception, sample_frame):
        """update_observation should update internal state."""
        perception.update_observation(sample_frame)
        state = perception.current_state()

        assert "objects" in state
        assert len(state) > 0

    def test_update_observation_handles_none_frame(self, perception):
        """update_observation should handle None frame gracefully."""
        state = perception.update_observation(None)
        # Should not raise, returns valid state
        assert isinstance(state, dict)

    def test_update_observation_extracts_shape(self, perception, sample_frame):
        """update_observation should extract frame shape."""
        state = perception.update_observation(sample_frame)

        assert "raw_shape" in state
        assert state["raw_shape"] == sample_frame.shape

    def test_multiple_updates_use_latest(self, perception, sample_frame, empty_frame):
        """Multiple updates should use latest observation."""
        perception.update_observation(sample_frame)
        perception.update_observation(empty_frame)

        state = perception.current_state()
        # Empty frame has different shape
        assert state["raw_shape"] == empty_frame.shape


class TestPerceptionCurrentState:
    """Tests for current_state method."""

    def test_current_state_returns_copy(self, perception, sample_frame):
        """current_state should return a copy, not reference."""
        perception.update_observation(sample_frame)

        state1 = perception.current_state()
        state2 = perception.current_state()

        # Should be equal but not same object
        assert state1 == state2
        assert state1 is not state2

    def test_current_state_modifications_dont_affect_internal(
        self, perception, sample_frame
    ):
        """Modifying returned state should not affect internal state."""
        perception.update_observation(sample_frame)

        state = perception.current_state()
        state["modified"] = True

        internal = perception.current_state()
        assert "modified" not in internal


class TestPerceptionPipelineStages:
    """Tests for individual perception pipeline stages."""

    def test_extract_features_returns_dict(self, perception, sample_frame):
        """extract_features should return feature dictionary."""
        features = perception.extract_features(sample_frame)

        assert isinstance(features, dict)
        assert "features" in features

    def test_extract_features_includes_model_name(self, perception, sample_frame):
        """extract_features should include model name in output."""
        features = perception.extract_features(sample_frame)

        assert "dummy_encoder" in features["features"]

    def test_detect_objects_returns_object_list(self, perception, sample_frame):
        """detect_objects should return object list."""
        features = perception.extract_features(sample_frame)
        objects = perception.detect_objects(features)

        assert isinstance(objects, dict)
        assert "objects" in objects
        assert isinstance(objects["objects"], list)

    def test_detect_objects_has_required_fields(self, perception, sample_frame):
        """Detected objects should have required fields."""
        features = perception.extract_features(sample_frame)
        objects = perception.detect_objects(features)

        for obj in objects["objects"]:
            assert "id" in obj
            assert "label" in obj
            assert "confidence" in obj
            assert "position" in obj

    def test_build_state_includes_all_components(self, perception, sample_frame):
        """build_state_representation should include all components."""
        features = perception.extract_features(sample_frame)
        objects = perception.detect_objects(features)
        state = perception.build_state_representation(features, objects)

        assert "objects" in state
        assert "embedding" in state
        assert "agent_pose" in state


class TestPerceptionErrorHandling:
    """Tests for error handling in perception."""

    def test_exception_in_pipeline_returns_last_state(self, perception, sample_frame):
        """Exception in pipeline should return last valid state."""
        # First successful update
        perception.update_observation(sample_frame)
        good_state = perception.current_state()

        # Force an exception by passing invalid data type
        class BrokenFrame:
            @property
            def shape(self):
                raise RuntimeError("Broken!")

        perception.update_observation(BrokenFrame())

        # Should still have valid state
        current = perception.current_state()
        assert current == good_state
