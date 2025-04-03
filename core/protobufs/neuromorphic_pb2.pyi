from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SpikeInput(_message.Message):
    __slots__ = ("encoded_spikes", "temporal_window", "neuron_profile", "context_tags")
    class ContextTagsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ENCODED_SPIKES_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_WINDOW_FIELD_NUMBER: _ClassVar[int]
    NEURON_PROFILE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_TAGS_FIELD_NUMBER: _ClassVar[int]
    encoded_spikes: bytes
    temporal_window: int
    neuron_profile: str
    context_tags: _containers.ScalarMap[str, str]
    def __init__(self, encoded_spikes: _Optional[bytes] = ..., temporal_window: _Optional[int] = ..., neuron_profile: _Optional[str] = ..., context_tags: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SpikeOutput(_message.Message):
    __slots__ = ("activations", "energy_usage", "processing_metadata")
    class EnergyUsageEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: float
        def __init__(self, key: _Optional[int] = ..., value: _Optional[float] = ...) -> None: ...
    ACTIVATIONS_FIELD_NUMBER: _ClassVar[int]
    ENERGY_USAGE_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_METADATA_FIELD_NUMBER: _ClassVar[int]
    activations: _containers.RepeatedScalarFieldContainer[float]
    energy_usage: _containers.ScalarMap[int, float]
    processing_metadata: str
    def __init__(self, activations: _Optional[_Iterable[float]] = ..., energy_usage: _Optional[_Mapping[int, float]] = ..., processing_metadata: _Optional[str] = ...) -> None: ...

class TrainingData(_message.Message):
    __slots__ = ("samples", "targets", "training_profile")
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    TRAINING_PROFILE_FIELD_NUMBER: _ClassVar[int]
    samples: _containers.RepeatedCompositeFieldContainer[SpikeInput]
    targets: _containers.RepeatedCompositeFieldContainer[SpikeOutput]
    training_profile: str
    def __init__(self, samples: _Optional[_Iterable[_Union[SpikeInput, _Mapping]]] = ..., targets: _Optional[_Iterable[_Union[SpikeOutput, _Mapping]]] = ..., training_profile: _Optional[str] = ...) -> None: ...

class TrainingResult(_message.Message):
    __slots__ = ("loss", "accuracy", "resources_used")
    LOSS_FIELD_NUMBER: _ClassVar[int]
    ACCURACY_FIELD_NUMBER: _ClassVar[int]
    RESOURCES_USED_FIELD_NUMBER: _ClassVar[int]
    loss: float
    accuracy: float
    resources_used: NeuroResourceUsage
    def __init__(self, loss: _Optional[float] = ..., accuracy: _Optional[float] = ..., resources_used: _Optional[_Union[NeuroResourceUsage, _Mapping]] = ...) -> None: ...

class ResourceQuery(_message.Message):
    __slots__ = ("component_id", "history_window")
    COMPONENT_ID_FIELD_NUMBER: _ClassVar[int]
    HISTORY_WINDOW_FIELD_NUMBER: _ClassVar[int]
    component_id: str
    history_window: int
    def __init__(self, component_id: _Optional[str] = ..., history_window: _Optional[int] = ...) -> None: ...

class NeuroResourceUsage(_message.Message):
    __slots__ = ("energy_joules", "compute_seconds", "spike_ops", "memory_bytes")
    ENERGY_JOULES_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SPIKE_OPS_FIELD_NUMBER: _ClassVar[int]
    MEMORY_BYTES_FIELD_NUMBER: _ClassVar[int]
    energy_joules: float
    compute_seconds: float
    spike_ops: int
    memory_bytes: int
    def __init__(self, energy_joules: _Optional[float] = ..., compute_seconds: _Optional[float] = ..., spike_ops: _Optional[int] = ..., memory_bytes: _Optional[int] = ...) -> None: ...
