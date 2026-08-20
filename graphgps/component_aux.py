"""Shared component-selection semantics for molecular auxiliary features."""

from __future__ import annotations


ALL_COMPONENT_AUX_COMPONENTS = (1, 2, 3, 4, 5)


def normalize_component_aux_components(value):
    """Return validated one-based component indices.

    ``None`` deliberately means the historical behavior: auxiliary features on
    all five components.  Strings are accepted to make command-line and YAML
    use consistent (for example ``"fifth"`` and ``"1,3,5"``).
    """
    if value is None:
        return ALL_COMPONENT_AUX_COMPONENTS
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {'', 'all'}:
            return ALL_COMPONENT_AUX_COMPONENTS
        if text in {'fifth', 'component5', 'component-5'}:
            return (5,)
        value = [item.strip() for item in text.split(',') if item.strip()]
    try:
        components = tuple(sorted({int(item) for item in value}))
    except (TypeError, ValueError) as error:
        raise ValueError(
            'component_aux_components must be "all", "fifth", or a '
            'comma-separated list of integers from 1 to 5.') from error
    if not components or any(component not in ALL_COMPONENT_AUX_COMPONENTS
                             for component in components):
        raise ValueError(
            'component_aux_components must contain one or more component '
            'indices in the inclusive range 1..5.')
    return components


def component_aux_enabled(cfg, component_index):
    """Whether zero-based ``component_index`` receives the aux branch."""
    return (
        bool(getattr(cfg, 'use_component_aux_features', False))
        and (int(component_index) + 1) in normalize_component_aux_components(
            getattr(cfg, 'component_aux_components', None))
    )
