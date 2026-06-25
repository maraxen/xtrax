"""Identity-hashable wrapper for batch plan objects."""


# Sentinel wrapper to make BatchPlan hashable by identity
class _BatchPlanWrapper:
    """Wrapper to make BatchPlan hashable by identity (id-based) for use with eqx.filter_jit."""

    __slots__ = ("_id", "_plan")

    def __init__(self, plan):
        self._plan = plan
        self._id = id(plan) if plan is not None else None

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if not isinstance(other, _BatchPlanWrapper):
            return False
        return self._id == other._id

    def __getattr__(self, name):
        if name in ("_plan", "_id"):
            return object.__getattribute__(self, name)
        return getattr(self._plan, name)

    def __repr__(self):
        return f"_BatchPlanWrapper({self._plan!r})"
