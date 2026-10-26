- An `xtrax.Struct` subclass that is also decorated with `@dataclass` raises `TypeError` at
  construction. The second decorator made the class mutable and gave it a field-wise `==`
  that returns an array for array fields. (`@dataclass(frozen=True)` was already refused by
  `dataclasses` at class definition.)
