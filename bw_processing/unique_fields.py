from .errors import InconsistentFields, NonUnique
import pandas as pd


def greedy_set_cover(data, exclude=None, raise_error=True):
    """Find unique set of attributes that uniquely identifies each element in ``data``.

    Feature selection is a well known problem, and is analogous to the `set cover problem <https://en.wikipedia.org/wiki/Set_cover_problem>`__, for which there is a `well known heuristic <https://en.wikipedia.org/wiki/Set_cover_problem#Greedy_algorithm>`__.

    Example:

        data = [
            {'a': 1, 'b': 2, 'c': 3},
            {'a': 2, 'b': 2, 'c': 3},
            {'a': 1, 'b': 2, 'c': 4},
        ]
        greedy_set_cover(data)
        >>> {'a', 'c'}

    Args:
        data (iterable): List of dictionaries with the same fields.
        exclude (iterable): Fields to exclude during search for uniqueness. ``id`` is Always excluded.

    Returns:
        Set of attributes (strings)

    Raises:
        NonUnique: The given fields are not enough to ensure uniqueness.

    Note that ``NonUnique`` is not raised if ``raise_error`` is false.

    """
    exclude = set([]) if exclude is None else set(exclude)
    exclude.add("id")

    fields = {field for obj in data for field in obj if field not in exclude}
    chosen = set([])

    def values_for_fields(data, exclude, fields):
        return sorted(
            [(len({obj[field] for obj in data}), field) for field in fields],
            reverse=True,
        )

    def coverage(data, chosen):
        return len({tuple([obj[field] for field in chosen]) for obj in data})

    while coverage(data, chosen) != len(data):
        if not fields:
            if raise_error:
                raise NonUnique
            else:
                break
        next_field = values_for_fields(data, exclude, fields)[0][1]
        fields.remove(next_field)
        chosen.add(next_field)

    return chosen


def as_unique_attributes_dataframe(df, exclude=None, include=None, raise_error=False):
    assert isinstance(df, pd.DataFrame)
    include = greedy_set_cover(
        df.reset_index().to_dict("records"), exclude=exclude, raise_error=raise_error
    ).union(include or [])
    to_drop = [col for col in df.columns if col not in include]
    return df.drop(columns=to_drop)


def as_unique_attributes(data, exclude=None, include=None, raise_error=False):
    """Format ``data`` as unique set of attributes and values for use in ``create_processed_datapackage``.

    Each element in ``data`` must have the attribute ``id``, and it must be unique. However, the field "id" is not used in selecting the unique set of attributes.

    If no set of attributes is found that uniquely identifies all features is found, all fields are used. To have this case raise an error, pass ``raise_error=True``.

        data = [
            {},
        ]

    Args:
        data (iterable): List of dictionaries with the same fields.
        exclude (iterable): Fields to exclude during search for uniqueness. ``id`` is Always excluded.
        include (iterable): Fields to include when returning, even if not unique

    Returns:
        (list of field names as strings, dictionary of data ids to values for given field names)

    Raises:
        InconsistentFields: Not all features provides all fields.
    """
    include = set(include or [])
    fields = greedy_set_cover(data, exclude=exclude, raise_error=raise_error)

    if len({tuple(sorted(obj.keys())) for obj in data}) > 1:
        raise InconsistentFields

    def formatter(obj, fields, include):
        return {
            key: value
            for key, value in obj.items()
            if (key in fields or key in include or key == "id")
        }

    return (
        fields.union(include).union({"id"}),
        [formatter(obj, fields, include) for obj in data],
    )
